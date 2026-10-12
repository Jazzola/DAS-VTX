"""Interpret stacked Virtual Shot Gathers in the frequency-velocity domain

The workflow includes
1. select the requested offsets and correlation lags;
2. optionally normalize the stacked VSG;
3. optionally enhance coherent arrivals with slowness-dependent semblance;
4. compute the frequency-phase-velocity (f-v) image; and
5. optionally construct an aperture-stability mask from progressively shorter
   VSG apertures and apply it to the full-aperture dispersion image (masking)

Diagnostic figures are written alongside the numerical dispersion products so
that every processing choice can be visually inspected.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from config import (
    PROCESSED_DIR,
    coherence_enhancing,
    decim,
    disp_masking,
    dx,
    freqs,
    lags_to_keep,
    max_cut_dist,
    min_cut_dist,
    n_processes,
    offsets_to_keep,
    slw_list,
    stack_norm,
    step_factor,
    twin,
    vels,
    xwin,
)
from utils import Coherence_Enhancement

# Plot limits describe the diagnostic presentation only; they do not restrict
# the numerical dispersion grids defined in the configuration files.
DISP_FREQ_LIM = (0.7, 20.0)
DISP_VEL_LIM = (250.0, 2000.0)


def _prepare_output_directories(stack_name: str):
    """Create and return the figure directories used for one VSG stack
    """
    fig_root = PROCESSED_DIR / "FIG"
    stack_dir = fig_root / "Stack"
    disp_dir = fig_root / "Disp" / stack_name
    snr_dir = fig_root / "SNR"
    for directory in (stack_dir, disp_dir, snr_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return stack_dir, disp_dir, snr_dir


def _select_offsets(stack):
    """Restrict a VSG to the requested propagation side(s)
    """
    if offsets_to_keep == "backward":
        keep = stack.x_axis <= 0
        stack.XCF_out = stack.XCF_out[keep, :]
        stack.x_axis = stack.x_axis[keep]
    elif offsets_to_keep == "forward":
        keep = stack.x_axis >= 0
        stack.XCF_out = stack.XCF_out[keep, :]
        stack.x_axis = stack.x_axis[keep]
    elif offsets_to_keep == "both":
        # Match the positive/negative aperture lengths before combining the two
        # propagation directions in the f-v transform.
        if np.abs(stack.x_axis[0]) > np.abs(stack.x_axis[-1]):
            stack.x_axis = stack.x_axis[1:]
            stack.XCF_out = stack.XCF_out[1:]
        elif np.abs(stack.x_axis[-1]) > np.abs(stack.x_axis[0]):
            stack.x_axis = stack.x_axis[:-1]
            stack.XCF_out = stack.XCF_out[:-1]
    else:
        raise ValueError(
            "offsets_to_keep must be one of {'backward', 'forward', 'both'}"
        )


def _select_lags(stack):
    """Restrict a VSG to causal, acausal, or both correlation lags
    """
    if lags_to_keep == "causal":
        keep = stack.t_axis >= 0
    elif lags_to_keep == "acausal":
        keep = stack.t_axis <= 0
    elif lags_to_keep == "both":
        return
    else:
        raise ValueError("lags_to_keep must be one of {'causal', 'acausal', 'both'}")

    stack.XCF_out = stack.XCF_out[:, keep]
    stack.t_axis = stack.t_axis[keep]


def _plot_stack(stack, output_path, fontsize=12):
    """Save a time-lag/inter-channel-distance representation of a VSG
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    pclip = np.percentile(np.abs(stack.XCF_out), 99)
    ax.pcolormesh(stack.t_axis, stack.x_axis,stack.XCF_out, cmap="seismic", vmax=pclip,vmin=-pclip, shading="auto")
    ax.set_xlabel("Time lag [s]", fontsize=fontsize)
    ax.set_ylabel("Inter-channel distance [m]", fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _dispersion_color_limits(dispersion, velocity_axis):
    """Return the original percentile-based limits over the displayed velocity band
    """
    # The original code intended to use the 250--2000 m/s display interval but
    # used ``>= 2000`` for the upper condition. The corrected selector affects
    # figure contrast only, not the computed dispersion image.
    keep = (velocity_axis >= DISP_VEL_LIM[0]) & (velocity_axis <= DISP_VEL_LIM[1])
    values = np.abs(dispersion[keep]) if np.any(keep) else np.abs(dispersion)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None, None
    return (np.percentile(finite_values, 99.0) * 5.0,
        np.percentile(finite_values, 50.0))


def _plot_dispersion(freq_axis, velocity_axis, dispersion, output_path):
    """Save a diagnostic f-v image using the established DAS-VTX log axes
    """
    vmax, vmin = _dispersion_color_limits(dispersion, velocity_axis)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.pcolormesh(freq_axis, velocity_axis, dispersion,  cmap="jet",  vmax=vmax,  vmin=vmin, shading="auto")
    ax.set( xscale="log", xlim=DISP_FREQ_LIM, yscale="log", ylim=DISP_VEL_LIM)
    ax.set_xlabel("Frequency [Hz]", fontsize="x-large")
    ax.set_ylabel("Phase velocity [m/s]", fontsize="x-large")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def interpretation_process(stack_file):
    """Compute and save dispersion products for one stacked VSG archive.

    Parameters
    ----------
    stack_file : str or pathlib.Path
        ``.npz`` archive created by :func:`xcorr_process.xcorr_process`.

    Notes
    -----
    When coherence enhancement is enabled, DAS-VTX sums the semblance fields
    computed for the configured slowness values, interpolates the decimated
    coherence field back onto the VSG grid, and multiplies the VSG by the
    **squared** coherence field. The exponent is intentionally preserved from
    the reference implementation.

    When aperture masking is enabled, f-v images are recomputed for
    progressively shorter maximum offsets. Only positive real-valued energy is
    accumulated. The normalized accumulation therefore acts as an
    aperture-stability weight: energy repeatedly recovered across aperture
    choices receives a larger mask value.
    """
    stack_path = Path(stack_file)
    stack_name = stack_path.name
    stack_fig_dir, disp_root_dir, snr_fig_dir = _prepare_output_directories(stack_name)

    with np.load(stack_path, allow_pickle=True) as content:
        parameters = content["parameters"].item()
        stack_full = content["stack"].item()
        snr_values = content["SNR"]
        vsg_times = content["vsg_times"]

    _plot_stack(stack_full,stack_fig_dir / stack_name.replace(".npz", ".png"),fontsize=12)
    fig, ax = plt.subplots()
    ax.plot(snr_values, "k")
    ax.set_xlabel("Number of VSGs in the stack", fontsize=12)
    ax.set_ylabel("SNR [dB]", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(visible=True)
    fig.tight_layout()
    fig.savefig(snr_fig_dir / stack_name.replace(".npz", ".png"), dpi=300)
    plt.close(fig)

    _select_offsets(stack_full)
    _select_lags(stack_full)

    if stack_norm:
        stack_full.norm()

    if coherence_enhancing:
        interpolator = RegularGridInterpolator(
            (stack_full.x_axis, stack_full.t_axis),
            stack_full.XCF_out,bounds_error=False,fill_value=0,)
        enhancer = Coherence_Enhancement(
            stack_full,interpolator,xwin,twin,list(slw_list),decim,n_processes)
        results = enhancer.calculate_enhanced_stack()
        sem = 2 * np.asarray(sum(results)) / len(slw_list)

        if stack_full.t_axis.shape[0] % decim == 0:
            shift = stack_full.t_axis[::decim]
        else:
            shift = stack_full.t_axis[:-decim:decim]
        sem_interp = RegularGridInterpolator(
            (stack_full.x_axis, shift),
            sem,bounds_error=False, fill_value=0)
        xx, yy = np.meshgrid(stack_full.x_axis, stack_full.t_axis, indexing="ij")
        sem = sem_interp((xx, yy))

        # Squared semblance weighting is part of the reference DAS-VTX method.
        stack_full.XCF_out *= sem**2

    stack_full.compute_disp_image(freqs=freqs, vels=vels, mod=True, apodis=True)
    disp_full = stack_full.disp.fv_map

    disp_figs_dir = disp_root_dir / (
        f"norm={stack_norm}_coher-enhancing={coherence_enhancing}_"
        f"offsets={offsets_to_keep}_lags={lags_to_keep}")
    disp_figs_dir.mkdir(parents=True, exist_ok=True)

    _plot_stack(stack_full, disp_figs_dir / "Processed_stack_used.png", fontsize="x-large")
    _plot_dispersion(stack_full.disp.freqs,stack_full.disp.vels,disp_full,disp_figs_dir / "Disp_without_masking.png",)

    final_mask = None
    disp_masked = None
    if disp_masking:
        if step_factor <= 0 or max_cut_dist <= min_cut_dist:
            raise ValueError(
                "Dispersion masking requires step_factor > 0 and max_cut_dist > min_cut_dist"
            )

        final_mask = np.zeros((len(vels), len(freqs)))
        n_cuts = int((max_cut_dist - min_cut_dist) // (step_factor * dx))

        for k in range(n_cuts):
            # The +1 m boundary offset is retained from the reference
            # implementation to preserve channel-inclusion behavior.
            cut_dist = max_cut_dist - ((k + 1) * step_factor * dx) + 1

            stack_cut = deepcopy(stack_full)
            keep = np.abs(stack_cut.x_axis) <= cut_dist
            stack_cut.XCF_out = stack_cut.XCF_out[keep]
            stack_cut.x_axis = stack_cut.x_axis[keep]

            stack_cut.compute_disp_image(freqs=freqs, vels=vels, apodis=True,  mod=False,)
            masking_disp = stack_cut.disp.fv_map
            masking_disp[np.isnan(masking_disp)] = 0
            masking_disp[masking_disp < 0] = 0
            final_mask += masking_disp

            fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4))
            pclip = np.percentile(np.abs(stack_cut.XCF_out), 99)
            axes[0].pcolormesh(stack_cut.t_axis,stack_cut.x_axis,stack_cut.XCF_out,cmap="seismic",vmax=pclip,vmin=-pclip,shading="auto",)
            axes[0].set_xlabel("Time lag [s]", fontsize="x-large")
            axes[0].set_ylabel("Inter-channel distance [m]", fontsize="x-large")

            vmax, vmin = _dispersion_color_limits(masking_disp, stack_cut.disp.vels)
            axes[1].pcolormesh( stack_cut.disp.freqs, stack_cut.disp.vels, masking_disp, cmap="jet", vmax=vmax,vmin=vmin, shading="auto",)
            axes[1].set( xscale="log", xlim=DISP_FREQ_LIM, yscale="log", ylim=DISP_VEL_LIM, )
            axes[1].set_xlabel("Frequency [Hz]", fontsize="x-large")
            axes[1].set_ylabel("Phase velocity [m/s]", fontsize="x-large")
            fig.tight_layout()
            fig.savefig(disp_figs_dir / f"Masking_disp_{k}_cut-dist={cut_dist:.1f}.png",dpi=300,)
            plt.close(fig)

        mask_max = np.max(final_mask)
        if mask_max > 0:
            final_mask /= mask_max

        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        finite_mask = np.abs(final_mask[np.isfinite(final_mask)])
        vmax = np.percentile(finite_mask, 99.0) if finite_mask.size else None
        vmin = np.percentile(finite_mask, 50.0) if finite_mask.size else None
        ax.pcolormesh( freqs,vels, final_mask,
            cmap="jet", vmax=vmax, vmin=vmin,shading="auto", )
        ax.set( xscale="log", xlim=DISP_FREQ_LIM, yscale="log", ylim=DISP_VEL_LIM,)
        ax.set_xlabel("Frequency [Hz]", fontsize="x-large")
        ax.set_ylabel("Phase velocity [m/s]", fontsize="x-large")
        fig.tight_layout()
        fig.savefig(disp_figs_dir / "Final_mask.png", dpi=300)
        plt.close(fig)

        disp_masked = deepcopy(disp_full)
        disp_masked *= final_mask
        _plot_dispersion( stack_full.disp.freqs,stack_full.disp.vels, disp_masked, disp_figs_dir / "Masked_disp.png",)

    savedir = PROCESSED_DIR / "DISPs"
    savedir.mkdir(parents=True, exist_ok=True)
    disp_name = "Disp_" + stack_name
    output = {"parameters": parameters,
        "disp": disp_full,
        "vels": vels,
        "freqs": freqs,
        "vsg_times": vsg_times}
    
    if disp_masking:
        output.update(mask=final_mask, masked_disp=disp_masked)
    np.savez(savedir / disp_name, **output)
