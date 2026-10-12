"""Frequency-wavenumber and frequency-velocity dispersion utilities

The DAS-VTX dispersion image is obtained by transforming the available
propagation quadrants of a Virtual Shot Gather to the f-k domain and sampling
the resulting spectrum along k = f / v for the requested phase velocities
This module mainly centralizes validation, normalization, persistence, and
plotting around that calculation.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import savgol_filter
from skimage.filters import window

logger = logging.getLogger(__name__)


def compute_fk_spectrum(data, dx, dt, apodis=False):
    """Compute the positive-frequency/positive-wavenumber complex f-k spectrum

    """
    transformed_data = np.asarray(data)
    if apodis:
        transformed_data = transformed_data * window("hann", transformed_data.shape)

    fk_raw = np.fft.fft2(transformed_data)
    fft_k = np.fft.fftfreq(fk_raw.shape[0], dx)
    fft_f = np.fft.fftfreq(fk_raw.shape[1], dt)

    positive_k = fft_k >= 0
    positive_f = fft_f >= 0
    fk_res = fk_raw[positive_k, :][:, positive_f]
    return fk_res, fft_f[positive_f], fft_k[positive_k]


def _normalize_rows_l1(data):
    """L1-normalize traces while leaving zero-norm traces unchanged
    """
    norms = np.linalg.norm(data, axis=-1, keepdims=True, ord=1)
    return np.divide(data, norms, out=np.zeros_like(data, dtype=float), where=norms != 0)


def map_fv(data, x_axis, t_axis, dx, dt, freqs, vels, norm=False, apodis=False, mod=True):
    """Map a VSG from offset/time coordinates to frequency/phase velocity

    combines whichever causal/acausal and forward/backward propagation
    quadrants are present. Quadrants are oriented consistently before their f-k
    spectra are summed. The result is then interpolated along k = f / v

    Parameters
    ----------
    data : numpy.ndarray
        VSG amplitudes shaped ``(n_offsets, n_lags)``.
    x_axis : numpy.ndarray
        Inter-channel offsets [m]. Negative/positive values represent the two
        propagation sides of the virtual source.
    t_axis : numpy.ndarray
        Correlation-lag axis [s].
    dx, dt : float
        Spatial and temporal sampling intervals.
    freqs : numpy.ndarray
        Frequencies [Hz] sampled in the dispersion image.
    vels : numpy.ndarray
        Trial phase velocities [m/s].
    norm : bool
        Apply trace-wise L1 normalization before the f-k transform.
    apodis : bool
        Apply a 2-D Hann taper before each f-k transform.
    mod : bool
        Use the modulus of the summed complex spectrum. If ``False``, use its
        real part; the latter is used by the aperture-stability masking stage.
    """
    data = np.asarray(data)
    x_axis = np.asarray(x_axis)
    t_axis = np.asarray(t_axis)
    freqs = np.asarray(freqs)
    vels = np.asarray(vels)

    if data.shape != (x_axis.size, t_axis.size):
        raise ValueError("data shape must match x_axis and t_axis lengths")
    if np.any(vels == 0):
        raise ValueError("Phase-velocity grid cannot contain zero")

    if norm:
        data = _normalize_rows_l1(data)

    backward_prop = np.count_nonzero(x_axis < 0) > 1
    forward_prop = np.count_nonzero(x_axis > 0) > 1
    acausal = np.count_nonzero(t_axis < 0) > 1
    causal = np.count_nonzero(t_axis > 0) > 1

    spectra = {}
    if forward_prop and acausal:
        spectra["forward_acausal"] = compute_fk_spectrum(
            data[x_axis >= 0, :][:, t_axis <= 0], dx, dt, apodis
        )
    if forward_prop and causal:
        spectra["forward_causal"] = compute_fk_spectrum(
            data[x_axis >= 0, :][:, t_axis >= 0][::-1, :], dx, dt, apodis
        )
    if backward_prop and acausal:
        spectra["backward_acausal"] = compute_fk_spectrum(
            data[x_axis <= 0, :][:, t_axis <= 0][:, ::-1], dx, dt, apodis
        )
    if backward_prop and causal:
        spectra["backward_causal"] = compute_fk_spectrum(
            data[x_axis <= 0, :][:, t_axis >= 0], dx, dt, apodis
        )

    if not spectra:
        raise ValueError("VSG does not contain enough samples in any propagation quadrant")

    # Reproduce the original quadrant-combination logic.
    if backward_prop and not forward_prop:
        keys = ["backward_acausal"] if acausal and not causal else ["backward_causal"]
        if causal and acausal:
            keys = ["backward_acausal", "backward_causal"]
    elif forward_prop and not backward_prop:
        keys = ["forward_acausal"] if acausal and not causal else ["forward_causal"]
        if causal and acausal:
            keys = ["forward_acausal", "forward_causal"]
    else:
        if acausal and not causal:
            keys = ["forward_acausal", "backward_acausal"]
        elif causal and not acausal:
            keys = ["forward_causal", "backward_causal"]
        else:
            keys = [
                "forward_acausal",
                "forward_causal",
                "backward_acausal",
                "backward_causal",
            ]

    first_spectrum, fft_f, fft_k = spectra[keys[0]]
    fk_res = first_spectrum.copy()
    for key in keys[1:]:
        fk_res += spectra[key][0]

    fk_res = np.abs(fk_res) if mod else np.real(fk_res)
    interp_fun = RegularGridInterpolator(
        (fft_k, fft_f), fk_res, bounds_error=False, fill_value=0.0
    )

    nscanv = vels.size
    ones_arr = np.ones(nscanv)
    fv_map = np.zeros((freqs.size, vels.size), dtype=np.float32)
    for index, frequency in enumerate(freqs):
        wavenumbers = frequency * ones_arr / vels
        interpolation_points = np.column_stack(
            (wavenumbers, np.full(nscanv, frequency))
        )
        fv_map[index, :] = interp_fun(interpolation_points)

    # The 25-sample, fourth-order Savitzky-Golay smoothing is part of the
    # reference implementation. For unusually short frequency grids, use the
    # largest valid odd window instead of failing.
    if fv_map.shape[0] >= 5:
        smooth_window = min(25, fv_map.shape[0] if fv_map.shape[0] % 2 else fv_map.shape[0] - 1)
        if smooth_window >= 5:
            fv_map = savgol_filter(fv_map, smooth_window, min(4, smooth_window - 1), axis=0)

    return fv_map.T


def plot_fv_map(fv_map, freqs, vels, norm=True, fig_dir="Fig/", fig_name=None,  ax=None,
    pclip=90, fontsize=24, tickfont=20, **kwargs,):
    """Plot a frequency-phase-velocity image
    
    """
    image = np.asarray(fv_map, dtype=float).copy()
    if norm:
        column_max = np.nanmax(image, axis=0, keepdims=True)
        image = np.divide(image, column_max, out=np.zeros_like(image), where=column_max != 0)

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 10)))
    else:
        fig = ax.figure

    finite_values = np.abs(image[np.isfinite(image)])
    vmax = np.percentile(finite_values, pclip) if finite_values.size else None
    vmin = np.percentile(finite_values, 100 - pclip) if finite_values.size else None

    ax.imshow(
        image,
        aspect="auto",
        extent=[freqs[0], freqs[-1], vels[0], vels[-1]],
        cmap=kwargs.get("cmap", "jet"),
        vmax=vmax,
        vmin=vmin,
        origin="lower",
    )
    ax.grid(kwargs.get("grid", True))
    ax.set_xscale(kwargs.get("set_xscale", "linear"))
    ax.set_xlabel("Frequency [Hz]", fontsize=fontsize)
    ax.set_ylabel("Phase velocity [m/s]", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tickfont)
    fig.tight_layout()

    if fig_name:
        fig_path = Path(fig_dir) / fig_name
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving dispersion figure: %s", fig_path)
        fig.savefig(fig_path)
        if created_figure:
            plt.close(fig)
    elif created_figure:
        plt.show()

    return ax


class Dispersion:
    """Container for an f-v dispersion image and its coordinate axes
    
    """

    def __init__(
        self,
        data,
        x_axis,
        t_axis,
        dx,
        dt,
        freqs,
        vels,
        norm=False,
        apodis=False,
        mod=True,
        compute_fv=True,
    ):
        self.data = data
        self.x_axis = x_axis
        self.t_axis = t_axis
        self.dx = dx
        self.dt = dt
        self.freqs = freqs
        self.vels = vels
        self.norm = norm
        self.apodis = apodis
        self.mod = mod
        if compute_fv:
            self._map_fv()

    def save_to_npz(self, fname, fdir="./"):
        """Save the dispersion axes and image to an ``.npz`` archive
        """
        output = Path(fdir) / fname
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, freqs=self.freqs, vels=self.vels, fv_map=self.fv_map)

    @classmethod
    def get_dispersion_obj(cls, fname, fdir="./"):
        """Reconstruct a :class:`Dispersion` object from an ``.npz`` archive
        """
        with np.load(Path(fdir) / fname) as file:
            obj = cls(
                data=None,
                x_axis=None,
                t_axis=None,
                dx=None,
                dt=None,
                freqs=file["freqs"],
                vels=file["vels"],
                compute_fv=False,
            )
            obj.fv_map = file["fv_map"]
        return obj

    def _map_fv(self):
        """Compute and store the f-v image from the object's VSG data
        """
        self.fv_map = map_fv(
            self.data,
            self.x_axis,
            self.t_axis,
            self.dx,
            self.dt,
            freqs=self.freqs,
            vels=self.vels,
            norm=self.norm,
            apodis=self.apodis,
            mod=self.mod,
        )

    def plot_image(self, fig_dir=None, fig_name=None, norm=False, **kwargs):
        """Plot the stored dispersion image
        """
        use_norm = norm or self.norm
        return plot_fv_map(
            self.fv_map,
            self.freqs,
            self.vels,
            norm=use_norm,
            fig_dir=fig_dir or "Fig/",
            fig_name=fig_name,
            **kwargs,
        )

    def __add__(self, other):
        """Add two dispersion images on the same grid
        """
        if not np.array_equal(self.freqs, other.freqs) or not np.array_equal(self.vels, other.vels):
            raise ValueError("Dispersion objects must use identical frequency/velocity grids")
        sum_ = copy.deepcopy(self)
        sum_.fv_map = self.fv_map + other.fv_map
        return sum_

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __truediv__(self, other: float):
        if other == 0:
            raise ZeroDivisionError("Cannot divide a dispersion image by zero")
        div_ = copy.deepcopy(self)
        div_.fv_map /= other
        return div_
