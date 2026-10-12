"""Virtual Shot Gather construction and plotting for DAS-VTX

This module implements the trajectory-guided cross-correlation method used by
DAS-VTX. Successive positions along a tracked moving-source trajectory act as
virtual-source locations. Correlations are accumulated toward each propagation
side, averaged by the number of contributing virtual-source positions, and
assembled into a VSG
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import correlate, welch

from Disp_class import Dispersion
from SW_class import SurfaceWaveWindow
from utils import repeat1d

logger = logging.getLogger(__name__)


def _normalized_window_data(window):
    """Return globally L2-normalized window data without modifying the input
    """
    norm = np.linalg.norm(window.data)
    if norm == 0:
        raise ValueError("Surface-wave window contains zero energy")
    return window.data / norm


def _validate_xcorr_window(wlen, dt, overlap_ratio):
    """Convert correlation-window length to samples and validate overlap
    """
    if dt <= 0 or wlen <= 0:
        raise ValueError("wlen and dt must be positive")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must satisfy 0 <= overlap_ratio < 1")
    wlen_samples = int(wlen / dt)
    if wlen_samples < 1:
        raise ValueError("Correlation window is shorter than one time sample")
    offset = int(wlen_samples * (1 - overlap_ratio))
    if offset < 1:
        raise ValueError("Overlap leaves a zero-sample correlation-window step")
    return wlen_samples, offset


def preprocessing_window(window, pivot, delta_t, time_window_to_xcorr):
    """Locate the correlation interval relative to the tracked source time

    The source trajectory is linearly interpolated in distance and extrapolated
    when required. delta_t shifts the correlation interval relative to the
    estimated source passage at each virtual-source position
    """
    trajectory_time = interpolate.interp1d(window.veh_state_x,
        window.veh_state_t,
        fill_value="extrapolate",)
    dt = window.t_axis[1] - window.t_axis[0]
    pivot_t = float(trajectory_time(pivot)) + delta_t
    pivot_t_idx = np.argmax(window.t_axis >= pivot_t)
    nsamp = int(time_window_to_xcorr // dt)
    if nsamp <= 0:
        raise ValueError("time_window_to_xcorr is shorter than one sample")
    return pivot_t_idx, nsamp, _normalized_window_data(window), dt, trajectory_time


def cross_correlate_virtual_shot(data, ivs, wlen, dt, overlap_ratio=0.5, reverse=False):
    """Cross-correlate one virtual-source trace with the remaining traces
    """
    nch, nt = data.shape
    if nch < 2:
        raise ValueError("At least two channels are required for a VSG")
    if not 0 <= ivs < nch:
        raise IndexError("Virtual-source index is outside the input data")

    wlen_samples, wlen_offset = _validate_xcorr_window(
        wlen, dt, overlap_ratio)
    nwin = (nt - wlen_samples) // wlen_offset + 1
    if nwin <= 0:
        raise ValueError("Correlation interval is shorter than the requested window")

    xcf_out = np.zeros((nch - 1, wlen_samples))
    for iwin in range(nwin):
        start = iwin * wlen_offset
        stop = start + wlen_samples
        data_vs = repeat1d(data[ivs, start:stop])
        current_xcf = []

        if reverse:
            for ivr in range(1, nch):
                current_xcf.append(correlate(data[ivr, start:stop], data_vs, mode="valid", method="fft"))
        else:
            for ivr in range(nch - 1):
                current_xcf.append(correlate(data_vs, data[ivr, start:stop], mode="valid", method="fft"))
        xcf_out += np.asarray(current_xcf)

    return np.roll(xcf_out, wlen_samples // 2, axis=-1) / nwin


def construct_shot_gather(window: SurfaceWaveWindow, start_x,end_x,wlen=2, norm=True, norm_amp=True, time_window_to_xcorr=4,
    delta_t=1,overlap=0.5):
    """Construct a VSG on the first side of the tracked trajectory
    """
    idx0 = np.abs(window.x_axis - window.veh_state_x[-1]).argmin()
    start_x_idx = np.abs(window.x_axis - start_x).argmin()
    end_x_idx = np.abs(window.x_axis - end_x).argmin()
    idx0 = min(idx0, end_x_idx)
    nch = idx0 - start_x_idx
    if nch <= 0:
        raise ValueError("No valid channels available on the requested VSG side")

    dt = window.t_axis[1] - window.t_axis[0]
    xcf_out = np.zeros((nch, int(wlen / dt)))
    contribution_count = np.arange(1, nch + 1)

    for pivot_idx in range(start_x_idx + 1, idx0 + 1):
        virtual_pivot = window.x_axis[pivot_idx]
        pivot_t_idx, nsamp, data, dt, _ = preprocessing_window(window, virtual_pivot, delta_t,
            time_window_to_xcorr)
        data_slice = data[start_x_idx : pivot_idx + 1,
            pivot_t_idx : pivot_t_idx + nsamp]
        current = cross_correlate_virtual_shot(data_slice,
            ivs=pivot_idx - start_x_idx,wlen=wlen,
            dt=dt,overlap_ratio=overlap)[::-1, :]
        xcf_out[: pivot_idx - start_x_idx, :] += current

    for i in range(xcf_out.shape[0]):
        xcf_out[i, :] /= contribution_count[::-1][i]

    return postprocess_cross_correlations(window,  idx0,  start_x_idx,
        end_x_idx,   xcf_out, dt,norm, norm_amp, reverse=False,)


def construct_shot_gather_other_side(window: SurfaceWaveWindow, start_x,end_x,  wlen=2, norm=True, norm_amp=True,
    time_window_to_xcorr=4, delta_t=1, overlap=0.5):
    
    """Construct a VSG on the opposite side of the tracked trajectory
    """
    idx0 = np.abs(window.x_axis - window.veh_state_x[0]).argmin()
    start_x_idx = np.abs(window.x_axis - start_x).argmin()
    end_x_idx = np.abs(window.x_axis - end_x).argmin()
    idx0 = max(idx0, start_x_idx)
    nch = end_x_idx - idx0
    if nch <= 0:
        raise ValueError("No valid channels available on the opposite VSG side")

    dt = window.t_axis[1] - window.t_axis[0]
    xcf_out = np.zeros((nch, int(wlen / dt)))
    contribution_count = np.arange(1, nch + 1)

    for pivot_idx in range(idx0, end_x_idx):
        virtual_pivot = window.x_axis[pivot_idx]
        pivot_t_idx, nsamp, data, dt, _ = preprocessing_window(window, virtual_pivot,
            -delta_t, time_window_to_xcorr)
        data_slice = data[pivot_idx : end_x_idx + 1,
            pivot_t_idx - nsamp : pivot_t_idx]
        current = cross_correlate_virtual_shot( data_slice, ivs=0,
            wlen=wlen, dt=dt,
            overlap_ratio=overlap, reverse=True)
        xcf_out[: end_x_idx - pivot_idx, :] += current

    for i in range(xcf_out.shape[0]):
        xcf_out[i, :] /= contribution_count[::-1][i]

    return postprocess_cross_correlations( window, idx0, start_x_idx, end_x_idx, xcf_out,dt, norm, norm_amp, reverse=True)


def postprocess_cross_correlations(window, pivot_idx, start_x_idx,end_x_idx, XCF_out,  dt, norm, norm_amp, reverse,):
    """Orient and optionally normalize an accumulated VSG
    """
    xcf_out = np.asarray(XCF_out, dtype=float)
    nt = xcf_out.shape[-1]
    t_axis = (np.arange(nt) - nt // 2) * dt

    if norm:
        trace_norm = np.linalg.norm(xcf_out, axis=-1, keepdims=True)
        xcf_out = np.divide(
            xcf_out,
            trace_norm,
            out=np.zeros_like(xcf_out),
            where=trace_norm != 0)
    if norm_amp:
        amplitude = np.amax(xcf_out[0])
        if amplitude != 0:
            xcf_out /= amplitude

    if not reverse:
        xcf_out = xcf_out[::-1, :]
        xcf_out = xcf_out[:, ::-1]
        x_axis = (window.x_axis[start_x_idx:pivot_idx]
            - window.x_axis[pivot_idx])
    else:
        x_axis = (window.x_axis[pivot_idx:end_x_idx]
            - window.x_axis[pivot_idx - 1])

    return xcf_out, x_axis, t_axis


# -----------------------------------------------------------------------------
# VSG visualization helpers
# -----------------------------------------------------------------------------
def plot_psd_vs_offset(XCF_out,
    x_axis,
    t_axis,
    ax=None,
    fhi=20,
    figsize=(8, 8),
    pclip=98,
    log_scale=False,
    x_max=200,
    x_min=0,
    fname=None,
    fdir="./",
    vmax=None,
    vmin=None):
    
    """Plot Welch power spectral density as a function of VSG offset
    """
    
    x_axis = np.asarray(x_axis).copy()
    if x_axis[0] > x_axis[-1]:
        x_axis *= -1

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    dt = t_axis[1] - t_axis[0]
    fs = 1 / dt
    freq, pxx = welch(XCF_out, fs, nperseg=256, nfft=1024)
    fhi_idx = np.argmax(freq >= fhi)
    if fhi_idx == 0:
        fhi_idx = len(freq)
    spec = pxx[:, :fhi_idx]
    if log_scale:
        with np.errstate(divide="ignore"):
            spec = 10 * np.log10(spec)

    finite = spec[np.isfinite(spec)]
    if vmax is None and finite.size:
        vmax = np.percentile(finite, pclip)
    if vmin is None and finite.size:
        vmin = np.percentile(finite, 100 - pclip)

    x_max_idx = np.abs(x_max - x_axis).argmin()
    x_min_idx = np.abs(x_min - x_axis).argmin()
    min_idx, max_idx = sorted((x_max_idx, x_min_idx))
    if max_idx == min_idx:
        max_idx = min(max_idx + 1, len(x_axis) - 1)
    spec = spec[min_idx : max_idx + 1]

    ax.imshow(spec.T,extent=[x_axis[min_idx], x_axis[max_idx], freq[fhi_idx - 1], freq[0]],
        cmap="jet",aspect="auto",
        vmax=vmax, vmin=vmin, interpolation="antialiased")
    ax.set_xlabel("Inter-channel distance [m]")
    ax.set_ylabel("Frequency [Hz]")

    if fname:
        output = Path(fdir) / fname
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        logger.info("Saved spectrum figure: %s", output)
        if created_figure:
            plt.close(fig)
    return ax


def plot_spectrum_vs_offset(XCF_out, x_axis, t_axis,ax=None,fhi=20,figsize=(8, 8),fname=None,fdir="./"):
    """Plot FFT amplitude as a function of VSG offset
    """
    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    nt = XCF_out.shape[-1]
    dt = t_axis[1] - t_axis[0]
    freq = np.fft.fftfreq(nt, d=dt)
    fhi_idx = np.argmax(freq >= fhi)
    spec = np.fft.fft(XCF_out, axis=-1)[:, :fhi_idx]
    ax.imshow(
        np.abs(spec).T,
        extent=[x_axis[0], x_axis[-1], freq[fhi_idx], freq[0]],
        cmap="jet",
        aspect="auto",
    )
    ax.set_xlabel("Inter-channel distance [m]")
    ax.set_ylabel("Frequency [Hz]")

    if fname:
        output = Path(fdir) / fname
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        logger.info("Saved spectrum figure: %s", output)
        if created_figure:
            plt.close(fig)
    return ax


def plot_xcorr(xcorr,
    t_axis,
    x_axis=None,
    ax=None,
    figsize=(8, 10),
    cmap="seismic",
    vmax_use_max=False,
    fig_dir=None,
    fig_name=None,
    fontsize=24,
    tickfont=20,
    x_lim=None,
    **plot_kwargs):
    
    """Plot a VSG in inter-channel-distance/time-lag coordinates
    """
    
    if x_axis is None:
        x_axis = np.arange(xcorr.shape[0])
    x_lim = [-120, 120] if x_lim is None else x_lim

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    xcorr_to_plot = np.asarray(xcorr, dtype=float).copy()
    x_origin_index = np.abs(x_axis).argmin()
    origin_amplitude = np.amax(xcorr_to_plot[x_origin_index])
    if origin_amplitude != 0:
        xcorr_to_plot /= origin_amplitude

    vmax = (plot_kwargs.get("vmax", np.percentile(np.abs(xcorr_to_plot), 100))
        if vmax_use_max
        else 1)
    start_x = -np.abs(x_axis[0])
    end_x = np.abs(x_axis[-1])

    ax.imshow(xcorr_to_plot.T,aspect="auto",
        vmax=vmax,vmin=-vmax, cmap=cmap, extent=[start_x, end_x, t_axis[-1], t_axis[0]], interpolation="bicubic")
    ax.set_xlabel("Inter-channel distance [m]", fontsize=fontsize)
    ax.set_ylabel("Time lag [s]", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tickfont)
    ax.set_xlim(x_lim)

    if fig_name and fig_dir:
        output = Path(fig_dir) / fig_name
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output, format=output.suffix.lstrip(".") or "svg", bbox_inches="tight")
        logger.info("Saved VSG figure: %s", output)
        if created_figure:
            plt.close(fig)
    return ax


class VirtualShotGather:
    
    """Container for a virtual shot gather and optional dispersion image
    """

    def __init__(self,
        window: SurfaceWaveWindow | None,  compute_xcorr=True, include_other_side=False,
        *args, **kwargs):
        """Create a VSG from a surface-wave window

        Parameters
        ----------
        window
            Surface-wave window containing the tracked source trajectory and
            DAS data. May be ``None`` when constructing an empty object for
            loading a previously saved VSG.
        compute_xcorr
            If ``True``, construct the trajectory-guided VSG immediately.
        include_other_side
            If ``True``, also construct and append the VSG from the opposite
            side of the tracked trajectory.
        *args, **kwargs
            Forwarded to :func:`construct_shot_gather`.
        """
        self.window = window
        self.disp = None
        if not compute_xcorr:
            return
        if window is None:
            raise ValueError("A SurfaceWaveWindow is required to compute a VSG")

        self.XCF_out, self.x_axis, self.t_axis = construct_shot_gather(
            window, *args, **kwargs
        )
        if include_other_side:
            other_xcf, other_x, _ = construct_shot_gather_other_side(
                window, *args, **kwargs
            )
            self.XCF_out = np.vstack((self.XCF_out, other_xcf))
            self.x_axis = np.hstack((self.x_axis, other_x))

    def __add__(self, other):
        """Add VSG amplitudes over their common time-lag length."""
        if self.XCF_out.shape[0] != other.XCF_out.shape[0]:
            raise ValueError("VSGs must contain the same number of offset channels")
        summed = copy.deepcopy(self)
        length = min(self.XCF_out.shape[-1], other.XCF_out.shape[-1])
        summed.XCF_out[:, :length] += other.XCF_out[:, :length]
        return summed

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __truediv__(self, other):
        if other == 0:
            raise ZeroDivisionError("Cannot divide a VSG by zero")
        new_obj = copy.deepcopy(self)
        new_obj.XCF_out /= other
        return new_obj

    @classmethod
    def get_VirtualShotGather_obj(cls, fdir, fname):
        """Load a directly saved VSG archive containing XCF_out/x_axis/t_axis."""
        new_obj = cls(window=None, compute_xcorr=False)
        with np.load(Path(fdir) / fname, allow_pickle=True) as file:
            new_obj.XCF_out = file["XCF_out"]
            new_obj.x_axis = file["x_axis"]
            new_obj.t_axis = file["t_axis"]
        return new_obj

    def plot_spec_vs_offset(self,
        ax=None, psd=True,
        pclip=98,fdir="Fig/virtual_gathers",fname=None,x_max=100,
        x_min=-100, log_scale=False,vmin=None, vmax=None):
        
        """Plot PSD or FFT amplitude versus inter-channel distance
        """
        
        Path(fdir).mkdir(parents=True, exist_ok=True)
        if psd:
            return plot_psd_vs_offset(
                self.XCF_out,
                self.x_axis,
                self.t_axis,
                ax=ax,
                pclip=pclip,
                x_max=x_max,
                x_min=x_min,
                fdir=fdir,
                fname=fname,
                log_scale=log_scale,
                vmax=vmax,
                vmin=vmin,
            )
        return plot_spectrum_vs_offset(self.XCF_out, self.x_axis,self.t_axis,
            ax=ax,fdir=fdir, fname=fname)

    def save_to_npz(self, fname, fdir, **kwargs):
        """Save the VSG arrays and optional metadata to an .np z archive
        """
        output = Path(fdir) / fname
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            XCF_out=self.XCF_out,
            x_axis=self.x_axis,
            t_axis=self.t_axis,
            **kwargs,
        )

    def plot_image(self, fig_name=None, fig_dir=None,x_lim=None, norm=False, plot_disp=False, plot_kwargs=None,**kwargs):
        
        """Plot either the VSG itself or its computed dispersion image
    """
        x_lim = [-200, 200] if x_lim is None else x_lim
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs

        if plot_disp:
            if self.disp is None:
                raise RuntimeError("Run compute_disp_image() before plotting dispersion")
            return self.disp.plot_image(fig_dir, fig_name, norm=norm,  **kwargs)

        ax = kwargs.get("ax")
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 10))
        return plot_xcorr(self.XCF_out, self.t_axis,   self.x_axis, ax=ax,
            fig_dir=fig_dir,  fig_name=fig_name,  x_lim=x_lim,  **plot_kwargs)

    def compute_disp_image( self, freqs=np.arange(0.8, 25, 0.1),
        vels=np.arange(200, 1200),
        norm=False,
        apodis=False,
        mod=True,
        start_x=None,
        end_x=None):
        
        """Compute an f-v dispersion image over a selected VSG offset interval
        """
        
        start_x = self.x_axis[0] if start_x is None else start_x
        end_x = self.x_axis[-1] if end_x is None else end_x
        start_idx = np.abs(self.x_axis - start_x).argmin()
        end_idx = np.abs(self.x_axis - end_x).argmin()
        start_idx, end_idx = sorted((start_idx, end_idx))
        if end_idx == start_idx:
            raise ValueError("Dispersion computation requires at least two offset samples")

        self.disp = Dispersion( self.XCF_out[start_idx : end_idx + 1],
            self.x_axis[start_idx : end_idx + 1],
            self.t_axis,
            np.abs(self.x_axis[0] - self.x_axis[1]),
            np.abs(self.t_axis[1] - self.t_axis[0]),
            freqs=freqs,
            vels=vels,
            norm=norm,
            apodis=apodis,
            mod=mod)

    def plot_disp(self, fig_name=None, fig_dir="Fig/dispersion/", norm=True, **kwargs):
        """Plot the previously computed dispersion image
        """
        if self.disp is None:
            raise RuntimeError("Run compute_disp_image() before plotting dispersion")
        return self.disp.plot_image(fig_dir, fig_name, norm=norm, **kwargs)

    def save_disp_to_npz(self, *args, **kwargs):
        
        """Save the previously computed dispersion image
        """
        if self.disp is None:
            raise RuntimeError("Run compute_disp_image() before saving dispersion")
        self.disp.save_to_npz(*args, **kwargs)

    def norm(self):
        
        """L2-normalize each VSG trace in place, preserving zero traces
        """
        
        trace_norm = np.linalg.norm(self.XCF_out, axis=-1, keepdims=True)
        self.XCF_out = np.divide(
            self.XCF_out,
            trace_norm,
            out=np.zeros_like(self.XCF_out, dtype=float),
            where=trace_norm != 0,
        )
