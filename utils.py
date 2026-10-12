"""Shared numerical, formatting, plotting, and I/O utilities 

"""

from __future__ import annotations

from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, Trace
from obspy.core import UTCDateTime
from scipy.fft import fft2, fftfreq, fftshift, ifft2, ifftshift
from scipy.signal.windows import tukey
from scipy.stats import norm

from config import DAS_FILE_FORMAT, DATE_FORMAT, DATE_TIME_FORMAT
from func_data_imports import data_matrix_import


# -----------------------------------------------------------------------------
# Tracking preprocessing
# -----------------------------------------------------------------------------
def fk_velocity_filter(data_for_tracking, dist_along_fiber_tracking,
    t_axis_tracking, param_dict):
    """Retain f-k energy inside the configured apparent-velocity wedges.

    """
    dt = t_axis_tracking[1] - t_axis_tracking[0]
    dx = dist_along_fiber_tracking[1] - dist_along_fiber_tracking[0]
    fk = fftshift(fft2(data_for_tracking))
    f_axis = fftshift(fftfreq(fk.shape[1], dt))
    k_axis = fftshift(fftfreq(fk.shape[0], dx))
    fk_filt = fk.copy()

    slope_hi = param_dict.get("slope_hi", 3.6 / 20)
    slope_lo = param_dict.get("slope_lo", 3.6 / 60)
    if slope_hi == 0 or slope_lo == 0:
        raise ValueError("FK slopes must be non-zero")

    for p, frequency in enumerate(f_axis):
        lim_hi = frequency * slope_hi
        lim_lo = frequency * slope_lo
        kk1 = np.where((k_axis <= lim_hi) & (k_axis >= lim_lo))[0]
        kk2 = np.where((k_axis >= lim_hi) & (k_axis <= lim_lo))[0]
        kk3 = np.where((k_axis <= -lim_hi) & (k_axis >= -lim_lo))[0]
        kk4 = np.where((k_axis >= -lim_hi) & (k_axis <= -lim_lo))[0]
        keep = np.concatenate((kk1, kk2, kk3, kk4))
        reject = np.setdiff1d(np.arange(len(k_axis)), keep)
        fk_filt[reject, p] = 0

    center = len(k_axis) // 2
    fk_filt[center - 2 : center + 2, :] = 0

    f_axis = ifftshift(f_axis)
    k_axis = ifftshift(k_axis)
    fk_filt = ifftshift(fk_filt)
    return ifft2(fk_filt).real, fk_filt, f_axis, k_axis


# -----------------------------------------------------------------------------
# Tracking criteria
# -----------------------------------------------------------------------------
def likelihood_1d(peak_loc, das_time_ds, sigma):
    """Sum Gaussian likelihood kernels centered on detected peak times
    
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    likelihood = np.zeros(len(das_time_ds))
    for peak_index in peak_loc:
        likelihood += norm.pdf(das_time_ds,
            loc=das_time_ds[peak_index],
            scale=sigma)
    return likelihood


def interp_nan_value(veh_states):
    """Linearly interpolate missing trajectory samples between valid samples
    
    """
    veh_states = np.asarray(veh_states)
    for k, state in enumerate(veh_states):
        valid_indices = np.where(~np.isnan(state))[0]
        if valid_indices.size == 0:
            continue
        nan_indices = np.where(np.isnan(state))[0]
        if nan_indices.size:
            state[nan_indices] = np.interp(nan_indices,
                valid_indices,
                state[valid_indices])
        veh_states[k] = state
    return veh_states


def max_total_nan_crit(veh_state, portion):
    """Reject a trajectory when its missing-data fraction exceeds portion
    """
    valid_count = np.count_nonzero(~np.isnan(veh_state))
    return valid_count < (1 - portion) * len(veh_state)


def max_adjacent_nan_crit(veh_state, value):
    """Apply the reference adjacent-NaN-pair criterion.

    Despite the historical parameter name, the implementation counts adjacent
    pairs of NaN indices over the complete trajectory rather than measuring the
    single longest NaN run. This behavior is preserved for reproducibility
    """
    nan_indices = np.where(np.isnan(veh_state))[0]
    adjacency_count = np.sum(np.diff(nan_indices) == 1)
    return adjacency_count >= value


def average_speed_crit(veh_state, dx, dt, speed_values):
    
    """Reject trajectories whose endpoint-based average speed is out of range
    """
    valid = veh_state[~np.isnan(veh_state)]
    if valid.size < 2:
        return True
    elapsed_samples = np.abs(valid[0] - valid[-1])
    if elapsed_samples == 0:
        return True
    average_speed = dx * len(veh_state) / (elapsed_samples * dt)
    return average_speed < speed_values[0] or average_speed > speed_values[1]


def curve_break_crit(veh_state, win_size, factor, portion, q):
    """Reject trajectories with too many large local trajectory increments
    """
    no_nan_diff = np.diff(veh_state)
    no_nan_diff = no_nan_diff[~np.isnan(no_nan_diff)]
    if no_nan_diff.size < win_size or win_size <= 0:
        return True
    sliding = np.convolve(
        np.abs(no_nan_diff), np.ones(win_size), mode="valid"
    )
    threshold = factor * np.percentile(sliding, q)
    return np.sum(sliding >= threshold) > portion * len(sliding)


def speed_fluctuations_crit(veh_state, dx, dt, factor, portion):
    """Reject trajectories with excessive instantaneous-speed excursions.

    """
    time_diff = np.diff(veh_state) * dt
    with np.errstate(divide="ignore", invalid="ignore"):
        instant_speed = dx / time_diff
    if np.any(time_diff == 0):
        return True
    instant_speed = instant_speed[~np.isnan(instant_speed)]
    if instant_speed.size == 0:
        return True
    average_speed = np.mean(instant_speed)
    return (
        np.sum(np.abs(instant_speed) >= factor * average_speed)
        > portion * len(instant_speed)
    )


# -----------------------------------------------------------------------------
# Surface-wave plotting
# -----------------------------------------------------------------------------

def plot_data(data, x_axis,  t_axis,
    pclip=98, ax=None, figsize=(10, 10),
    y_lim=None, x_lim=None,fig_name=None,
    fig_dir="Fig/",fontsize=16, tickfont=12):
    
    """Plot a DAS space-time matrix using symmetric percentile clipping
    
    """
    vmax = np.percentile(np.abs(data), pclip)
    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.imshow(data.T,
        aspect="auto",
        extent=[x_axis[0], x_axis[-1], t_axis[-1], t_axis[0]],
        cmap="gray",
        vmax=vmax,
        vmin=-vmax)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    ax.set_xlabel("Distance along the fiber [m]", fontsize=fontsize)
    ax.set_ylabel("Time [s]", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tickfont)
    fig.tight_layout()

    if fig_name:
        fig_path = Path(fig_dir) / fig_name
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path)
        if created_figure:
            plt.close(fig)
    return ax


# -----------------------------------------------------------------------------
# Cross-correlation helpers
# -----------------------------------------------------------------------------

def repeat1d(trace):
    """Append ``trace[:-1]`` to a trace for valid-mode circular lag sampling
    """
    return np.hstack((trace, trace[:-1]))


def whiten_signals(X, fmin, fmax, fs=1.0, alpha=0.2):
    """Spectrally whiten traces inside the configured frequency interval

    """
    signals = np.asarray(X)
    was_1d = signals.ndim == 1
    if was_1d:
        signals = signals[np.newaxis, :]
    if signals.ndim != 2:
        raise ValueError("X must be one- or two-dimensional")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie between 0 and 1")

    n_signals, n_samples = signals.shape
    whitened = np.zeros(signals.shape, dtype=float)
    freqs = np.fft.fftfreq(n_samples, 1 / fs)
    ifmin = np.argmin(np.abs(freqs - fmin))
    ifmax = np.argmin(np.abs(freqs - fmax))
    if ifmax <= ifmin:
        raise ValueError("fmax must map to a frequency bin above fmin")

    taper = tukey(ifmax + 1 - ifmin, alpha=alpha)
    for i in range(n_signals):
        spectrum = np.fft.fft(signals[i])
        amplitude = np.abs(spectrum)
        spectrum = np.divide(
            spectrum,
            amplitude,
            out=np.zeros_like(spectrum),
            where=amplitude != 0)
        spectrum[:ifmin] = 0
        spectrum[ifmax + 1 :] = 0
        spectrum[ifmin : ifmax + 1] *= taper
        whitened[i] = np.real(np.fft.ifft(spectrum))

    return whitened


def calculate_SNR(VSG):
    """Estimate VSG signal-to-noise ratio in decibels.

    """
    yy, xx = np.meshgrid(VSG.t_axis, VSG.x_axis)
    yy_safe = np.where(yy == 0, 1e-10, yy)
    ratio = np.abs(xx / yy_safe)

    noise_mask = ((ratio > 1200)
        & (xx > 0.1)
        & (yy > 0.1)
        & (yy <= 1.0))
    signal_mask = ((ratio >= 250)
        & (ratio <= 1200)
        & (xx > 0.1)
        & (yy > 0.1)
        & (yy <= 1.0))

    if not np.any(signal_mask) or not np.any(noise_mask):
        return np.nan
    signal = np.sqrt(np.mean(VSG.XCF_out[signal_mask] ** 2))
    noise = np.sqrt(np.mean(VSG.XCF_out[noise_mask] ** 2))
    if noise == 0:
        return np.inf
    if signal == 0:
        return -np.inf
    
    return 20 * np.log10(signal / noise)


# -----------------------------------------------------------------------------
# Interpretation / coherence enhancement
# -----------------------------------------------------------------------------
class Coherence_Enhancement:
    """Compute slowness-dependent local semblance weights for a stacked VSG."""

    def __init__( self, stack, interpolator,  xwin, twin, slw_list, decimation_factor,nprocesses):
        
        if decimation_factor <= 0:
            raise ValueError("decimation_factor must be positive")
        self.stack = stack
        self.interpolator = interpolator
        self.xwin = xwin
        self.twin = twin
        self.slw_list = slw_list
        self.decimation_factor = decimation_factor
        self.nprocesses = nprocesses

    def semb(self, slw):
        """Compute the local semblance field for one trial slowness [s/m]
        
        """
        n_time = self.stack.XCF_out.shape[1] // self.decimation_factor
        sem = np.empty((self.stack.XCF_out.shape[0], n_time))
        causal_t = self.stack.t_axis[self.stack.t_axis >= 0]
        acausal_t = self.stack.t_axis[self.stack.t_axis < 0]

        for i in range(self.stack.XCF_out.shape[0]):
            g_sum = 0
            gh_sum = 0
            for x in range(max(0, i - self.xwin),
                min(i + self.xwin + 1, self.stack.XCF_out.shape[0])):
                
                offset_delta = (
                    np.abs(self.stack.x_axis[x])
                    - np.abs(self.stack.x_axis[i]))
                t_interp_caus = causal_t + slw * offset_delta
                t_interp_acaus = acausal_t - slw * offset_delta
                t_interpolation_axis = np.hstack(
                    (t_interp_acaus, t_interp_caus))
                og, tg = np.meshgrid(
                    self.stack.x_axis[x],
                    t_interpolation_axis,
                    indexing="ij" )
                interpolated_data = self.interpolator((og, tg))[0, :]
                g_sum += interpolated_data
                gh_sum += interpolated_data**2

            numerator = g_sum**2
            denominator = gh_sum
            for semj in range(n_time):
                j = semj * self.decimation_factor
                num_win = numerator[max(0, j - self.twin) : min(
                        j + self.twin + 1, self.stack.XCF_out.shape[1])]
                den_win = denominator[max(0, j - self.twin) : min(
                        j + self.twin + 1, self.stack.XCF_out.shape[1] ) ]
                num = np.real(np.sum(num_win[~np.isnan(num_win)]))
                den = np.real(np.sum(den_win[~np.isnan(den_win)]))
                sem[i, semj] = num / den if den != 0 else 0.0

        return np.abs(sem) / (2 * self.xwin + 1)

    def calculate_enhanced_stack(self):
        """Evaluate semblance fields for every configured trial slowness
        
        """
        with Pool(self.nprocesses) as pool:
            return pool.map(self.semb, self.slw_list)


# -----------------------------------------------------------------------------
# Date / naming helpers
# -----------------------------------------------------------------------------
def generate_date_range(start_date: str, end_date: str) -> list:
    """Generate inclusive dates formatted with ``DATE_FORMAT``."""
    start = datetime.strptime(start_date, DATE_FORMAT)
    end = datetime.strptime(end_date, DATE_FORMAT)
    if end < start:
        raise ValueError("end_date must not precede start_date")

    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime(DATE_FORMAT))
        current += timedelta(days=1)
    return date_list


def get_date_from_file_path(file_path, format):
    """Parse a datetime from a file/directory name using format"""
    return datetime.strptime(Path(file_path).name, format)


def get_file_section(file_path):
    """Parse ``from_offset<start>to<end>_...`` section bounds from a filename"""
    name = Path(file_path).name
    try:
        chunk = name.split("offset", 1)[1].split("_", 1)[0]
        start_ch, end_ch = chunk.split("to", 1)
        return int(start_ch), int(end_ch)
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot parse section bounds from filename: {name}") from exc


# -----------------------------------------------------------------------------
# Data conversion helpers
# -----------------------------------------------------------------------------
def create_npz_data(filepath,
    destination_path,
    decimation_factor=10,chunk_size=600, return_full=False):
    
    """Decimate one raw DAS file and save fixed-duration tracking chunks
    """
    if decimation_factor <= 0:
        raise ValueError("decimation_factor must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    destination_path = Path(destination_path)
    destination_path.mkdir(parents=True, exist_ok=True)

    data, distance, t_axis, data_attributes = data_matrix_import(filepath)
    dt = t_axis[1] - t_axis[0]
    stream, _ = from_npArray2streamV2(data, 1, data_attributes[-1], 1 / dt, distance)

    stream.decimate(decimation_factor)
    t_axis_decimated = t_axis[::decimation_factor]
    data_decimated, _ = from_stream2npArray(stream)

    chunk_npts = int(chunk_size / (dt * decimation_factor))
    if chunk_npts <= 0:
        raise ValueError("chunk_size is too short for the decimated sampling interval")

    for k in range(t_axis_decimated.shape[0] // chunk_npts):
        start_idx = k * chunk_npts
        end_idx = (k + 1) * chunk_npts
        data_portion = data_decimated[:, start_idx:end_idx]

        start = get_date_from_file_path(filepath, DAS_FILE_FORMAT)
        start += timedelta(minutes=k * chunk_size // 60)
        destination = destination_path / start.strftime(DATE_TIME_FORMAT)
        np.savez(destination, data=data_portion, x_axis=distance,
            t_axis=t_axis_decimated[start_idx:end_idx],)

    if return_full:
        return data, distance, t_axis
    return None


def from_stream2npArray(stQ):
    """Convert an ObsPy stream into a channel-by-time matrix and distance axis
    """
    if len(stQ) == 0:
        return [np.empty((0, 0)), np.array([])]
    data = np.vstack([trace.data for trace in stQ])
    distance = np.asarray([trace.stats.distance for trace in stQ])
    return [data, distance]


def from_npArray2streamV2(SR_data,Q_stepping_index, Qtimming, Qsampling, distance,):
    """Convert a DAS matrix to an ObsPy stream while retaining fiber distance
    """
    if len(distance) != np.shape(SR_data)[0]:
        raise ValueError("Distance and strain-rate arrays have incompatible sizes")
    if Q_stepping_index <= 0:
        raise ValueError("Q_stepping_index must be positive")

    stream = Stream()
    indices = np.arange(0, np.shape(SR_data)[0], Q_stepping_index)
    for ii in indices:
        trace = Trace()
        trace.data = SR_data[ii, :]
        trace.stats.starttime = UTCDateTime(Qtimming)
        trace.stats.distance = distance[ii]
        trace.stats.location = "00"
        trace.stats.network = "KB"
        trace.stats.station = f"{int(distance[ii]):04d}"
        trace.stats.channel = "SR"
        trace.stats.sampling_rate = Qsampling
        stream += trace

    return [stream, np.asarray(distance)[indices]]


# -----------------------------------------------------------------------------
# Multiprocessing / array helpers
# -----------------------------------------------------------------------------
def multiprocess_iterable_on_dates(start_date, end_date, n_processes, section):
    """Partition an inclusive date range into tracking worker arguments
    """
    if n_processes <= 0:
        raise ValueError("n_processes must be positive")

    date_list = generate_date_range(start_date, end_date)
    n_days = len(date_list) // n_processes
    remainder = len(date_list) % n_processes

    if n_days > 1:
        multi_args = []
        for p in range(remainder):
            multi_args.append(
                [section,
                    date_list[(n_days + 1) * p],
                    date_list[(n_days + 1) * (p + 1) - 1]])
        for p in range(remainder, n_processes):
            multi_args.append(
                [section,
                    date_list[n_days * p + remainder],
                    date_list[n_days * (p + 1) + remainder - 1] ])
    else:
        multi_args = [[section, day, day] for day in date_list]
    return multi_args


def multiprocess_iterable_on_sections(start_date, end_date, section_list):
    """Create one cross-correlation worker argument set per fiber section
    """
    return [[section, start_date, end_date] for section in section_list]


def diagonal_shift(arr, shift_vel, dx, dt):
    """Shift each spatial trace in time according to a constant apparent velocity


    """
    if shift_vel == 0:
        raise ValueError("shift_vel must be non-zero")
    shift_factor = dx / (shift_vel * dt)
    shifted_arr = arr.copy()
    for i in range(shifted_arr.shape[0]):
        shift = int(i * shift_factor)
        shifted_arr[i] = np.roll(shifted_arr[i], shift)
    return shifted_arr
