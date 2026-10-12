"""Cross-correlation and Virtual Shot Gather (VSG) stacking workflow

This module converts isolated moving-source surface-wave windows into virtual
shot gathers, stacks the gathers over a configurable date interval, and records
the evolution of a diagnostic SNR estimate as the stack grows.
"""

from __future__ import annotations

import copy
import logging
import os
from datetime import timedelta
from pathlib import Path
from random import shuffle

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey

from config import (
    DAS_FILE_FORMAT,
    DATE_TIME_FORMAT,
    PROCESSED_DIR,
    freq_hi,
    freq_lo,
    include_other_side,
    n_vsg_per_stack,
    randomize_vsg,
    sw_bp_filt,
    sw_whiten,
    taper,
    wlen_sw,
    xcorr_parameters,
)
from utils import (
    calculate_SNR,
    generate_date_range,
    get_date_from_file_path,
    get_file_section,
    whiten_signals,
)
from VSG_class import VirtualShotGather

logger = logging.getLogger(__name__)


def xcorr_process(section, start_date, end_date):
    """Build and stack VSGs for one fiber section and date interval

    Parameters
    ----------
    section : tuple[float, float, float]
        (start, pivot, end) distances [m] used for virtual-shot retrieval
    start_date, end_date : str
        Inclusive processing dates formatted according to DATE_FORMAT

    Returns
    -------
    list[str]
        Paths to the stacked VSG archives created by this call.

    Notes
    -----
    The implementation deliberately preserves the original workflow: each
    detected surface-wave window is tapered, optionally band-pass filtered and
    whitened, converted to a VSG, and recursively added to the current stack.
    SNR is evaluated after every successful addition.
    """
    start_ch, end_ch = section
    stacks_dir = PROCESSED_DIR / "VSGs" / "STACKs"
    stacks_dir.mkdir(parents=True, exist_ok=True)

    daily_dirs = []
    for date in generate_date_range(start_date, end_date):
        day_dir = PROCESSED_DIR / "detects" / date
        if day_dir.exists():
            daily_dirs.append(day_dir)
        else:
            logger.info("No detections found for %s", date)
    daily_dirs.sort()

    detection_files = []
    for output_day in daily_dirs:
        for detection_hour in os.scandir(output_day):
            if not detection_hour.is_dir():
                continue
            for detection in os.scandir(detection_hour.path):
                if not detection.is_file():
                    continue
                file_section = get_file_section(detection)
                if start_ch >= file_section[0] and end_ch <= file_section[1]:
                    detection_files.append(detection.path)

    detection_files.sort()
    if not detection_files:
        logger.warning(
            "No compatible detections for section %s between %s and %s",
            section,start_date,end_date)
        return []

    stack_files = []
    for parameters in xcorr_parameters:
        wlen = parameters.get("wlen", 1)
        overlap = parameters.get("overlap", 0.8)
        delta_t = parameters.get("delta_t", 0.5)
        time_window_to_xcorr = parameters.get("time_window_to_xcorr", 5)
        norm = parameters.get("norm", True)
        norm_amp = parameters.get("norm_amp", True)

        parameter_tag = ( f"st:{start_date}-end:{end_date}_o={overlap};dt={delta_t};"
            f"w={wlen};twin={time_window_to_xcorr}")

        # Chunking is intentionally performed before optional randomization to
        # preserve the original stacking behavior.
        if n_vsg_per_stack is None:
            detection_subsets = [list(detection_files)]
        else:
            if n_vsg_per_stack <= 0:
                raise ValueError("n_vsg_per_stack must be a positive integer or None")
            detection_subsets = [
                list(detection_files[i : i + n_vsg_per_stack])
                for i in range(0, len(detection_files), n_vsg_per_stack)]

        for subset_index, detection_subset in enumerate(detection_subsets):
            if randomize_vsg:
                shuffle(detection_subset)

            stack = None
            vsg_times = []
            snr_values = []
            from_off = None
            to_off = None

            for detection_file in detection_subset:
                surface_wave = np.load(detection_file, allow_pickle=True).item()
                from_off = int(surface_wave.veh_state_x[0])
                to_off = int(surface_wave.veh_state_x[-1])

                taper_window = tukey(surface_wave.data.shape[1], taper / wlen_sw)
                surface_wave.data *= taper_window[np.newaxis, :]

                dt = surface_wave.t_axis[1] - surface_wave.t_axis[0]
                data_filt = surface_wave.data
                if sw_bp_filt:
                    sos = butter( 5, [freq_lo, freq_hi], btype="bandpass", output="sos",
                        fs=1 / dt,)
                    data_filt = sosfiltfilt(sos, data_filt, axis=-1)

                if sw_whiten:
                    data_filt = whiten_signals(data_filt, freq_lo, freq_hi, fs=1 / dt)

                filtered_surface_wave = copy.deepcopy(surface_wave)
                filtered_surface_wave.data = data_filt

                try:
                    vsg = VirtualShotGather( filtered_surface_wave,start_x=start_ch,end_x=end_ch,wlen=wlen,
                        overlap=overlap,delta_t=delta_t,time_window_to_xcorr=time_window_to_xcorr,norm=norm,norm_amp=norm_amp,
                        include_other_side=include_other_side)
                except ValueError as exc:
                    logger.warning("Skipping %s: VSG calculation failed (%s)",
                        detection_file,
                        exc)
                    continue

                if np.isnan(vsg.XCF_out).any():
                    logger.warning("Skipping %s: NaNs in VSG (likely a window-boundary issue)",
                        detection_file )
                    continue

                stack = vsg if stack is None else stack + vsg
                snr_values.append(calculate_SNR(stack))

                detection_path = Path(detection_file)
                detection_time = get_date_from_file_path(detection_path.parent,
                    DAS_FILE_FORMAT.removesuffix(".h5"))
                detection_time += timedelta(seconds=int(detection_path.stem.split("t_")[-1]))
                vsg_times.append(detection_time.strftime(DATE_TIME_FORMAT))

            if stack is None:
                logger.warning("No valid VSGs in subset %d; nothing saved", subset_index)
                continue

            n_vsg = len(vsg_times)
            subset_str = f"{subset_index:03d}"
            n_vsg_str = f"{n_vsg:04d}"
            output_name = ( f"Stack_off={from_off}-{to_off}_n={n_vsg_str}_"
                f"Subset={subset_str}_{parameter_tag}")
            output_path = stacks_dir / output_name
            np.savez(output_path,
                parameters=parameters,
                SNR=snr_values,
                vsg_times=vsg_times,
                stack=stack)
            stack_files.append(str(output_path) + ".npz")
            logger.info("Saved VSG stack: %s.npz", output_path)

    return stack_files
