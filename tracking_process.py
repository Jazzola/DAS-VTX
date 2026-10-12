"""Moving-source tracking and window extraction

"""

from __future__ import annotations

import logging

import numpy as np

from config import (DAS_FILE_FORMAT,
    DATE_FORMAT,
    PROCESSED_DIR,
    dx,
    length_sw,
    temporal_spacing,
    tracking_args,
    wlen_sw)
from Tracker_class import Tracker
from utils import get_date_from_file_path

logger = logging.getLogger(__name__)


def tracking_process(data_loader, section):
    """Track vehicles and save isolated surface-wave windows

    Parameters
    ----------
    data_loader : DataLoader
        Loader already configured for the requested date interval.
    section : tuple[float, float, float]
        ``(start, pivot, end)`` distances [m] defining the tracking section and
        the center of the extracted surface-wave aperture.
    """
    start_ch, pivot_ch, end_ch = section
    output_prefix = f"from_offset{start_ch}to{end_ch}_"

    for _ in range(data_loader.nfiles):
        try:
            (data, distance,  t_axis, data_attributes, file_path, decimated_files) = data_loader.get_next_data()
        except (ValueError, StopIteration) as exc:
            logger.warning("Skipping input file: %s", exc)
            continue

        date = get_date_from_file_path(file_path, DAS_FILE_FORMAT)
        logger.info("Processing tracking file: %s", file_path.stem)

        output_dir = PROCESSED_DIR / "detects" / date.strftime(DATE_FORMAT)
        output_dir.mkdir(parents=True, exist_ok=True)

        detections = []
        for decimated_file in decimated_files:
            tracker = Tracker(decimated_file, dx)
            tracker.track_cars(start_ch, end_ch, tracking_args, show_plot=False)
            tracker.select_surface_wave_windows(data,distance,  t_axis,
                pivot_ch,
                wlen_sw=wlen_sw,
                length_sw=length_sw,
                temporal_spacing=temporal_spacing)

            n_detections = len(tracker.sw_selector)
            logger.info("Isolated detections: %d", n_detections)
            if n_detections:
                detections.extend(tracker.sw_selector)

        if detections:
            file_output_dir = output_dir / file_path.stem
            file_output_dir.mkdir(parents=True, exist_ok=True)
            for detection in detections:
                detection_name = f"{output_prefix}t_{round(detection.t_axis[0])}"
                np.save(file_output_dir / detection_name, detection, allow_pickle=True)
        else:
            logger.info("No isolated detections for %s", file_path.name)

        # Large raw arrays are intentionally released between hourly files.
        del data, distance, t_axis, data_attributes, decimated_files, detections
