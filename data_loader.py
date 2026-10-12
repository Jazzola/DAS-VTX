"""Data discovery and loading utilities for DAS-VTX tracking
The :class:`DataLoader` scans the configured raw DAS archive, reuses existing
tracking-decimated files when available, and creates them on demand otherwise.
Raw data are still loaded for surface-wave window extraction; decimated files
are used only by the tracking stage
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from pathlib import Path

from config import (
    DAS_FILE_FORMAT,
    DATE_FORMAT,
    DECIM_FILE_FORMAT,
    datapath,
    decimateddatapath,
    length_sw,
    tracking_data_decimation_factor,
)
from func_data_imports import data_matrix_import
from utils import create_npz_data, generate_date_range, get_date_from_file_path

logger = logging.getLogger(__name__)


class DataLoader:
    """Locate raw/decimated DAS files for one tracking section

    Parameters
    ----------
    section : tuple[float, float, float]
        Tracking section expressed as ``(start, pivot, end)`` distances along
        the fiber [m]. The pivot is used to center the raw-data aperture loaded
        for surface-wave extraction
    """

    def __init__(self, section):
        self.data_dir = Path(datapath)
        self.decimated_data_dir = Path(decimateddatapath)
        self.pivot_channel = section[1]
        self.file_paths: list[Path] = []
        self.nfiles = 0

    def scan_data(
        self,
        start_date: str,
        end_date: str,
        start_hour: int,
        end_hour: int,
    ) -> None:
        """Scan raw DAS files over the requested date/hour interval.

        The hour interval follows the convention ``start_hour <= hour <
        end_hour`` and is interpreted in the time system encoded by
        ``DAS_FILE_FORMAT`` (UTC for the provided configurations)
        """
        logger.info(
            "Scanning DAS files from %s to %s between %02d:00 and %02d:00",
            start_date,
            end_date,
            start_hour,
            end_hour,
        )

        self.file_paths = []
        for date in generate_date_range(start_date, end_date):
            file_day_dir = self.data_dir / date
            if not file_day_dir.exists():
                logger.warning("No DAS directory for %s: %s", date, file_day_dir)
                continue

            for entry in os.scandir(file_day_dir):
                if not entry.is_file():
                    continue
                file_path = file_day_dir / entry.name
                file_date = get_date_from_file_path(file_path, DAS_FILE_FORMAT)
                if start_hour <= file_date.hour < end_hour:
                    self.file_paths.append(file_path)

        self.file_paths.sort()
        self.nfiles = len(self.file_paths)
        logger.info("Found %d raw DAS files", self.nfiles)

    def get_next_data(self):
        """Load the next raw DAS file and its matching decimated chunks

        Decimated tracking files are created automatically when no matching
        chunk exists for the raw-file hour. The returned raw matrix is limited
        to an aperture centered on the tracking pivot.
        """
        if not self.file_paths:
            raise StopIteration("No unprocessed DAS files remain in DataLoader")

        file_path = self.file_paths.pop(0)
        file_date = get_date_from_file_path(file_path, DAS_FILE_FORMAT)
        decimated_day_dir = self.decimated_data_dir / file_date.strftime(DATE_FORMAT)
        decimated_day_dir.mkdir(parents=True, exist_ok=True)

        corresponding_decimated_files = self._matching_decimated_files(
            decimated_day_dir, file_date
        )
        if corresponding_decimated_files:
            logger.info("Using existing decimated tracking data for %s", file_path.name)
        else:
            logger.info("Creating decimated tracking data for %s", file_path.name)
            create_npz_data(
                file_path,
                decimated_day_dir,
                decimation_factor=tracking_data_decimation_factor,
                return_full=False,
            )
            corresponding_decimated_files = self._matching_decimated_files(
                decimated_day_dir, file_date
            )

        strain_rate, distance, time, data_attributes = data_matrix_import(
            file_path,
            ch1=self.pivot_channel - length_sw // 2,
            ch2=self.pivot_channel + length_sw // 2,
        )
        return (
            strain_rate,
            distance,
            time,
            data_attributes,
            file_path,
            corresponding_decimated_files,
        )

    @staticmethod
    def _matching_decimated_files(directory: Path, file_date):
        """Return decimated chunks whose start time falls within one raw hour
        """
        files = []
        for entry in os.scandir(directory):
            if not entry.is_file():
                continue
            decimated_date = get_date_from_file_path(entry, DECIM_FILE_FORMAT)
            if file_date <= decimated_date < file_date + timedelta(hours=1):
                files.append(entry.path)
        files.sort()
        return files
