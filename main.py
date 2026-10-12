"""Entry point for the DAS-VTX processing workflow

DAS-VTX is organized as three sequential but independently switchable stages:
tracking, cross-correlation/VSG construction, and dispersion interpretation.
The active once, geometry, and parameters are controlled by the selected 
configuration module
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

# Keep the original configuration mechanism for compatibility: the selected
# experiment-specific file is copied to ``config.py`` before modules that import
# ``config`` are loaded. The environment variable provides a convenient way to
# switch configuration without editing this file.
CONFIG_TO_USE = os.environ.get("DAS_VTX_CONFIG", "trains").lower()
if CONFIG_TO_USE not in {"cars", "trains"}:
    raise ValueError("DAS_VTX_CONFIG must be either 'cars' or 'trains'")

_REPO_DIR = Path(__file__).resolve().parent
shutil.copy2(_REPO_DIR / f"config_{CONFIG_TO_USE}.py", _REPO_DIR / "config.py")

from config import (  # noqa: E402  (config must be selected before import)
    PROCESSED_DIR,
    RUN_INTERPRETATION,
    RUN_TRACKING,
    RUN_XCORR,
    end_hour,
    n_processes,
    stack_files_list,
    start_hour,
    tracking_end_date,
    tracking_sections,
    tracking_start_date,
    xcorr_end_date,
    xcorr_sections,
    xcorr_start_date,
)
from data_loader import DataLoader  # noqa: E402
from interpretation_process import interpretation_process  # noqa: E402
from tracking_process import tracking_process  # noqa: E402
from utils import multiprocess_iterable_on_dates, multiprocess_iterable_on_sections  # noqa: E402
from xcorr_process import xcorr_process  # noqa: E402

logger = logging.getLogger(__name__)


def run_tracking_analysis(args):
    """Worker wrapper used to parallelize tracking over date interval
    """
    section, start_date, end_date = args
    loader = DataLoader(section)
    loader.scan_data(start_date, end_date, start_hour, end_hour)
    tracking_process(loader, section)


def run_xcorr_analysis(args):
    """Worker wrapper used to parallelize VSG construction over sections
    """
    section, start_date, end_date = args
    return xcorr_process(section, start_date, end_date)


def _iter_precomputed_stacks(stack_source):
    """Yield explicitly configured stack files for interpretation-only runs
    """
    if isinstance(stack_source, (list, tuple)):
        yield from stack_source
        return

    source = Path(stack_source)
    if source.is_file():
        yield source
    elif source.is_dir():
        for entry in sorted(source.iterdir()):
            if entry.is_file() and entry.suffix == ".npz":
                yield entry
    else:
        raise FileNotFoundError(f"Stack source does not exist: {source}")


def main():
    """Execute the processing stages enabled
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = PROCESSED_DIR / f"{timestamp}.log"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("DAS-VTX started with '%s' configuration", CONFIG_TO_USE)

    stack_files = []

    if RUN_TRACKING:
        (PROCESSED_DIR / "detects").mkdir(parents=True, exist_ok=True)
        for section in tracking_sections:
            args = multiprocess_iterable_on_dates(
                tracking_start_date,
                tracking_end_date,
                n_processes,
                section,
            )
            with Pool(n_processes) as pool:
                pool.map(run_tracking_analysis, args)

    if RUN_XCORR:
        args = multiprocess_iterable_on_sections(
            xcorr_start_date,
            xcorr_end_date,
            xcorr_sections,
        )
        with Pool(n_processes) as pool:
            per_section_files = pool.map(run_xcorr_analysis, args)
        stack_files = [item for sublist in per_section_files for item in sublist]

    if RUN_INTERPRETATION:
        if RUN_XCORR:
            interpretation_inputs = stack_files
        elif stack_files_list is not None:
            interpretation_inputs = list(_iter_precomputed_stacks(stack_files_list))
        else:
            interpretation_inputs = []

        if not interpretation_inputs:
            logger.warning("No stack files found; skipping interpretation")
        for stack_file in interpretation_inputs:
            interpretation_process(stack_file)

    logger.info("DAS-VTX finished")


if __name__ == "__main__":
    main()
