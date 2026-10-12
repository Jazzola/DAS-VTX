"""DAS input helpers
The HDF5 reader is intentionally isolated in this module because acquisition
formats and interrogation-unit libraries vary between deployments. Adapting
DAS-VTX to another vendor/file format should therefore primarily require
changes here rather than in the processing workflow
"""

from __future__ import annotations

import logging

import numpy as np
from obspy.core import UTCDateTime
from febus_optics_lib.reader import H5ReaderDas

logger = logging.getLogger(__name__)


def read_das_npz(fname, **kwargs):
    """Read a DAS matrix from a decimated .npz tracking file

    Parameters
    ----------
    fname : str or pathlib.Path
        Input archive containing ``data``, ``x_axis`` and ``t_axis`` arrays.
    ch1, ch2 : float, optional
        Minimum and maximum fiber distances [m] retained from ``x_axis``.

    Returns
    -------
    data, x_axis, t_axis : numpy.ndarray
        DAS matrix and associated spatial/time axes.
    """
    try:
        with np.load(fname) as data_file:
            data = data_file["data"]
            x_axis = data_file["x_axis"]
            t_axis = data_file["t_axis"]
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"Could not read DAS npz file: {fname}") from exc

    ch1 = kwargs.get("ch1", x_axis[0])
    ch2 = kwargs.get("ch2", x_axis[-1])
    if ch1 > ch2:
        raise ValueError("ch1 must be smaller than or equal to ch2")

    ch1_idx = np.argmax(x_axis >= ch1)
    ch2_idx = np.argmax(x_axis >= ch2)
    data = data[ch1_idx : ch2_idx + 1, :]
    x_axis = x_axis[ch1_idx : ch2_idx + 1]
    return data, x_axis, t_axis


def data_matrix_import(filepath, spatial_sampling=9.6, starttime=None, endtime=None, maxdist=25000, mindist=40, **kwargs):
    """Read a raw Febus DAS file as a channel-by-time strain-rate matrix

    Parameters
    ----------
    filepath : str or pathlib.Path
        Raw HDF5 DAS file.
    spatial_sampling : float, default 9.6
        Expected channel spacing [m]. This argument is retained for compatibility
        and diagnostic reporting; the spatial axis is read from the file.
    starttime, endtime : optional
        Relative extraction limits passed to the Febus reader.
    maxdist, mindist : float
        Allowed fiber-distance bounds [m] for the current deployment.
    ch1, ch2 : float, optional
        Requested fiber-distance bounds [m].

    Returns
    -------
    strain_rate : numpy.ndarray
        DAS strain-rate matrix with shape ``(n_channels, n_samples)``.
    distance : numpy.ndarray
        Distance along fiber [m].
    time : numpy.ndarray
        Relative time axis [s].
    data_attributes : list
        Acquisition metadata retained for the downstream ObsPy conversion.
    """
    ch1 = kwargs.get("ch1", mindist)
    ch2 = kwargs.get("ch2", maxdist)
    if ch1 < mindist or ch2 > maxdist or ch1 > ch2:
        raise ValueError(
            f"Requested distance range [{ch1}, {ch2}] m is outside "
            f"the supported interval [{mindist}, {maxdist}] m.")

    instance = H5ReaderDas(filepath)
    first_zone = instance.list_zones[0]
    concat_results = instance.extract_concat(from_time=starttime, to_time=endtime, time_type="relative", from_dist=ch1, to_dist=ch2,dist_type="meter", zones=first_zone)

    strain_rate = np.asarray(concat_results[first_zone]["data"]).T
    distance = np.asarray(concat_results[first_zone]["distance_vect"])
    dt = instance.param_dict[first_zone]["derivation_time"] / 1000.0

    data_attributes = [
        instance.param_dict[first_zone]["spacing"][1],
        instance.param_dict[first_zone]["spacing"][0],
        instance.param_dict[first_zone]["gauge_length"],
        instance.param_dict[first_zone]["derivation_time"],
        instance.param_dict[first_zone]["pulse_width"],
        instance.param_dict[first_zone]["sampling_res"],
        instance.param_dict[first_zone]["pulse_rate_freq"],
        UTCDateTime(concat_results[first_zone]["time_vect"][0]),
    ]

    time = np.arange(strain_rate.shape[1]) * dt
    logger.debug("H5 data attributes: %s", data_attributes)
    logger.debug("Configured spatial sampling: %s m", spatial_sampling)
    return strain_rate, distance, time, data_attributes
