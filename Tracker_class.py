"""Moving-source detection and trajectory tracking for DAS-VTX

The tracker preprocesses decimated DAS data to emphasize coherent moving-source
signatures, detects candidate crossings near the beginning of a tracking
section, and propagates those candidates along the fiber with the original
Kalman-filter formulation. Trajectories are then screened using the configured
coverage, speed, curvature, and speed-fluctuation criteria


"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import butter, find_peaks, resample_poly, savgol_filter, sosfiltfilt

from config import R, nx_init, preprocessing_dict, preselection_dict, reverse_amp, sigma_a
from func_data_imports import read_das_npz
from SW_class import SurfaceWaveSelector
from utils import (average_speed_crit,
    curve_break_crit,
    diagonal_shift,
    fk_velocity_filter,
    interp_nan_value,
    likelihood_1d,
    max_adjacent_nan_crit,
    max_total_nan_crit,
    speed_fluctuations_crit)

logger = logging.getLogger(__name__)

# Candidate peaks are associated with a predicted trajectory when they fall
# within this many time-sample indices of the Kalman prediction. This is kept
# as a named constant to expose an existing implementation choice without
# changing it.
TRACKING_SEARCH_RADIUS_SAMPLES = 100


class Tracker:
    """Preprocess DAS data and estimate moving-source trajectories
    
    """

    def __init__(self, decimated_data_file, dx):
        self.dx = dx
        self.decimated_data_file = decimated_data_file
        self._preprocess_for_tracking()


    def _preprocess_for_tracking(self):
        
        """Apply the configured preprocessing chain used before peak tracking
        
        """
        
        dx = self.dx
        data_for_tracking, x_axis_tracking, t_axis_tracking = read_das_npz(self.decimated_data_file)
        self.t_axis_tracking = t_axis_tracking
        self.dt = t_axis_tracking[1] - t_axis_tracking[0]

        # Optional spatial interpolation. resample_poly acts along axis 0 by
        # default, which is the channel axis in DAS-VTX arrays.
        param_x_inter = preprocessing_dict.get("x_inter", (240, 25))
        if param_x_inter:
            data_for_tracking = resample_poly(  data_for_tracking,
                param_x_inter[0],
                param_x_inter[1] )
            dist_along_fiber_tracking = (  np.arange(data_for_tracking.shape[0])
                * dx
                * param_x_inter[1]
                / param_x_inter[0]
                + x_axis_tracking[0])
        else:
            dist_along_fiber_tracking = ( np.arange(data_for_tracking.shape[0]) * dx + x_axis_tracking[0])

        # Optional apparent-velocity selection in the f-k domain.
        param_fk = preprocessing_dict.get("FK", {})
        if param_fk:
            data_for_tracking, _, _, _ = fk_velocity_filter(data_for_tracking, dist_along_fiber_tracking,
                t_axis_tracking, param_fk)

        # Temporal band-pass filter.
        param_bp = preprocessing_dict.get("BP", {})
        if param_bp:
            freq_lo = param_bp.get("freq_lo", 0.2)
            freq_hi = param_bp.get("freq_hi", 1.0)
            sos = butter(5, [freq_lo, freq_hi],   btype="bandpass", output="sos", fs=1 / self.dt)
            data_for_tracking = sosfiltfilt(sos, data_for_tracking, axis=1)

        # Preserve the original Savitzky-Golay call: the configured tuple maps
        # to (window_length, polynomial_order), and filtering is along time.
        param_smoothing = preprocessing_dict.get("smoothing", (21, 15))
        if param_smoothing:
            data_for_tracking = savgol_filter( data_for_tracking,  param_smoothing[0], param_smoothing[1])

        # Align several nearby trial velocities, spatially average, shift back,
        # then average the three results. This increases the continuity of
        # moving-source signatures before peak picking.
        spatial_av_vel = preprocessing_dict.get("spatial_av_vel", False)
        if spatial_av_vel:
            av_win = preprocessing_dict.get("av_win", 5)
            aligned = []
            for velocity_factor in (1.0, 1.1, 0.9):
                velocity = spatial_av_vel * velocity_factor
                shifted = diagonal_shift(  data_for_tracking,
                    -velocity, self.dx, self.dt )
                for time_index in range(shifted.shape[1]):
                    shifted[:, time_index] = np.convolve( shifted[:, time_index], np.ones(av_win), mode="same")
                aligned.append( diagonal_shift(shifted, velocity, self.dx, self.dt))
            data_for_tracking = sum(aligned) / len(aligned)

        # Square-root amplitude weighting suppresses the dominance of very
        # strong traces while retaining sign information.
        if preprocessing_dict.get("SQRT", False):
            for j in range(data_for_tracking.shape[1]):
                maximum = np.max(np.abs(data_for_tracking[:, j]))
                if maximum > 0:
                    data_for_tracking[:, j] *= np.sqrt(
                        np.abs(data_for_tracking[:, j]) / maximum
                    )
            for i in range(data_for_tracking.shape[0]):
                maximum = np.max(np.abs(data_for_tracking[i, :]))
                if maximum > 0:
                    data_for_tracking[i, :] *= np.sqrt(
                        np.abs(data_for_tracking[i, :]) / maximum
                    )

        # Preserve the original nonlinear attenuation of amplitudes below the
        # 98th percentile of the signed data distribution.
        percentile_98 = np.percentile(data_for_tracking, 98)
        if percentile_98 != 0:
            mask = np.abs(data_for_tracking) < percentile_98
            reduced_array = np.square(data_for_tracking[mask] / percentile_98)
            data_for_tracking[mask] *= (
                np.sign(data_for_tracking[mask]) * reduced_array)

        # Oversample the spatial axis after temporarily straightening the
        # moving signature. The additional /3.6 factor is retained exactly from
        # the reference implementation.
        oversampling_factor = preprocessing_dict.get("oversampling_factor", 5)
        if oversampling_factor > 1:
            if not spatial_av_vel:
                raise ValueError( "oversampling_factor > 1 requires a non-zero spatial_av_vel")
            data_for_tracking = diagonal_shift(
                data_for_tracking,
                -spatial_av_vel / 3.6,
                self.dx,
                self.dt)
            interpolator = RegularGridInterpolator((dist_along_fiber_tracking, t_axis_tracking),
                data_for_tracking)
            new_distance = np.linspace( dist_along_fiber_tracking[0],
                dist_along_fiber_tracking[-1],
                oversampling_factor * len(dist_along_fiber_tracking))
            xx, yy = np.meshgrid(new_distance, t_axis_tracking, indexing="ij")
            data_for_tracking = interpolator((xx, yy))
            dist_along_fiber_tracking = new_distance
            data_for_tracking = diagonal_shift(data_for_tracking,
                spatial_av_vel / 3.6,
                self.dx / oversampling_factor,
                self.dt)

        self.data_for_tracking = data_for_tracking
        self.dist_along_fiber_tracking = dist_along_fiber_tracking
        self.dx = (
            dist_along_fiber_tracking[1] - dist_along_fiber_tracking[0]
        )

    def _detect_in_one_section( self,
        start_x, nx,
        sigma, detection_args=None,
        show_plot=False):
        
        """Detect initial trajectory times from the first nx spatial samples
        """

        t_axis = self.t_axis_tracking
        x_axis = self.dist_along_fiber_tracking
        detection_args = detection_args or self.tracking_args["detect"]

        minprominence = detection_args["minprominence"]
        minseparation = detection_args["minseparation"]
        prominence_window = detection_args["prominenceWindow"]

        peak_likelihood = np.zeros(len(t_axis))
        start_x_idx = np.argmin(np.abs(start_x - x_axis))
        end_seed_idx = min(start_x_idx + nx, self.data_for_tracking.shape[0])

        for i in range(start_x_idx, end_seed_idx):
            trace = self.data_for_tracking[i]
            peak_loc = find_peaks(trace,  prominence=minprominence,
                wlen=prominence_window,
                distance=minseparation)[0]
            peak_likelihood += likelihood_1d(peak_loc, t_axis, sigma)

        if peak_likelihood.size == 0:
            return np.array([], dtype=int)

        vehicle_base, _ = find_peaks( peak_likelihood, height=np.max(peak_likelihood) * 0.0,
            distance=minseparation )

        if show_plot:
            fig, axes = plt.subplots( 1, 2,figsize=(15, 10),
                gridspec_kw={"width_ratios": [3, 1]},sharey=True)
            axes[0].imshow(
                self.data_for_tracking.T,
                aspect="auto",
                extent=[self.dist_along_fiber_tracking[0],
                    self.dist_along_fiber_tracking[-1],
                    self.t_axis_tracking[-1],
                    self.t_axis_tracking[0]],
                cmap="gray")
            axes[0].axvline(self.dist_along_fiber_tracking[start_x_idx], c="r")
            axes[0].axvline(self.dist_along_fiber_tracking[end_seed_idx - 1], c="b")
            axes[1].plot(peak_likelihood, self.t_axis_tracking, "b")
            axes[1].plot(peak_likelihood[vehicle_base], self.t_axis_tracking[vehicle_base],"r^")
            middle_idx = start_x_idx + max(0, end_seed_idx - start_x_idx - 1) // 2
            axes[0].plot( [self.dist_along_fiber_tracking[middle_idx]] * len(vehicle_base),  self.t_axis_tracking[vehicle_base],"r^")

        return vehicle_base

    def _remove_unrealistic_tracking(self, veh_base, veh_states, dt, dx):
        """Apply the configured trajectory-quality criteria in two passes
        """
        rejection_counts = [0, 0, 0, 0, 0]

        # First pass: remove trajectories with insufficient temporal coverage.
        invalid_indices = []
        for v in range(len(veh_base)):
            vehicle_state = veh_states[v]
            invalid = False

            value = preselection_dict.get("max_adjacent_nan", False)
            if value:
                criterion = max_adjacent_nan_crit(vehicle_state, value)
                invalid = invalid or criterion
                rejection_counts[0] += int(criterion)

            portion = preselection_dict.get("max_total_nan", False)
            if portion:
                criterion = max_total_nan_crit(vehicle_state, portion)
                invalid = invalid or criterion
                rejection_counts[1] += int(criterion)

            if invalid:
                invalid_indices.append(v)

        valid_indices = np.setdiff1d(np.arange(len(veh_base)), invalid_indices)
        veh_states = veh_states[valid_indices, :]

        # Second pass: apply kinematic/shape criteria. Invalid numerical states
        # are rejected and logged rather than silently accepted.
        invalid_indices = []
        for v in range(veh_states.shape[0]):
            vehicle_state = veh_states[v]
            invalid = False
            try:
                speed_values = preselection_dict.get("average_speed", False)
                if speed_values:
                    criterion = average_speed_crit(vehicle_state, dx, dt, speed_values)
                    invalid = invalid or criterion
                    rejection_counts[2] += int(criterion)

                win_size, factor, portion, q = preselection_dict.get("curve_break", (False, False, False, False))
                if win_size:
                    criterion = curve_break_crit(vehicle_state, win_size, factor, portion, q)
                    invalid = invalid or criterion
                    rejection_counts[3] += int(criterion)

                factor, portion = preselection_dict.get("speed_fluctuations", (False, False))
                if factor:
                    criterion = speed_fluctuations_crit(vehicle_state, dx, dt, factor, portion)
                    invalid = invalid or criterion
                    rejection_counts[4] += int(criterion)

                if invalid:
                    invalid_indices.append(v)
            except (ValueError, IndexError, ZeroDivisionError, FloatingPointError) as exc:
                logger.debug("Rejecting numerically invalid trajectory %d: %s", v, exc)
                invalid_indices.append(v)

        valid_indices = np.setdiff1d(
            np.arange(veh_states.shape[0]), invalid_indices)
        tracked_vehicles = veh_states[valid_indices, :]
        logger.info(
            "Trajectory rejection counts [adjacent_nan, total_nan, speed, curve, fluctuation]: %s",
            rejection_counts)
        return tracked_vehicles

    def _tracking_with_veh_base(self,
        start_x, end_x, veh_base,
        sigma_a=0.01, R=1, detection_args=None):
        
        """Propagate initial detections along the fiber with the Kalman filter
        """
        x_axis = self.dist_along_fiber_tracking
        detection_args = detection_args or self.tracking_args["detect"]

        start_x_idx = np.argmin(np.abs(start_x - x_axis))
        end_x_idx = np.argmin(np.abs(end_x - x_axis))
        if end_x_idx < start_x_idx:
            start_x_idx, end_x_idx = end_x_idx, start_x_idx
        local_x_axis = x_axis[start_x_idx : end_x_idx + 1]

        minprominence = detection_args["minprominence"]
        minseparation = detection_args["minseparation"]
        prominence_window = detection_args["prominenceWindow"]

        veh_states = np.full( (len(veh_base), end_x_idx - start_x_idx + 1), np.nan)
        
        tk1k = np.full((2, len(veh_base)), np.nan)
        tkk = np.full((2, len(veh_base)), np.nan)
        pkk = np.full((2, 2, len(veh_base)), np.nan)
        pk1k = np.full((2, 2, len(veh_base)), np.nan)
        xv = np.full(len(veh_base), np.nan)

        observation = np.array([1, 0])
        veh_base_state = veh_base.copy().astype(float)

        for i in range(start_x_idx, end_x_idx + 1):
            local_i = i - start_x_idx
            for v in range(len(veh_base)):
                n_valid = np.count_nonzero(~np.isnan(veh_states[v, :]))

                if n_valid == 0:
                    veh_base_state[v] = veh_base[v]
                elif n_valid == 1:
                    # Preserve the reference initialization: the trajectory
                    # slope/state derivative starts at zero.
                    first_state = veh_states[v, ~np.isnan(veh_states[v, :])][0]
                    tkk[:, v] = [first_state, 0]
                    pkk[:, :, v] = np.zeros((2, 2))
                    xv[v] = local_x_axis[0]
                    veh_base_state[v] = veh_base[v]
                else:
                    delta_x = self.dist_along_fiber_tracking[i] - xv[v]
                    transition = np.array([[1, delta_x], [0, 1]])
                    process_covariance = sigma_a * np.array(
                        [[0.25 * delta_x**4, 0.5 * delta_x**3],
                            [0.5 * delta_x**3, delta_x**2]])
                    tk1k[:, v] = transition @ tkk[:, v]
                    pk1k[:, :, v] = ( transition @ pkk[:, :, v] @ transition.T
                        + process_covariance)
                    veh_base_state[v] = tk1k[0, v]

            trace = self.data_for_tracking[i]
            peak_loc, _ = find_peaks(
                trace,
                prominence=minprominence,
                wlen=prominence_window,
                distance=minseparation)

            for p in range(len(veh_base_state)):
                distances = peak_loc - veh_base_state[p]
                candidate_indices = np.where((distances > -TRACKING_SEARCH_RADIUS_SAMPLES)
                    & (distances <= TRACKING_SEARCH_RADIUS_SAMPLES))[0]

                if candidate_indices.size:
                    closest = candidate_indices[
                        np.argmin(np.abs(distances[candidate_indices]))]
                    veh_states[p, local_i] = peak_loc[closest]

            # Kalman correction when a trajectory has accumulated enough
            # observations and the current channel contains an associated peak.
            for v in range(len(veh_base)):
                n_valid = np.count_nonzero(~np.isnan(veh_states[v, :]))
                if n_valid > 2 and not np.isnan(veh_states[v, local_i]):
                    innovation_var = R + observation @ pk1k[:, :, v] @ observation.T
                    if innovation_var == 0:
                        continue
                    kalman_gain = (
                        pk1k[:, :, v] @ observation.T / innovation_var
                    )
                    corrected = tk1k[:, v] + kalman_gain * (
                        veh_states[v, local_i] - observation @ tk1k[:, v]
                    )
                    tkk[:, v] = corrected
                    pkk[:, :, v] = pk1k[:, :, v] - (
                        kalman_gain.reshape(2, 1)
                        @ observation.reshape(1, 2)
                    ) @ pk1k[:, :, v]
                    xv[v] = self.dist_along_fiber_tracking[i]

        logger.info("Trajectory matrix before preselection: %s", veh_states.shape)
        tracked_vehicles = self._remove_unrealistic_tracking( veh_base, veh_states, self.dt, self.dx )
       
        return interp_nan_value(tracked_vehicles)

    def track_cars( self,
        start_x,end_x,
        tracking_args, show_plot=True,reverse_amp=reverse_amp,
        sigma_a=sigma_a,R=R,nx=nx_init):
        """Detect and track moving sources over the requested fiber interval
        
        """
        self.start_x = start_x
        self.end_x = end_x
        self.tracking_args = tracking_args

        if reverse_amp:
            self.data_for_tracking = -self.data_for_tracking

        veh_base = self._detect_in_one_section( start_x=self.start_x,
            nx=nx,
            sigma=sigma_a,
            show_plot=show_plot )
        
        self.veh_states = self._tracking_with_veh_base( start_x=self.start_x,
            end_x=self.end_x,
            veh_base=veh_base,
            sigma_a=sigma_a,
            R=R)

    def select_surface_wave_windows(self, data_raw, x_axis, t_axis, x0, **kwargs):
        """Create isolated surface-wave windows around the tracked sources
        
        """
        self.sw_selector = SurfaceWaveSelector( data_raw, x_axis, t_axis, x0,
            self.data_for_tracking,
            self.start_x,
            self.veh_states,
            self.dist_along_fiber_tracking,
            self.t_axis_tracking,
            **kwargs)
