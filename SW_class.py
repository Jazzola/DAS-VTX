"""Surface-wave window containers and moving-source window selection


Tracked vehicle trajectories are converted into isolated spatio-temporal
windows around a configurable pivot position. These windows form the input to
the trajectory-guided cross-correlation stage

"""

from __future__ import annotations

import copy

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from utils import plot_data


class SurfaceWaveWindow:
    """Store one isolated surface-wave window and its tracked trajectory
    """

    def __init__(self, data, x_axis, t_axis, veh_state, start_x_tracking, distance_along_fiber_tracking, t_axis_tracking):
        self.data = data
        self.x_axis = x_axis
        self.t_axis = t_axis
        self.veh_state = veh_state
        self.start_x_tracking = start_x_tracking
        self.distance_along_fiber_tracking = distance_along_fiber_tracking
        self.t_axis_tracking = t_axis_tracking
        self._preprocess_veh_state()


    def _preprocess_veh_state(self):
        
        """Convert trajectory sample indices into physical distance/time axes
        """
        
        valid = ~np.isnan(self.veh_state)
        if not np.any(valid):
            raise ValueError("Surface-wave window trajectory contains only NaNs")

        time_indices = self.veh_state[valid].astype(int)
        if np.any(time_indices < 0) or np.any(time_indices >= self.t_axis_tracking.size):
            raise IndexError("Tracked trajectory contains time indices outside t_axis_tracking")

        start_idx = np.abs(
            self.start_x_tracking - self.distance_along_fiber_tracking
        ).argmin()
        distance_indices = np.where(valid)[0] + start_idx
        if np.any(distance_indices >= self.distance_along_fiber_tracking.size):
            raise IndexError("Tracked trajectory extends beyond the tracking distance axis")

        self.veh_state_x = self.distance_along_fiber_tracking[distance_indices]
        self.veh_state_t = self.t_axis_tracking[time_indices]

    def plot_on_data(self, ax, c="r"):
        
        """Draw the spatio-temporal extent of this window on an existing axis
        """
        length_sw = self.x_axis[-1] - self.x_axis[0]
        wlen_sw = self.t_axis[-1] - self.t_axis[0]
        rect = patches.Rectangle(
            (self.x_axis[0], self.t_axis[0]),
            length_sw,
            wlen_sw,
            linewidth=1,
            edgecolor=c,
            facecolor="none")
        ax.add_patch(rect)
        return rect

    def save_fig(self,fig_name=None, fig_dir="Fig/show_sw_time_windows/",t_min=None,t_max=None,x_min=None, x_max=None,    color="red"):
        
        """Plot the raw window with the tracked vehicle trajectory overlaid
        
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(self.veh_state_x, self.veh_state_t, ".", color=color, markersize=1)

        t_min = self.t_axis[0] if t_min is None else t_min
        t_max = self.t_axis[-1] if t_max is None else t_max
        x_min = self.x_axis[0] if x_min is None else x_min
        x_max = self.x_axis[-1] if x_max is None else x_max

        t_min_idx = np.abs(t_min - self.t_axis).argmin()
        t_max_idx = np.abs(t_max - self.t_axis).argmin()
        x_min_idx = np.abs(x_min - self.x_axis).argmin()
        x_max_idx = np.abs(x_max - self.x_axis).argmin()

        plot_data(self.data[x_min_idx : x_max_idx + 1, t_min_idx : t_max_idx + 1],
            x_axis=self.x_axis[x_min_idx : x_max_idx + 1],
            t_axis=self.t_axis[t_min_idx : t_max_idx + 1],
            fig_dir=fig_dir,
            fig_name=fig_name,
            ax=ax,
            pclip=90)
        
        return ax


class SurfaceWaveSelector:
    """Extract isolated surface-wave windows around tracked moving sources

    """

    def __init__(self,
        data_for_surface_wave,
        distances_along_fiber,
        t_axis,
        x0,
        data_for_tracking,
        start_x_tracking,
        veh_states,
        distance_along_fiber_tracking,
        t_axis_tracking,
        wlen_sw=8,
        length_sw=300,
        spatial_ratio=0.5,
        temporal_spacing=None):
        
        
        self.data_for_surface_wave = data_for_surface_wave
        self.data_for_tracking = data_for_tracking
        self.distances_along_fiber = distances_along_fiber
        self.t_axis = t_axis
        self.dt = self.t_axis[1] - self.t_axis[0]
        self.x0 = x0
        self.length_sw = length_sw
        self.start_x_tracking = start_x_tracking
        self.veh_states = veh_states
        self.distance_along_fiber_tracking = distance_along_fiber_tracking
        self.t_axis_tracking = t_axis_tracking
        self.dx_tracking = np.abs(distance_along_fiber_tracking[1] - distance_along_fiber_tracking[0])
        self.wlen_sw = wlen_sw
        self.spatial_ratio = spatial_ratio
        self.temporal_spacing = (self.wlen_sw if temporal_spacing is None else temporal_spacing)

        self.locate_windows()


    def locate_windows(self):
        """
        
        Identify isolated tracked events and construct their data windows
        
        """
        
        win_nsamp = int(self.wlen_sw / self.dt)
        x0_idx = int((self.x0 - self.start_x_tracking) // self.dx_tracking)
        if x0_idx < 0 or x0_idx >= self.veh_states.shape[1]:
            raise IndexError("x0 lies outside the tracked trajectory section")

        windows = []
        for k, vehicle_state in enumerate(self.veh_states):
            if np.isnan(vehicle_state[x0_idx]):
                continue
            t0_idx = int(vehicle_state[x0_idx])
            if not 0 <= t0_idx < self.t_axis_tracking.size:
                continue

            # Reject a following source that crosses the pivot too close in time.
            if k < len(self.veh_states) - 1:
                next_value = self.veh_states[k + 1, x0_idx]
                if not np.isnan(next_value):
                    next_idx = int(next_value)
                    if 0 <= next_idx < self.t_axis_tracking.size:
                        separation = np.abs(
                            self.t_axis_tracking[next_idx]
                            - self.t_axis_tracking[t0_idx])
                        if separation < self.temporal_spacing:
                            continue

            # Reject a preceding source that crosses the pivot too close in time.
            if k > 0:
                previous_value = self.veh_states[k - 1, x0_idx]
                if not np.isnan(previous_value):
                    previous_idx = int(previous_value)
                    if 0 <= previous_idx < self.t_axis_tracking.size:
                        delta_t = (
                            self.t_axis_tracking[t0_idx]
                            - self.t_axis_tracking[previous_idx]
                        )
                        if self.temporal_spacing > delta_t > 0:
                            continue

            t0 = self.t_axis_tracking[t0_idx]
            t0_sw_idx = np.abs(t0 - self.t_axis).argmin()
            if ( t0_sw_idx < win_nsamp // 2
                or t0_sw_idx + win_nsamp // 2 > self.t_axis.size):
                continue

            start_x = self.x0 - self.length_sw * self.spatial_ratio
            end_x = start_x + self.length_sw
            start_x_idx = np.abs(start_x - self.distances_along_fiber).argmin()
            end_x_idx = np.abs(end_x - self.distances_along_fiber).argmin()
            if end_x_idx <= start_x_idx:
                continue

            start_t_idx = t0_sw_idx - win_nsamp // 2
            end_t_idx = start_t_idx + win_nsamp
            if end_t_idx > self.t_axis.size:
                continue

            windows.append(
                SurfaceWaveWindow(
                    data=copy.deepcopy(
                        self.data_for_surface_wave[
                            start_x_idx:end_x_idx,
                            start_t_idx:end_t_idx,
                        ]
                    ),
                    t_axis=self.t_axis[start_t_idx:end_t_idx],
                    x_axis=self.distances_along_fiber[start_x_idx:end_x_idx],
                    veh_state=vehicle_state,
                    start_x_tracking=self.start_x_tracking,
                    distance_along_fiber_tracking=self.distance_along_fiber_tracking,
                    t_axis_tracking=self.t_axis_tracking,
                )
            )

        self.windows = windows


    def __len__(self):
        return len(self.windows)


    def __getitem__(self, item):
        return self.windows[item]


    def __setitem__(self, key, value):
        self.windows[key] = value


    def __contains__(self, item):
        return 0 <= item < len(self.windows)


    def save_figs(self, fig_dir="Fig/show_sw_time_windows/"):
        """Save diagnostic figures for all selected surface-wave windows
        
        """
        for k, window in enumerate(self):
            window.save_fig(fig_name=f"sw_car{k}.png", fig_dir=fig_dir)
