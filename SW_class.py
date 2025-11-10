import copy
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import numpy as np
from scipy import signal

from utils import plot_data

class SurfaceWaveWindow:
    def __init__(
            self,
            data,
            x_axis,
            t_axis,
            veh_state,
            start_x_tracking,
            distance_along_fiber_tracking,
            t_axis_tracking,
    ):
        self.data = data
        self.x_axis = x_axis
        self.t_axis = t_axis
        self.veh_state = veh_state
        self.start_x_tracking = start_x_tracking
        self.distance_along_fiber_tracking = distance_along_fiber_tracking
        self.t_axis_tracking = t_axis_tracking
        self._preprocess_veh_state()

    def _preprocess_veh_state(self):
        tmp = self.veh_state[~np.isnan(self.veh_state)].astype(int)
        start_x_tracking_idx = np.abs(self.start_x_tracking - self.distance_along_fiber_tracking).argmin()
        dist_idx_tmp = np.where(~np.isnan(self.veh_state))[0] + start_x_tracking_idx
        self.veh_state_x = self.distance_along_fiber_tracking[dist_idx_tmp]
        self.veh_state_t = self.t_axis_tracking[tmp]

    def plot_on_data(self, ax, c='r'):
        x_axis, t_axis = self.x_axis, self.t_axis
        length_sw = x_axis[-1] - x_axis[0]
        wlen_sw = t_axis[-1] - t_axis[0]
        rect = patches.Rectangle((x_axis[0], t_axis[0]), length_sw, wlen_sw, linewidth=1,
                                 edgecolor=c, facecolor='none')
        ax.add_patch(rect)
        #ON A ENLEVE UNE PARTIE ICI


    def save_fig(self, fig_name=None, fig_dir="Fig/show_sw_time_windows/", t_min=None, t_max=None, x_min=None, x_max=None, color='red'):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(self.veh_state_x, self.veh_state_t, '.', color=color, markersize=1)
        t_min = t_min if t_min is not None else self.t_axis[0]
        t_max = t_max if t_max is not None else self.t_axis[-1]

        x_min = x_min if x_min is not None else self.x_axis[0]
        x_max = x_max if x_max is not None else self.x_axis[-1]

        t_min_idx = np.abs(t_min - self.t_axis).argmin()
        t_max_idx = np.abs(t_max - self.t_axis).argmin()

        x_min_idx = np.abs(x_min - self.x_axis).argmin()
        x_max_idx = np.abs(x_max - self.x_axis).argmin()

        plot_data(self.data[x_min_idx: x_max_idx + 1, t_min_idx: t_max_idx + 1], x_axis=self.x_axis[x_min_idx: x_max_idx + 1],
                  t_axis=self.t_axis[t_min_idx: t_max_idx + 1],
                  fig_dir=fig_dir, fig_name=fig_name, ax=ax, pclip=90)
        


class SurfaceWaveSelector:
    def __init__(self, data_for_surface_wave,
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
                 spatial_ratio=0.5, #permet d'avoir un d�calage de la sw window par rapport au pivot.....
                 temporal_spacing=None,
                 channels=None #On avait ajouter cette modif pour pouvoir pr�ciser la valeur en offset et en num�ro de channel ??
                 ): #On pourra le supprimer si besoin
        """

        :param data_for_surface_wave:
        :param distances_along_fiber: x axis for the surface-wave data
        :param t_axis: t axis for the surface-wave data
        :param x0: middle distance for the window

        :param start_x_tracking: start_x for the tracking
        :param veh_states: tracked veh states
        :param distance_along_fiber_tracking: x axis for the tracking data
        :param t_axis_tracking: t axis for the tracking data
        :param wlen_sw: wlen for the surface wave window
        :param length_sw: spatial distance for the surface wave window
        """
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
        self.temporal_spacing = temporal_spacing if temporal_spacing else self.wlen_sw

        self.locate_windows()

    def locate_windows(self):
        win_nsamp = int(self.wlen_sw / self.dt)
        x0_idx = int((self.x0 - self.start_x_tracking)//self.dx_tracking)
        windows = []
        has_car_behind = []
        has_car_ahead = []
        for k, v in enumerate(self.veh_states):
            t0_idx = int(v[x0_idx])

            # reject cars behind it
            if k < len(self.veh_states) - 1:
                t0_next_v_idx = int(self.veh_states[k + 1, x0_idx])
                if np.abs(self.t_axis_tracking[t0_next_v_idx] - self.t_axis_tracking[t0_idx]) < self.temporal_spacing:
                    has_car_behind.append(k)
                    continue

            # reject cars ahead of it
            if k > 0:
                t0_before_v_idx = int(self.veh_states[k - 1, x0_idx])

                delta_t = self.t_axis_tracking[t0_idx] - self.t_axis_tracking[t0_before_v_idx]

                if self.temporal_spacing > delta_t > 0:
                    has_car_ahead.append(k)
                    print('b')
                    continue

            t0 = self.t_axis_tracking[t0_idx]
            t0_sw_idx = np.abs(t0 - self.t_axis).argmin()
            
            # reject windows at the boundaries
            if t0_sw_idx < win_nsamp // 2 or t0_sw_idx + win_nsamp // 2 > self.t_axis.size:
                print('c')
                continue

            start_x = self.x0 - self.length_sw * self.spatial_ratio #Est ce qu'on en a vraiment besoin??
            end_x = start_x + self.length_sw

            start_x_idx = np.abs(start_x - self.distances_along_fiber).argmin()
            end_x_idx = np.abs(end_x - self.distances_along_fiber).argmin()

            start_t0_idx = t0_sw_idx - win_nsamp // 2
            end_t0_idx = start_t0_idx + win_nsamp

            sw_window = SurfaceWaveWindow(
                data=copy.deepcopy(self.data_for_surface_wave[start_x_idx: end_x_idx, start_t0_idx: end_t0_idx]),
                t_axis=self.t_axis[start_t0_idx: end_t0_idx],
                x_axis=self.distances_along_fiber[start_x_idx: end_x_idx],
                veh_state=v,
                start_x_tracking=self.start_x_tracking,
                distance_along_fiber_tracking=self.distance_along_fiber_tracking,
                t_axis_tracking=self.t_axis_tracking,
            )

            windows.append(sw_window)
            
# =============================================================================
#             try:
#                 t0_idx_preprocess = np.abs(self.t_axis[start_t0_idx] - self.t_axis_tracking).argmin()
#                 tf_idx_preprocess = np.abs(self.t_axis[end_t0_idx] - self.t_axis_tracking).argmin()
#                 x0_idx_preprocess = np.abs(start_x - self.distance_along_fiber_tracking).argmin()
#                 xf_idx_preprocess = np.abs(end_x - self.distance_along_fiber_tracking).argmin()
#                 
#                 pclip = np.percentile(np.abs(self.data_for_tracking), 99)
#                 fig, ax = plt.subplots(figsize=(8,6))
#                 tomap = ax.pcolormesh(self.t_axis_tracking[t0_idx_preprocess: tf_idx_preprocess]%600,
#                                self.distance_along_fiber_tracking[x0_idx_preprocess: xf_idx_preprocess],
#                                self.data_for_tracking[x0_idx_preprocess: xf_idx_preprocess, t0_idx_preprocess: tf_idx_preprocess],
#                                cmap='seismic', vmin=-pclip, vmax=pclip)
#                 cb = fig.colorbar(tomap)
#                 cb.set_label(label="Strain Rate [n$\epsilon$.s$^{-1}$]", size='x-large')
#                 ax.set_xlabel('Time [s]', fontsize='x-large')
#                 ax.set_ylabel('Offset [m]', fontsize='x-large')
#                 plt.show()
#                 
#             except:
#                 print(self.distance_along_fiber_tracking[x0_idx_preprocess: xf_idx_preprocess].shape)
#                 print(self.t_axis_tracking[t0_idx_preprocess: tf_idx_preprocess].shape)
#                 print()
# =============================================================================

        self.windows = windows

    # return number of isolated cars
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, item):
        return self.windows[item]

    def __setitem__(self, key, value):
        self.windows[key] = value

    def __contains__(self, item):
        return item >= 0 and item < len(self.windows)

    def save_figs(self, muted=False, offset=450, fig_dir="Fig/show_sw_time_windows/"):
        for k, win in enumerate(self):
            fig_prefix = 'sw_car'
            if muted:
                win_to_save = copy.deepcopy(win)
                win_to_save.mute_along_traj(offset=offset, alpha=0.6)
                fig_prefix += '_muted'
            else:
                win_to_save = win
            win_to_save.save_fig(fig_name=f"{fig_prefix}{k}.png", fig_dir=fig_dir)