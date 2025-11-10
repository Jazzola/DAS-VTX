import numpy as np
from scipy.signal import find_peaks, resample_poly, savgol_filter, butter, sosfiltfilt
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from config import preprocessing_dict, preselection_dict
from config import sigma_a, R, nx_init, reverse_amp

from utils import likelihood_1d, interp_nan_value
from utils import fk_velocity_filter, k_manual_filt, diagonal_shift
from utils import max_total_nan_crit, max_adjacent_nan_crit, average_speed_crit, curve_break_crit, speed_fluctuations_crit
from func_data_imports import read_das_npz
from SW_class import SurfaceWaveSelector

from random import randint

import logging
logger = logging.getLogger('tests_trains')

class Tracker:
    def __init__(self, decimated_data_file, dx, **kwargs):
        self.dx = dx
        self.decimated_data_file = decimated_data_file

        self._preprocess_for_tracking()


    def _preprocess_for_tracking(self):
        
        dx = self.dx
        
        data_for_tracking, x_axis_tracking, t_axis_tracking = read_das_npz(self.decimated_data_file)
        self.t_axis_tracking = t_axis_tracking
        self.dt = t_axis_tracking[1]-t_axis_tracking[0]
        
        # Spatial interpolation
        param_x_inter = preprocessing_dict.get('x_inter', (240, 25))
        if param_x_inter:
            data_for_tracking = resample_poly(data_for_tracking,
                                                     param_x_inter[0],
                                                     param_x_inter[1])
            dist_along_fiber_tracking = np.arange(data_for_tracking.shape[0]) \
                *dx*param_x_inter[1]/param_x_inter[0] \
                + x_axis_tracking[0]

        else:
            dist_along_fiber_tracking = np.arange(data_for_tracking.shape[0])*dx + x_axis_tracking[0]
        

        
        # FK velocity filter
        param_FK = preprocessing_dict.get('FK', {})
        if param_FK:
            data_for_tracking, _, _, _ = fk_velocity_filter(data_for_tracking,
                                                            dist_along_fiber_tracking,
                                                            t_axis_tracking,
                                                            param_FK)
            
        # BP filter
        param_BP = preprocessing_dict.get('BP', {})
        if param_BP:
            freq_lo = param_BP.get('freq_lo', 0.2)
            freq_hi = param_BP.get('freq_hi', 1)
            sos_s = butter(5, [freq_lo, freq_hi], btype='bandpass', output='sos', fs=1/self.dt)
            data_for_tracking = sosfiltfilt(sos_s, data_for_tracking, axis=1)
        
                
        # Smoothing
        param_smoothing = preprocessing_dict.get('smoothing', (21, 15))
        if param_smoothing:
            data_for_tracking = savgol_filter(data_for_tracking,
                                                        param_smoothing[0],
                                                        param_smoothing[1])
            
        
        

        spatial_av_vel = preprocessing_dict.get('spatial_av_vel', False)
        if spatial_av_vel:
            av_win = preprocessing_dict.get('av_win', 5)
            
            A = diagonal_shift(data_for_tracking,
                                            -spatial_av_vel, self.dx, self.dt)
            for it in range(A.shape[1]):
                A[:, it] = np.convolve(A[:, it], np.ones(av_win), mode='same') #10
            A = diagonal_shift(A,
                                            spatial_av_vel, self.dx, self.dt)
            
            B = diagonal_shift(data_for_tracking,
                                            -spatial_av_vel*1.1, self.dx, self.dt)
            for it in range(B.shape[1]):
                B[:, it] = np.convolve(B[:, it], np.ones(av_win), mode='same')
            B = diagonal_shift(B,
                                            spatial_av_vel*1.1, self.dx, self.dt)
            
            C = diagonal_shift(data_for_tracking,
                                            -spatial_av_vel*0.9, self.dx, self.dt)
            for it in range(C.shape[1]):
                C[:, it] = np.convolve(C[:, it], np.ones(av_win), mode='same')
            C = diagonal_shift(C,
                                            spatial_av_vel*0.9, self.dx, self.dt)
            
            data_for_tracking = (A+B+C)/3
        

        #SQRT weighting
        if preprocessing_dict.get('SQRT', False):
            for j in range(data_for_tracking.shape[1]):
                data_for_tracking[:,j] *= np.sqrt(np.abs(data_for_tracking[:,j])/np.max(np.abs(data_for_tracking[:,j])))
            for i in range(data_for_tracking.shape[0]):
                data_for_tracking[i,:] *= np.sqrt(np.abs(data_for_tracking[i,:])/np.max(np.abs(data_for_tracking[i,:])))
        
        mask = np.abs(data_for_tracking)<np.percentile(data_for_tracking, 98)
        reduced_array = np.square(data_for_tracking[mask]/np.percentile(data_for_tracking, 98))
        data_for_tracking[mask] *= np.sign(data_for_tracking[mask])*reduced_array
        #data_for_tracking = np.sign(data_for_tracking)*np.square(data_for_tracking/np.percentile(data_for_tracking, 100))
        #data_for_tracking[np.abs(data_for_tracking)<np.percentile(data_for_tracking, 90)] = 0
        

        oversampling_factor = preprocessing_dict.get('oversampling_factor', 5)
        if oversampling_factor > 1:
            data_for_tracking = diagonal_shift(data_for_tracking,
                                            -spatial_av_vel/3.6, self.dx, self.dt)
            interpolator = RegularGridInterpolator((dist_along_fiber_tracking, t_axis_tracking),
                                                data_for_tracking)
            xx, yy = np.meshgrid(np.linspace(dist_along_fiber_tracking[0],
                                            dist_along_fiber_tracking[-1],
                                            oversampling_factor*len(dist_along_fiber_tracking)),
                                t_axis_tracking, indexing='ij')
            data_for_tracking = interpolator((xx, yy))
            dist_along_fiber_tracking = np.linspace(dist_along_fiber_tracking[0],
                                            dist_along_fiber_tracking[-1],
                                            oversampling_factor*len(dist_along_fiber_tracking))
            data_for_tracking = diagonal_shift(data_for_tracking,
                                            spatial_av_vel/3.6, self.dx/oversampling_factor, self.dt)
        

        self.data_for_tracking = data_for_tracking
        self.dist_along_fiber_tracking = dist_along_fiber_tracking
        self.dx = dist_along_fiber_tracking[1]-dist_along_fiber_tracking[0]
        
        

    def _detect_in_one_section(self, start_x, nx, sigma, plt_xlim, detection_args=None, pclip=98, show_plot=False):
        t_axis=self.t_axis_tracking
        x_axis=self.dist_along_fiber_tracking
        args=self.tracking_args
        
        if not detection_args:
            detection_args = args["detect"]

        minprominence = detection_args["minprominence"]
        minseparation = detection_args["minseparation"]
        prominenceWindow = detection_args["prominenceWindow"]

        peak_erode = np.zeros(len(t_axis))

        start_x_idx = np.argmin(np.abs(start_x - x_axis))
        for i in range(nx):
            das_for_dt = self.data_for_tracking[start_x_idx+i]
            peak_loc = find_peaks(das_for_dt,
                                  prominence=minprominence,
                                  wlen=prominenceWindow,
                                  distance=minseparation)[0]
            peak_erode_tmp = likelihood_1d(peak_loc, t_axis, sigma)
            peak_erode += peak_erode_tmp

        peak_loc_tmp, _ = find_peaks(peak_erode, height=max(peak_erode) * 0., distance=minseparation)
        vmax = np.percentile(np.abs(self.data_for_tracking), pclip)

        if show_plot: #adapter au cas ou
            fig, axes = plt.subplots(1, 2, figsize=(15, 10), gridspec_kw={'width_ratios':[3,1]}, sharey=True)
            axes[0].imshow(self.data_for_tracking.T,
                           aspect="auto",
                           extent=[self.dist_along_fiber_tracking[0], 
                                   self.dist_along_fiber_tracking[-1],
                                   self.t_axis_tracking[-1],
                                   self.t_axis_tracking[0]],
                           cmap="gray") #, vmax=vmax, vmin=-vmax)
            axes[0].axvline(x=self.dist_along_fiber_tracking[start_x_idx], c='r')
            axes[0].axvline(x=self.dist_along_fiber_tracking[start_x_idx+nx], c='b')
            axes[1].plot(peak_erode, self.t_axis_tracking, 'b')
            axes[1].plot(peak_erode[peak_loc_tmp], self.t_axis_tracking[peak_loc_tmp], 'r^')
            axes[0].plot([self.dist_along_fiber_tracking[start_x_idx+nx//2]] * len(peak_loc_tmp),
                         self.t_axis_tracking[peak_loc_tmp], 'r^')
            #axes[0].set_xlim([self.x_axis[0], plt_xlim])

        veh_base = peak_loc_tmp
        return veh_base
    
    
    def _remove_unrealistic_tracking(self, veh_base, veh_states, dt, dx):
        invalid_num_tmp = []
        
        test = [0, 0, 0, 0, 0]

        for v in range(len(veh_base)):
            veh_state = veh_states[v]
            
            invalidity = False
            
            value = preselection_dict.get('max_adjacent_nan', False)
            if value:
                invalidity += max_adjacent_nan_crit(veh_state, value)
                test[0]+=max_adjacent_nan_crit(veh_state, value)
            
            portion = preselection_dict.get('max_total_nan', False)
            if portion:
                invalidity += max_total_nan_crit(veh_state, portion)
                test[1]+=max_total_nan_crit(veh_state, portion)
                
            if invalidity:
                invalid_num_tmp.append(v)
                
        valid_num_tmp = list(range(len(veh_base)))
        for v in invalid_num_tmp:
            valid_num_tmp.remove(v)
        veh_states = veh_states[valid_num_tmp, :]
            
            

        
        invalid_num_tmp = []
        for v in range(veh_states.shape[0]):
            veh_state = veh_states[v]
            invalidity = False
                    
            try:
                speed_values = preselection_dict.get('average_speed', False)
                if speed_values:
                    invalidity += average_speed_crit(veh_state, dx, dt, speed_values)
                    test[2]+=average_speed_crit(veh_state, dx, dt, speed_values)
                    
                win_size, factor, portion, q = preselection_dict.get('curve_break', (False, False, False, False))
                if win_size:
                    invalidity += curve_break_crit(veh_state, win_size, factor, portion, q)
                    test[3]+=curve_break_crit(veh_state, win_size, factor, portion, q)
                    
                factor, portion = preselection_dict.get('speed_fluctuations', (False, False))
                if factor:
                    invalidity += speed_fluctuations_crit(veh_state, dx, dt, factor, portion)
                    test[4]+=speed_fluctuations_crit(veh_state, dx, dt, factor, portion)
                    
                if invalidity:
                    invalid_num_tmp.append(v)
                
            except:
                invalid_num_tmp.append(v)
                continue


        valid_num_tmp = list(range(veh_states.shape[0]))
        for v in invalid_num_tmp:
            valid_num_tmp.remove(v)
        tracked_v = veh_states[valid_num_tmp, :]
        #print('PRESELECT TEST', test)
        #print('output selc shape', tracked_v.shape)
        test_str = [str(t) for t in test]
        test_str = ', '.join(test_str)
        logger.info('PRESELECT TEST: '+test_str)
        
        return tracked_v
    
        
    def _tracking_with_veh_base(self, start_x, end_x, veh_base, sigma_a=0.01, R=1, detection_args=None):
        x_axis=self.dist_along_fiber_tracking
        args=self.tracking_args
        
        if detection_args is None:
            detection_args = args['detect']

        start_x_idx = np.argmin(np.abs(start_x - x_axis))
        end_x_idx = np.argmin(np.abs(end_x - x_axis))

        x_axis = x_axis[start_x_idx: end_x_idx + 1]

        minprominence = detection_args["minprominence"]
        minseparation = detection_args["minseparation"]
        prominenceWindow = detection_args["prominenceWindow"]
        height = detection_args.get("height", None)

        veh_states = np.empty((len(veh_base), end_x_idx - start_x_idx + 1))
        veh_states[:] = np.nan


        Tk1k = np.empty((2, len(veh_base)))
        Tk1k[:] = np.nan
        Tkk = np.empty((2, len(veh_base)))
        Tkk[:] = np.nan
        Pkk = np.empty((2, 2, len(veh_base)))
        Pkk[:] = np.nan
        Pk1k = np.empty((2, 2, len(veh_base)))
        Pk1k[:] = np.nan
        
        Xv = np.empty(len(veh_base))
        Xv[:] = np.nan
        
        C = np.array([1, 0])
        veh_base_state = veh_base.copy()

        for i in range(start_x_idx, end_x_idx + 1):
            for v in range(len(veh_base)):

                if sum(~np.isnan(veh_states[v, :])) == 0:
                    veh_base_state[v] = veh_base[v]
                    
                elif sum(~np.isnan(veh_states[v, :])) == 1:
                    Tkk[:, v] = list(veh_states[v, ~np.isnan(veh_states[v, :])]) + [0] #ATTENTION, vitesse init, si !=0 il faut define un param, 50/3.6
                    Pkk[:, :, v] = np.array([[0, 0], [0, 0]])
                    
                    Xv[v] = x_axis[0] #x_axis[~np.isnan(veh_states[v, :])]
                    veh_base_state[v] = veh_base[v]
                    
                else:
                    delta_x = self.dist_along_fiber_tracking[i] - Xv[v]
                    A = [[1, delta_x], [0, 1]]
                    Q = sigma_a * np.array([[0.25 * delta_x ** 4, 0.5 * delta_x ** 3], [0.5 * delta_x ** 3, delta_x ** 2]])
                    Tk1k[:, v] = np.matmul(A, Tkk[:, v])

                    Pk1k[:, :, v] = np.matmul(np.matmul(A, Pkk[:, :, v]), np.transpose(A)) + Q
                    veh_base_state[v] = Tk1k[0, v]

            das_for_dt = self.data_for_tracking[i]
            peak_loc, _ = find_peaks(das_for_dt, prominence=minprominence, wlen=prominenceWindow, distance=minseparation) #height=height) #minprominence/2

            for p in range(len(veh_base_state)):
                dist_tmp = peak_loc - veh_base_state[p]
                idx_tmp = np.where((dist_tmp > -100) & (dist_tmp <= 100))[0] #definir un param ?
                valid_tmp = np.abs(dist_tmp[idx_tmp])

                if len(valid_tmp) == 0:
                    min_idx = []
                else:
                    min_idx = np.where(valid_tmp == valid_tmp.min())[0]
                    min_idx= min_idx[:1]

                if len(min_idx) > 0:
                    veh_states[p, i - start_x_idx] = peak_loc[idx_tmp[min_idx]]
                else:
                    veh_states[p, i - start_x_idx] = np.nan

            # filtering
            for v in range(len(veh_base)):
                if (sum(~np.isnan(veh_states[v, :])) > 2) and (not np.isnan(veh_states[v, i - start_x_idx])):
                    K = Pk1k[:, :, v] @ C.T / (R + C @ Pk1k[:, :, v] @ C.T)
                    tkk = Tk1k[:, v] + K * (veh_states[v, i - start_x_idx] - C @ Tk1k[:, v])
                    Tkk[:, v] = tkk
                    Pkk[:, :, v] = Pk1k[:, :, v] - (K.reshape(2, 1) @ C.reshape(1, 2)) @ Pk1k[:, :, v]
                    Xv[v] = self.dist_along_fiber_tracking[i]
                
                

        #print('shape veh states before selection', veh_states.shape)
        logger.info(f'Shape veh states before selection: {veh_states.shape}')
        tracked_v = self._remove_unrealistic_tracking(veh_base, veh_states, self.dt, self.dx)
            
        tracked_v = interp_nan_value(tracked_v)
        return tracked_v

    def track_cars(self, start_x, end_x, tracking_args,
                   show_plot=True, plt_xlim=1000,
                   reverse_amp=reverse_amp, sigma_a=sigma_a, R=R, nx=nx_init):
        
        self.start_x = start_x
        self.end_x = end_x
        self.tracking_args = tracking_args
        
        if reverse_amp:
            self.data_for_tracking = -self.data_for_tracking

        veh_base = self._detect_in_one_section(start_x=self.start_x,
                                               nx=nx, sigma=sigma_a, show_plot=show_plot,
                                               plt_xlim=plt_xlim)
        
        self.veh_states = self._tracking_with_veh_base(start_x=self.start_x,
                                                       end_x=self.end_x, veh_base=veh_base,
                                                       sigma_a=sigma_a, R=R)


    def select_surface_wave_windows(self, data_raw, x_axis, t_axis, x0, **kwargs):
        self.sw_selector = SurfaceWaveSelector(
            data_raw,
            x_axis,
            t_axis,
            x0,
            self.data_for_tracking,
            self.start_x,
            self.veh_states,
            self.dist_along_fiber_tracking,
            self.t_axis_tracking,
            **kwargs)



