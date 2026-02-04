import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing  import Pool
from scipy.stats import norm
from scipy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift
from scipy.signal.windows import tukey
from scipy.signal import welch
from datetime import datetime, timedelta
from obspy import Stream, Trace
from obspy.core import UTCDateTime
from skimage.filters import window as taper_win

from config import DATE_TIME_FORMAT, DAS_FILE_FORMAT, DATE_FORMAT
from func_data_imports import data_matrix_import



#=========================#
### PREPROCESS TRACKING ###
#=========================#

def fk_velocity_filter(data_for_tracking, dist_along_fiber_tracking,
                       t_axis_tracking,
                       param_dict):
    
    dt = t_axis_tracking[1]-t_axis_tracking[0]
    fk = fft2(data_for_tracking)
    f_axis = fftfreq(fk.shape[1], dt)
    k_axis = fftfreq(fk.shape[0], dist_along_fiber_tracking[1] - dist_along_fiber_tracking[0])
    f_axis = fftshift(f_axis)
    k_axis = fftshift(k_axis)
    fk = fftshift(fk)
    fk_filt = fk.copy()
    
    slopehi = param_dict.get('slope_hi', 3.6/20)
    slopelo = param_dict.get('slope_lo', 3.6/60)
    
    for p in np.arange(len(f_axis)):
        f = f_axis[p]
        limhi = f*slopehi
        limlo = f*slopelo
        kk1 = np.where((k_axis<=limhi)&(k_axis>=limlo))[0]
        apo_win1 = tukey(kk1.shape[0], alpha=0.3)
        kk2 = np.where((k_axis>=limhi)&(k_axis<=limlo))[0]
        apo_win2 = tukey(kk2.shape[0], alpha=0.3)
        fk_filt[kk1, p] = fk_filt[kk1, p]
        fk_filt[kk2, p] = fk_filt[kk2, p]
        
        kk3 = np.where((k_axis<=-limhi)&(k_axis>=-limlo))[0]
        apo_win3 = tukey(kk3.shape[0], alpha=0.3)
        kk4 = np.where((k_axis>=-limhi)&(k_axis<=-limlo))[0]
        apo_win4 = tukey(kk4.shape[0], alpha=0.3)
        fk_filt[kk3, p] = fk_filt[kk3, p]
        fk_filt[kk4, p] = fk_filt[kk4, p]
        kk = np.setdiff1d(np.arange(0, len(k_axis), 1), np.concatenate((kk1,kk2,kk3,kk4)))
        
        fk_filt[kk, p] = 0
        
    for q in np.arange(len(k_axis)):
        k = k_axis[q]
        limhi = k/slopehi
        limlo = k/slopelo
        ff1 = np.where((f_axis<=limhi)&(f_axis>=limlo))[0]
        apo_win1 = tukey(ff1.shape[0], alpha=0.3)
        ff2 = np.where((f_axis>=limhi)&(f_axis<=limlo))[0]
        apo_win2 = tukey(ff2.shape[0], alpha=0.3)
        fk_filt[q, ff1] = fk_filt[q, ff1]
        fk_filt[q, ff2] = fk_filt[q, ff2]
        
        ff3 = np.where((f_axis<=-limhi)&(f_axis>=-limlo))[0]
        apo_win3 = tukey(ff3.shape[0], alpha=0.3)
        ff4 = np.where((f_axis>=-limhi)&(f_axis<=-limlo))[0]
        apo_win4 = tukey(ff4.shape[0], alpha=0.3)
        fk_filt[q, ff3] = fk_filt[q, ff3]
    
    fk_filt[int(len(k_axis)/2)-2 : int(len(k_axis)/2)+2, :] = 0 #Manual Fading removal
    f_axis = ifftshift(f_axis)
    k_axis = ifftshift(k_axis)
    fk_filt = ifftshift(fk_filt)
    data_for_tracking = ifft2(fk_filt).real
    return data_for_tracking, fk_filt, f_axis, k_axis



#==============#
### TRACKING ###
#==============#

def likelihood_1d(peak_loc, das_time_ds, sigma):
    data_tmp_thrd = np.zeros(len(das_time_ds))
    for j in range(len(peak_loc)):
        data_tmp_thrd = data_tmp_thrd + norm.pdf(das_time_ds, loc=das_time_ds[peak_loc[j]], scale=sigma)

    return data_tmp_thrd

def interp_nan_value(veh_states):
    for k, state in enumerate(veh_states):
        # Find indices of non-NaN values
        non_nan_indices = np.where(~np.isnan(state))[0]
        # Replace NaN values with linearly interpolated values
        state[np.isnan(state)] = np.interp(np.isnan(state).nonzero()[0], non_nan_indices, state[non_nan_indices])
        veh_states[k] = state
        
    return veh_states


        
def max_total_nan_crit(veh_state, portion):
    no_nan_traj = veh_state[~np.isnan(veh_state)]
    return len(no_nan_traj) < (1-portion) * len(veh_state)

def max_adjacent_nan_crit(veh_state, value):
    nan_indices = np.where(np.isnan(veh_state))[0]
    diffs = np.diff(nan_indices)
    adjacency_count = np.sum(diffs == 1)
    return adjacency_count >= value

def average_speed_crit(veh_state, dx, dt, speed_values):
    no_nan_traj = veh_state[~np.isnan(veh_state)]
    average_speed = dx*len(veh_state)/(np.abs(no_nan_traj[0]-no_nan_traj[-1])*dt)
    return average_speed < speed_values[0] or average_speed > speed_values[1]

def curve_break_crit(veh_state, win_size, factor, portion, q):
    diff = np.diff(veh_state)
    no_nan_diff = diff[~np.isnan(diff)]
    sliding = np.convolve(np.abs(no_nan_diff), np.ones(win_size), mode='valid') #sliding average to catch the curve breaks of a certain size (in number of points)
    return np.sum(sliding >= factor*np.percentile(sliding, q)) > portion*len(sliding) #number of points of the sliding average further than the quantile
    
def speed_fluctuations_crit(veh_state, dx, dt, factor, portion):
    time_diff = np.diff(veh_state)*dt
    instant_speed = dx/time_diff
    instant_speed = instant_speed[~np.isnan(instant_speed)]
    average_speed = np.mean(instant_speed)
    return np.sum(np.abs(instant_speed)>= factor*average_speed) > portion*len(instant_speed)



#==================#
### Surface Wave ###
#==================#

def plot_data(data, x_axis, t_axis, pclip=98, ax=None, figsize=(10, 10), y_lim=None, x_lim=None, fig_name=None, fig_dir="Fig/", fontsize=16, tickfont=12):
    vmax = np.percentile(np.abs(data), pclip)
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(data.T,
              aspect="auto",
              extent=[x_axis[0], x_axis[-1], t_axis[-1], t_axis[0]],
              cmap="gray",
              vmax=vmax,
              vmin=-vmax)
    ax.set_ylim(y_lim)
    ax.set_xlim(x_lim)
    plt.xlabel("Distance along the fiber [m]", fontsize=fontsize)
    ax.set_ylabel("Time [s]", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tickfont)
    if fig_name:
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path)
        
    

#
### XCORR ###
#

def repeat1d(trace):
    return np.hstack((trace, trace[:-1]))


def whiten_signals(X, fmin, fmax, fs=1.0, alpha=0.2):
    # Ensure the input is a numpy array
    X = np.array(X)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    
    n_signals, n_samples = X.shape
    X_whitened = np.zeros_like(X)
    
    for i in range(n_signals):
        # Calculate the FFT of the signal
        X_fft = np.fft.fft(X[i])
        freqs = np.fft.fftfreq(n_samples, 1/fs)
        
        X_fft /= np.abs(X_fft)
        
        
        #tapering the spectre before ifft and cliping to freq of interest
        ifmin = np.argmin(np.abs(freqs-fmin))
        ifmax = np.argmin(np.abs(freqs-fmax))
        nidx_taper = int((ifmax+1-ifmin)*alpha/2)
        
        #ATTENTION
        #On a un probleme de slicing
        #Si on choisi une freq dans le taper on a une erreur
        
        # X_fft[:ifmin -nidx_taper] = 0
        # X_fft[ifmax +1 +nidx_taper:] = 0
        # X_fft[ifmin -nidx_taper: ifmax +1 +nidx_taper] *= tukey(ifmax +1 -ifmin +2*nidx_taper, alpha=alpha)
        
        #ON VA ESSAYER EN INCLUANT LE TAPER DANS LA RANGE CHOISIE
        X_fft[:ifmin] = 0
        X_fft[ifmax + 1:] = 0
        X_fft[ifmin: ifmax +1] *= tukey(ifmax +1 -ifmin, alpha=alpha)
        
        X_fft_whitened = X_fft
        
        # Convert back to time domain
        X_whitened[i] = np.real(np.fft.ifft(X_fft_whitened))
    
    return X_whitened


def k_manual_filt(x_axis, t_axis, data, npts=2):
    dt = t_axis[1]-t_axis[0]
    fk = fft2(data)
    f_axis = fftfreq(fk.shape[1], dt)
    k_axis = fftfreq(fk.shape[0], x_axis[1] - x_axis[0])
    f_axis = fftshift(f_axis)
    k_axis = fftshift(k_axis)
    fk = fftshift(fk)
    fk_filt = fk.copy()
    taper = np.ones(2*npts+len(k_axis)%2) - tukey(2*npts+len(k_axis)%2, alpha=1)
    fk_filt[int(len(k_axis)/2)-npts : int(len(k_axis)/2)+npts+len(k_axis)%2, :] *= np.tile(taper, len(f_axis)).reshape((2*npts+len(k_axis)%2, len(f_axis)))
    f_axis = ifftshift(f_axis)
    k_axis = ifftshift(k_axis)
    fk_filt = ifftshift(fk_filt)
    return ifft2(fk_filt).real, np.real(fk), f_axis, k_axis


def calculate_SNR(VSG):
    # Create meshgrid of time (yy) and space (xx) coordinates
    yy, xx = np.meshgrid(VSG.t_axis, VSG.x_axis)
    
    # Prevent division by zero by replacing zero values in yy
    yy_safe = np.where(yy == 0, 1e-10, yy)  # Small value instead of zero
    ratio = np.abs(xx/yy_safe)
    
    # noise_mask = (  ((ratio > 1200) | (ratio < 250)) &
    #     (xx > 0.1) &
    #     (yy > 0.1) &
    #     (yy <= 1))
    
    noise_mask = ((ratio > 1200) &
        (xx > 0.1) & (yy > 0.1) & (yy <= 1))
    
    signal_mask = ((ratio >= 250) & (ratio <= 1200) &
        (xx > 0.1) & (yy > 0.1) & (yy <= 1.0))
    
    signal = np.sqrt(np.mean(VSG.XCF_out[signal_mask] ** 2))
    noise  = np.sqrt(np.mean(VSG.XCF_out[noise_mask] ** 2))
    
    # Avoid division errors
    if noise == 0:
        return np.inf
    
    # Compute SNR in dB
    SNR = 20*np.log10(signal / noise)
    
    return SNR


#================#
### INTERPRETATION 
#================#

class Coherence_Enhancement:
    
    def __init__(self, stack, interpolator, xwin, twin, slw_list, decimation_factor, nprocesses):
        self.stack = stack
        self.interpolator = interpolator
        self.xwin = xwin
        self.twin = twin
        self.slw_list = slw_list
        self.decimation_factor = decimation_factor
        self.nprocesses = nprocesses

    def semb(self, slw):
        sem = np.empty((self.stack.XCF_out.shape[0], self.stack.XCF_out.shape[1]//self.decimation_factor))
        causal_t = self.stack.t_axis[self.stack.t_axis>=0]
        acausal_t = self.stack.t_axis[self.stack.t_axis<0]
        for i in range(self.stack.XCF_out.shape[0]):
            Gtosum = 0
            GHtosum = 0
            for x in range(max(0, i-self.xwin), min(i+self.xwin+1, self.stack.XCF_out.shape[0])):
                t_interp_caus = causal_t + slw*(np.abs(self.stack.x_axis[x]) - np.abs(self.stack.x_axis[i]))
                t_interp_acaus = acausal_t - slw*(np.abs(self.stack.x_axis[x]) - np.abs(self.stack.x_axis[i]))
                t_interpolation_axis = np.hstack((t_interp_acaus, t_interp_caus))
                og, tg = np.meshgrid(self.stack.x_axis[x], t_interpolation_axis, indexing='ij')
                interpolated_data = self.interpolator((og,tg))[0,:]
                Gtosum += interpolated_data
                GHtosum += interpolated_data**2

            numerateur = Gtosum**2
            denominateur = GHtosum

            for semj in range(0, self.stack.XCF_out.shape[1]//self.decimation_factor):
                j=semj*self.decimation_factor
                num_win = numerateur[max(0, j-self.twin):min(j+self.twin+1, self.stack.XCF_out.shape[1])]
                num_win = num_win[~np.isnan(num_win)]
                num = np.real(np.sum(num_win))
                deno_win = denominateur[max(0, j-self.twin):min(j+self.twin+1, self.stack.XCF_out.shape[1])]
                deno_win = deno_win[~np.isnan(deno_win)]
                deno = np.real(np.sum(deno_win))
                sem[i, semj] = num/deno
        return np.abs(sem)/(2*self.xwin+1)
    
    def calculate_enhanced_stack(self):
        with Pool(self.nprocesses) as p:
            results = p.map(self.semb, self.slw_list)
        return results




#
### FORMAT UTILS ###
#

def generate_date_range(start_date: str, end_date: str) -> list:
    """Generate a list of dates between start_date and end_date inclusive."""
    start = datetime.strptime(start_date, DATE_FORMAT)
    end = datetime.strptime(end_date, DATE_FORMAT)
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime(DATE_FORMAT))
        current += timedelta(days=1)
    return date_list


def get_date_from_file_path(file_path, format):
    date = datetime.strptime(file_path.name, format)
    return date

def get_file_section(file_path): #FONCTIONNE POUR NOTRE FORMAT SPECIFIQUE, il faut adapter!!
    name = file_path.name
    chunk = name.split('offset')[1]
    chunk = chunk.split('_')[0]
    start_ch, end_ch = chunk.split('to')
    return int(start_ch), int(end_ch)



#
### DATA UTILS
#

#DATE format et nombre de minutes du chunk à mettre en variables
def create_npz_data(filepath, destination_path, decimation_factor=10, chunk_size=600, return_full=False):
    data, distance, t_axis, data_attributes = data_matrix_import(filepath)
    
    dt=t_axis[1]-t_axis[0]
    st_data, _ = from_npArray2streamV2(data, 1, data_attributes[-1], 1/dt, distance)
    del data_attributes
    
    st_data.decimate(decimation_factor)
    t_axis_decimated = t_axis[::decimation_factor]
    
    data_decimated, _ = from_stream2npArray(st_data)
    del st_data
    
    chunk_npts = int(chunk_size/(dt*decimation_factor))
    for k in range(t_axis_decimated.shape[0]//chunk_npts):
        data_portion = data_decimated[:, k*chunk_npts:(k+1)*chunk_npts]

        start = get_date_from_file_path(filepath, DAS_FILE_FORMAT)
        start += timedelta(minutes=k*chunk_size//60)
        destination = destination_path / start.strftime(DATE_TIME_FORMAT)
        
        np.savez(destination, data=data_portion, x_axis=distance, t_axis=t_axis_decimated[k*chunk_npts:(k+1)*chunk_npts])
    
    if return_full:
        return data, distance, t_axis
    del data, distance, t_axis
    return


def from_stream2npArray(stQ):
    compteur = 0
    np2Darray = []
    Qdistance = []
    for Qtr in stQ:
        if compteur == 0:
            np2Darray = Qtr.data
            Qdistance = Qtr.stats.distance
            compteur += 1
        else:
            np2Darray = np.vstack((np2Darray, Qtr.data))
            Qdistance = np.hstack((Qdistance, Qtr.stats.distance))
    del stQ, compteur
    return [np2Darray, Qdistance]


def from_npArray2streamV2(SR_data, Q_stepping_index, Qtimming, Qsampling, distance):
    st = Stream()
    
    if len(distance) != np.shape(SR_data)[0]:
        print('Missmatching distance and strain rate datasets')
        sys.exit()
        
    if len(distance)>1:
        for ii in np.arange(0, np.shape(SR_data)[0], Q_stepping_index):
            tr = Trace()
            tr.data = SR_data[ii, :]
            tr.stats.starttime = UTCDateTime(Qtimming)
            tr.stats.distance = distance[ii]
            tr.stats.location = '00'
            tr.stats.network = 'KB'
            if distance[ii]<10:
                tr.stats.station = '000'+str(int(distance[ii]))
            elif distance[ii]<100:
                tr.stats.station = '00'+str(int(distance[ii]))
            elif distance[ii]<1000:
                tr.stats.station = '0'+str(int(distance[ii]))
            else:
                tr.stats.station = str(int(distance[ii]))
            tr.stats.channel = 'SR'
            tr.stats.sampling_rate = Qsampling
            st += tr
        distance2 = distance[np.arange(0, len(distance), Q_stepping_index)]
        
    else:
        tr = Trace()
        tr.data = SR_data[0, :]
        tr.stats.starttime = UTCDateTime(Qtimming)#-0.5
        tr.stats.distance = distance[0]
        tr.stats.location = '00'
        tr.stats.network = 'KB'
        if distance[0]<10:
            tr.stats.station = '000'+str(int(distance[0]))
        elif distance[0]<100:
            tr.stats.station = '00'+str(int(distance[0]))
        elif distance[0]<1000:
            tr.stats.station = '0'+str(int(distance[0]))
        else:
            tr.stats.station = str(int(distance[0]))
        tr.stats.channel = 'SR'
        tr.stats.sampling_rate = Qsampling
        st += tr
        distance2 = distance
    
    return [st, distance2]
    


#
### OTHERS
#


def multiprocess_iterable_on_dates(start_date, end_date, n_processes, section):
    date_list = generate_date_range(start_date, end_date)
    n_days = len(date_list)//n_processes
    remind = len(date_list)%n_processes
    if n_days > 1:
        multi_args = []
        for p in range(remind):
            multi_args.append([section, date_list[(n_days+1)*p], date_list[(n_days+1)*(p+1)-1]])
        for p in range(remind, n_processes):
            multi_args.append([section, date_list[n_days*p+remind], date_list[n_days*(p+1)+remind-1]])

    else:
        multi_args = [[section, day, day] for day in date_list]
        
    return multi_args


def multiprocess_iterable_on_sections(start_date, end_date, section_list):
    multi_args = [[section, start_date, end_date] for section in section_list]
    return multi_args



def diagonal_shift(arr, shift_vel, dx, dt):
    shift_factor = dx/(shift_vel*dt)
    
    shifted_arr = arr.copy()
    rows, cols = shifted_arr.shape
    # Perform diagonal shift
    for i in range(rows):
        # Calculate shift amount based on row and shift factor
        shift = int(i * shift_factor)
        shifted_arr[i] = np.roll(shifted_arr[i], shift)
    
    return shifted_arr #IL FAUT POUVOIR PRENDRE EN COMPTE LES VITESSES NEG!!