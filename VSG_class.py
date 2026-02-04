import copy
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import welch, correlate

from SW_class import SurfaceWaveWindow
from Disp_class import Dispersion
from utils import repeat1d



def preprocessing_window_NEW(window, pivot, delta_t, time_window_to_xcorr):
    f = interpolate.interp1d(window.veh_state_x, window.veh_state_t, fill_value='extrapolate')

    dt = window.t_axis[1] - window.t_axis[0]
    pivot_t = f(pivot) + delta_t
    pivot_t_idx = np.argmax(window.t_axis >= pivot_t)


    nsamp = int(time_window_to_xcorr // dt)

    data = window.data / np.linalg.norm(window.data)
    
    return pivot_t_idx, nsamp, data, dt, f


def XCORR_vshot_NEW(data, ivs, wlen, dt, overlap_ratio=0.5, reverse=False):
    nch, nt = data.shape[0], data.shape[1]

    wlen = int(wlen / dt)

    wlen_offset = int(wlen * (1 - overlap_ratio))
    nwin = (nt - wlen) // wlen_offset + 1
    XCF_out = np.zeros((nch-1, wlen))
    for iwin in range(nwin):
        data_vs = repeat1d(data[ivs, (iwin * wlen_offset):(iwin * wlen_offset) + wlen])
        curt_XCF = []
        if reverse:
            for ivr in range(1, nch):
                vs = data[ivr, (iwin * wlen_offset):(iwin * wlen_offset) + wlen]
                vr = data_vs
                curt_XCF.append(correlate(vs, vr, mode='valid', method='fft'))
        else:
            for ivr in range(nch-1):
                vs = data_vs
                vr = data[ivr, (iwin * wlen_offset):(iwin * wlen_offset) + wlen]
                curt_XCF.append(correlate(vs, vr, mode='valid', method='fft'))

        XCF_out += np.asarray(curt_XCF)
    if nwin == 0:
        #return np.zeros((nch-1, wlen))
        raise ValueError('nwin=0')
    else:
        return np.roll(XCF_out, wlen // 2, axis=-1) / nwin


def construct_shot_gather_NEW(window: SurfaceWaveWindow, start_x, end_x, pivot,
                          wlen=2, norm=True, norm_amp=True,
                          time_window_to_xcorr = 4, delta_t=1, overlap=0.5):

    idx0 = np.abs(window.x_axis - window.veh_state_x[-1]).argmin()
    start_x_idx = np.abs(window.x_axis - start_x).argmin()
    end_x_idx = np.abs(window.x_axis - end_x).argmin()
    idx0 = min(idx0, end_x_idx) #the end of xcorr section can be taken before the max trajectory offset
    nch = idx0 - start_x_idx
    
    dt = window.t_axis[1] - window.t_axis[0]
    XCF_out = np.zeros((nch, int(wlen/dt)))
    rel_dist_fact = np.arange(1, nch+1)
    for pivot_idx in range(start_x_idx+1, idx0+1):
        pivot = window.x_axis[pivot_idx]
        pivot_t_idx, nsamp, data, dt, f = preprocessing_window_NEW(window, pivot, delta_t, time_window_to_xcorr)

        # xcorr with the channels to the left of the pivot
        XCF_out[: pivot_idx - start_x_idx, :] += XCORR_vshot_NEW(data[start_x_idx: pivot_idx+1, pivot_t_idx: pivot_t_idx+nsamp],
                              ivs=pivot_idx-start_x_idx, wlen=wlen, dt=dt, overlap_ratio=overlap)[::-1, :] #car l'odre des correls n'est pas le bon dans ce cas
        #ATTENTION, on change XCORR_vshot car on ne veut pas l'autocorr (on prend nch-1 plutot que nch)
        
    for i in range(XCF_out.shape[0]):
        XCF_out[i, :] /= rel_dist_fact[::-1][i]
    return post_processing_XCF_NEW(window, idx0, start_x_idx, end_x_idx, XCF_out, dt, norm, norm_amp, reverse=False)


def construct_shot_gather_otherside_NEW(window: SurfaceWaveWindow, start_x, end_x, pivot,
                          wlen=2, norm=True, norm_amp=True,
                          time_window_to_xcorr = 4, delta_t=1, overlap=0.5):

    idx0 = np.abs(window.x_axis - window.veh_state_x[0]).argmin()
    start_x_idx = np.abs(window.x_axis - start_x).argmin()
    end_x_idx = np.abs(window.x_axis - end_x).argmin()
    idx0 = max(idx0, start_x_idx) #the start of xcorr section can be taken after the min trajectory offset
    nch = end_x_idx - idx0
    
    dt = window.t_axis[1] - window.t_axis[0]
    XCF_out = np.zeros((nch, int(wlen/dt)))
    rel_dist_fact = np.arange(1, nch+1)
    for pivot_idx in range(idx0, end_x_idx):
        pivot = window.x_axis[pivot_idx]
        pivot_t_idx, nsamp, data, dt, f = preprocessing_window_NEW(window, pivot, -delta_t, time_window_to_xcorr)

        # xcorr with the channels to the left of the pivot
        XCF_out[: end_x_idx - pivot_idx, :] += XCORR_vshot_NEW(data[pivot_idx: end_x_idx+1, pivot_t_idx - nsamp:pivot_t_idx],
                              ivs=0, wlen=wlen, dt=dt, overlap_ratio=overlap, reverse=True)
        #ATTENTION, on change XCORR_vshot car on ne veut pas l'autocorr (on prend nch-1 plutot que nch)
    
    for i in range(XCF_out.shape[0]):
        XCF_out[i, :] /= rel_dist_fact[::-1][i]
    return post_processing_XCF_NEW(window, idx0, start_x_idx, end_x_idx, XCF_out, dt, norm, norm_amp, reverse=True)


def post_processing_XCF_NEW(window, pivot_idx, start_x_idx, end_x_idx, XCF_out, dt, norm, norm_amp, reverse):
    nt = XCF_out.shape[-1]
    t_axis = (np.arange(nt) - (nt // 2)) * dt

    if norm:
        XCF_out /= np.linalg.norm(XCF_out, axis=-1, keepdims=True)
    if norm_amp:
        XCF_out /= np.amax(XCF_out[0])
        
        
    if not reverse:
        XCF_out = XCF_out[::-1, :]
        XCF_out = XCF_out[:, ::-1]
        x_axis = window.x_axis[start_x_idx: pivot_idx] - window.x_axis[pivot_idx]
        
    else:
        x_axis = window.x_axis[pivot_idx: end_x_idx] - window.x_axis[pivot_idx-1]
        

    return XCF_out, x_axis, t_axis


# ATTENTION !!!
# ON DOIT ENCORE RETIRER DELTA_T POUR LES PIVOTS EN DEHORS DE TRAJ






def preprocessing_window(window, pivot, delta_t, start_x, end_x, time_window_to_xcorr):
    f = interpolate.interp1d(window.veh_state_x, window.veh_state_t, fill_value='extrapolate')

    dt = window.t_axis[1] - window.t_axis[0]
    pivot_idx = np.argmax(window.x_axis >= pivot)
    pivot_t = f(pivot) + delta_t
    pivot_t_idx = np.argmax(window.t_axis >= pivot_t)

    start_x_idx = np.argmax(window.x_axis >= start_x)
    end_x_idx = np.abs(window.x_axis - end_x).argmin()

    nsamp = int(time_window_to_xcorr // dt)

    data = window.data / np.linalg.norm(window.data)
    return pivot_idx, pivot_t_idx, start_x_idx, end_x_idx, nsamp, data, dt, f


def post_processing_XCF(window, pivot_idx, start_x_idx, end_x_idx, XCF_out, dt, norm, norm_amp, reverse=False):
    x_axis = window.x_axis[start_x_idx: end_x_idx] - window.x_axis[pivot_idx]
    nt = XCF_out.shape[-1]
    t_axis = (np.arange(nt) - (nt // 2)) * dt

    if norm:
        XCF_out /= np.linalg.norm(XCF_out, axis=-1, keepdims=True)

    if norm_amp:
        XCF_out /= np.amax(XCF_out[pivot_idx - start_x_idx])
    if not reverse:
        XCF_out = XCF_out[:, ::-1]

    return XCF_out, x_axis, t_axis


def construct_shot_gather_other_side(window: SurfaceWaveWindow, start_x, end_x, pivot, direction=1,
                                     wlen=2, norm=True, norm_amp=True,
                                     time_window_to_xcorr = 4, delta_t=1, overlap=0.5):
    pivot_idx, pivot_t_idx, start_x_idx, end_x_idx, nsamp, data, dt, f = preprocessing_window(window, pivot, -delta_t,
                                                                                              start_x, end_x,
                                                                                              time_window_to_xcorr)

    if direction >= 0:
        # xcorr with the channels from the sources to the right of the pivot
        XCF_out_right = XCORR_vshot(data[pivot_idx: end_x_idx, pivot_t_idx - nsamp:pivot_t_idx], ivs=0,
                              wlen=wlen, dt=dt, overlap_ratio=overlap,reverse=True)
    
        # xcorr with the channels to the left of the pivot
        XCF_out_left = xcorr_two_traces_based_on_traj(data, window.t_axis, pivot_idx, f,
                                                      start_x_idx, wlen, dt, nsamp, window.x_axis, delta_t=delta_t, overlap=overlap, reverse=True)
        
    else:
        XCF_out_left = XCORR_vshot(data[start_x_idx: pivot_idx + 1, pivot_t_idx - nsamp:pivot_t_idx],
                              pivot_idx - start_x_idx, wlen=wlen, dt=dt, overlap_ratio=overlap, reverse=True)
        
        XCF_out_right = xcorr_two_traces_based_on_traj(data, window.t_axis, pivot_idx, f,
                                                       end_x_idx, wlen, dt, nsamp, window.x_axis, delta_t=delta_t, overlap=overlap, reverse=True)

    XCF_out = np.concatenate((XCF_out_left, XCF_out_right), axis=0)

    return post_processing_XCF(window, pivot_idx, start_x_idx, end_x_idx, XCF_out, dt, norm, norm_amp, reverse=True)



def construct_shot_gather(window: SurfaceWaveWindow, start_x, end_x, pivot, direction=1,
                          wlen=2, norm=True, norm_amp=True,
                          time_window_to_xcorr = 4, delta_t=1, overlap=0.5):

    pivot_idx, pivot_t_idx, start_x_idx, end_x_idx, nsamp, data, dt, f = preprocessing_window(window, pivot, delta_t, start_x, end_x, time_window_to_xcorr)

    if direction >= 0:
        # xcorr with the channels to the left of the pivot
        XCF_out_left = XCORR_vshot(data[start_x_idx: pivot_idx + 1, pivot_t_idx:pivot_t_idx+nsamp],
                              pivot_idx - start_x_idx, wlen=wlen, dt=dt, overlap_ratio=overlap)
        # xcorr with the channels to the right of the pivot up to the source
        XCF_out_right = xcorr_two_traces_based_on_traj(data, window.t_axis, pivot_idx, f,
                                                       end_x_idx, wlen, dt, nsamp, window.x_axis, delta_t=delta_t, overlap=overlap)
        
    else:
        XCF_out_right = XCORR_vshot(data[pivot_idx: end_x_idx, pivot_t_idx:pivot_t_idx+nsamp], ivs=0,
                              wlen=wlen, dt=dt, overlap_ratio=overlap)
    
        XCF_out_left = xcorr_two_traces_based_on_traj(data, window.t_axis, pivot_idx, f,
                                                      start_x_idx, wlen, dt, nsamp, window.x_axis, delta_t=delta_t, overlap=overlap)
    

    XCF_out = np.concatenate((XCF_out_left, XCF_out_right), axis=0)


    return post_processing_XCF(window, pivot_idx, start_x_idx, end_x_idx, XCF_out, dt, norm, norm_amp)


def XCORR_two_traces(tr1, tr2, wlen, dt, overlap_ratio=0.5):
    nt = tr1.size
    wlen = int(wlen / dt)
    wlen_offset = int(wlen * (1 - overlap_ratio))
    nwin = (nt - wlen) // wlen_offset + 1

    XCF_out = np.zeros((1, wlen))

    for iwin in range(nwin):
        data_vs = repeat1d(tr1[(iwin * wlen_offset):(iwin * wlen_offset) + wlen])
        data_vr = tr2[(iwin * wlen_offset):(iwin * wlen_offset) + wlen]

        XCF_out += np.asarray(correlate(data_vs, data_vr,
                                                mode='valid', method='fft'))
    XCF_out = np.roll(XCF_out, wlen // 2, axis=-1)
    if nwin > 0:
        XCF_out /= nwin
    return XCF_out


def XCORR_vshot(data, ivs, wlen, dt, overlap_ratio=0.5, reverse=False):
    nch, nt = data.shape[0], data.shape[1]

    wlen = int(wlen / dt)

    wlen_offset = int(wlen * (1 - overlap_ratio))
    nwin = (nt - wlen) // wlen_offset + 1
    XCF_out = np.zeros((nch, wlen))
    for iwin in range(nwin):
        data_vs = repeat1d(data[ivs, (iwin * wlen_offset):(iwin * wlen_offset) + wlen])
        curt_XCF = []
        for ivr in range(nch):
            if reverse:
                vs = data[ivr, (iwin * wlen_offset):(iwin * wlen_offset) + wlen]
                vr = data_vs
            else:
                vs = data_vs
                vr = data[ivr, (iwin * wlen_offset):(iwin * wlen_offset) + wlen]
            curt_XCF.append(correlate(vs, vr, mode='valid', method='fft'))

        XCF_out += np.asarray(curt_XCF)
    if nwin == 0:
        return np.zeros((nch, wlen))
    else:
        return np.roll(XCF_out, wlen // 2, axis=-1) / nwin


def xcorr_two_traces_based_on_traj(data, t_axis, pivot_idx, f, boundary_idx, wlen, dt, nsamp, x_axis, delta_t=1, overlap=0.5, reverse=False):
    nch = abs(boundary_idx - pivot_idx) - 1
    if reverse:
        nch += 1
        
    XCF_out = np.zeros((nch, int(wlen // dt)))
    start_idx = min(pivot_idx, boundary_idx)
    end_idx = max(pivot_idx, boundary_idx)
    if reverse:
        start_idx -= 1
    for k, x_idx in enumerate(range(start_idx + 1, end_idx)):
        
        t = f(x_axis[x_idx])
        if reverse:
            t -= delta_t
        else:
            t += delta_t
        t_idx = np.argmax(t_axis >= t)
        if reverse:
            tr1 = data[pivot_idx, t_idx - nsamp: t_idx]
            tr2 = data[x_idx, t_idx - nsamp: t_idx]
        else:
            tr1 = data[pivot_idx, t_idx: t_idx + nsamp]
            tr2 = data[x_idx, t_idx:  t_idx + nsamp]

        if reverse:
            vs, vr = tr1, tr2
        else:
            vs, vr = tr2, tr1
        XCF_out[k] = XCORR_two_traces(vs, vr, wlen, dt, overlap_ratio=overlap)

    return XCF_out






def plot_psd_vs_offset(XCF_out, x_axis, t_axis, ax=None, fhi=20, figsize=(8, 8), pclip=98, log_scale=False,
                       x_max=200, x_min=0, fname=None, fdir='./', vmax=None, vmin=None):

    if x_axis[0] > x_axis[-1]:
        x_axis = x_axis * -1
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    dt = t_axis[1] - t_axis[0]
    fs = int(1 / dt)

    freq, Pxx_den = welch(XCF_out, fs, nperseg=256, nfft=1024)

    fhi_idx = np.argmax(freq >= fhi)
    spec = Pxx_den[:, :fhi_idx]
    if log_scale:
        spec = 10 * np.log10(spec)
    if not vmax:
        vmax = np.percentile(spec, pclip)
    if not vmin:
        vmin = np.percentile(spec, 100-pclip)

    x_max_idx = np.abs(x_max - x_axis).argmin()
    x_min_idx = np.abs(x_min - x_axis).argmin()
    min_idx = min(x_max_idx, x_min_idx)
    max_idx = max(x_max_idx, x_min_idx)
    spec = spec[min_idx: max_idx]

    ax.imshow(spec.T, extent=[x_axis[min_idx], x_axis[max_idx], freq[fhi_idx], freq[0]],
              cmap='jet', aspect='auto', vmax=vmax, vmin=vmin, interpolation='antialiased')
    ax.set_xlabel("Distance along the fiber [m]")
    ax.set_ylabel("Frequency [Hz]")

    if fname:
        fpath = os.path.join(fdir, fname)
        plt.savefig(fpath)
        print(f'{fpath} has been saved...')
        plt.close()
    else:
        plt.show()


def plot_spectrum_vs_offset(XCF_out, x_axis, t_axis, ax=None, fhi=20, figsize=(8, 8), fname=None, fdir='./'):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    Nt = XCF_out.shape[-1]
    dt = t_axis[1] - t_axis[0]
    freq = np.fft.fftfreq(Nt, d=dt)
    fhi_idx = np.argmax(freq >= fhi)
    spec = np.fft.fft(XCF_out, axis=-1)[:, :fhi_idx]
    ax.imshow(np.abs(spec).T, extent=[x_axis[0], x_axis[-1], freq[fhi_idx], freq[0]], cmap='jet', aspect='auto')
    ax.set_xlabel("Distance along the fiber [m]")
    ax.set_ylabel("Frequency [Hz]")
    if fname:
        fpath = os.path.join(fdir, fname)
        plt.savefig(fpath)
        print(f'{fpath} has been saved...')
        plt.close()
    else:
        plt.show()
        
def plot_xcorr(xcorr, t_axis, x_axis=None, ax=None, figsize=(8, 10),
               cmap='seismic', vmax_use_max=False,
               fig_dir=None,
               fig_name=None,
               fontsize=24, tickfont=20,
               x_lim=None,
               **plot_kwargs):
    if x_lim is None:
        x_lim = [-120, 120]
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    x_origin_index = np.abs(x_axis).argmin()
    xcorr /= np.amax(xcorr[x_origin_index])
    vmax = plot_kwargs.get("vmax", np.percentile(np.absolute(xcorr), 100)) if vmax_use_max else 1

    start_x = 0
    end_x = xcorr.shape[0]
    if x_axis is not None:
        start_x = -np.abs(x_axis[0])
        end_x = np.abs(x_axis[-1])

    if x_axis is not None:
        xcorr_to_plot = copy.deepcopy(xcorr)
    else:
        xcorr_to_plot = xcorr

    plt.imshow(xcorr_to_plot.T, aspect="auto", vmax=vmax, vmin=-vmax, cmap=cmap,
               extent=[start_x, end_x, t_axis[-1], t_axis[0]], interpolation='bicubic')
    plt.xlabel("Distance along the fiber [m]", fontsize=fontsize)
    plt.ylabel("Time lag [s]", fontsize=fontsize)

    ax.tick_params(axis='both', which='major', labelsize=tickfont)

    plt.xlim(x_lim)
    if fig_name and fig_dir:
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path, format = 'svg', bbox_inches = "tight")
        print(f'{fig_path} has saved...')
    else:
        plt.show()




class VirtualShotGather:
    def __init__(self, window: SurfaceWaveWindow, compute_xcorr=True, disp=None, include_other_side=False, new_xcorr=False, *args, **kwargs):
        self.window = window
        self.disp = disp
        if compute_xcorr:
            
            veh_direction = 1 #np.sign(np.mean(np.sign(np.diff(window.veh_state_x))))
            #print(veh_direction)
            
            if new_xcorr:
                self.XCF_out, self.x_axis, self.t_axis = construct_shot_gather_NEW(window, *args, **kwargs)
                print(self.XCF_out.shape)
                if include_other_side:
                    XCF_out_other_side, x_axis, _ = construct_shot_gather_otherside_NEW(window, *args, **kwargs)
                    print(XCF_out_other_side.shape)
                    self.XCF_out = np.vstack((self.XCF_out, XCF_out_other_side))
                    self.x_axis = np.hstack((self.x_axis, x_axis))
                    
            else:
                self.XCF_out, self.x_axis, self.t_axis = construct_shot_gather(window, *args, direction=veh_direction, **kwargs)
                if include_other_side:
                    XCF_out_other_side, _, _ = construct_shot_gather_other_side(window, *args, direction=veh_direction, **kwargs)
                    print(self.XCF_out.shape)
                    print(XCF_out_other_side.shape)
                    ch_idx_to_stack = np.linalg.norm(XCF_out_other_side, axis=-1) > 0
                    self.XCF_out[ch_idx_to_stack] = (self.XCF_out[ch_idx_to_stack] + XCF_out_other_side[ch_idx_to_stack]) / 2



    def __add__(self, other):
        sum_ = copy.deepcopy(self)
        length = min(self.XCF_out.shape[-1], other.XCF_out.shape[-1])
        sum_.XCF_out[:, :length] += other.XCF_out[:, :length]
        return sum_

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __truediv__(self, other):
        new_obj = copy.deepcopy(self)
        new_obj.XCF_out /= other
        return new_obj

    @classmethod
    def get_VirtualShotGather_obj(cls, fdir, fname):
        new_obj = cls(window=None, compute_xcorr=False)
        file = np.load(os.path.join(fdir, fname), allow_pickle=True)
        new_obj.XCF_out, new_obj.x_axis, new_obj.t_axis = file["XCF_out"], file["x_axis"], file["t_axis"]
        return new_obj

    def plot_spec_vs_offset(self, ax=None, psd=True, pclip=98, fdir='Fig/virtual_gathers', fname=None,
                            x_max=100, x_min=-100, log_scale=False,
                            vmin=None, vmax=None):
        if not os.path.exists(fdir):
            os.makedirs(fdir)
        if not psd:
            plot_spectrum_vs_offset(self.XCF_out, self.x_axis, self.t_axis, ax=ax, fdir=fdir, fname=fname)
        else:
            plot_psd_vs_offset(self.XCF_out, self.x_axis, self.t_axis, ax=ax, pclip=pclip,
                               x_max=x_max, x_min=x_min, fdir=fdir, fname=fname, log_scale=log_scale,
                               vmax=vmax, vmin=vmin)

    def save_to_npz(self, fname, fdir, **kwargs):
        np.savez(os.path.join(fdir, fname), XCF_out=self.XCF_out, x_axis=self.x_axis, t_axis=self.t_axis, **kwargs)

    def plot_image(self, fig_name=None, fig_dir=None,  x_lim=None, norm=False, plot_disp=False, plot_kwargs={}, **kwargs):
        if x_lim is None:
            x_lim = [-200, 200]
        if not plot_disp:
            ax = kwargs.get('ax')
            if not ax:
                fig, ax = plt.subplots(figsize=(8, 10))
            plot_xcorr(self.XCF_out, self.t_axis, self.x_axis, ax=ax, fig_dir=fig_dir, fig_name=fig_name, x_lim=x_lim, **plot_kwargs)
        else:
            assert self.disp, "please run obj.compute_disp_image() first"
            self.disp.plot_image(fig_dir, fig_name, norm=norm, **kwargs)


    def compute_disp_image(self, freqs = np.arange(0.8, 25, 0.1), vels = np.arange(200, 1200), norm=False, apodis=False, mod=True, start_x=None, end_x=None):
        if start_x is None:
            start_x = self.x_axis[0]
        if end_x is None:
            end_x = self.x_axis[-1]

        start_x_idx = np.abs(self.x_axis - start_x).argmin()
        end_x_idx = np.abs(self.x_axis - end_x).argmin()
        self.disp = Dispersion(self.XCF_out[start_x_idx: end_x_idx + 1],
                               self.x_axis[start_x_idx: end_x_idx + 1],
                               self.t_axis,
                               np.abs(self.x_axis[0]-self.x_axis[1]),
                               np.abs(self.t_axis[1] - self.t_axis[0]),
                               freqs=freqs, vels=vels,
                               norm=norm, apodis=apodis, mod=mod)

    def plot_disp(self, fig_name=None, fig_dir="Fig/dispersion/", norm=True, **kwargs):
        assert self.disp, "please run obj.compute_disp_image() first"
        self.disp.plot_image(fig_dir, fig_name, norm=norm, **kwargs)

    def save_disp_to_npz(self, *args, **kwargs):
        assert self.disp, "please run obj.compute_disp_image() first"
        self.disp.save_to_npz(*args, **kwargs)


    def norm(self):
        self.XCF_out /= np.linalg.norm(self.XCF_out, axis=-1, keepdims=True)