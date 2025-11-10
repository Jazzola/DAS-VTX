import os
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import RegularGridInterpolator

from skimage.filters import window


def fk(data, dx, dt):
    # From Ariel's repo
    (nch, nt) = np.shape(data)
    nf = 2 ** (1 + math.ceil(math.log(nt, 2)))
    nk = 2 ** (1 + math.ceil(math.log(nch, 2)))

    fft_f = np.arange(-nf / 2, nf / 2) / nf / dt
    fft_k = np.arange(-nk / 2, nk / 2) / nk / dx

    fk_res = np.fft.fftshift(np.fft.fft2(data, s=[nk, nf]))
    fk_res = np.absolute(fk_res) #cette fonction prend le module du fk et pas la partie réelle... essayer en changeant
    #fk_res = fk_res.real "la partie réelle ne semble pas etre exploitable, on a seulement l'énergie sur les valeur de k qui vont bien ??
    
    return fk_res, fft_f, fft_k

def fk_new(data, dx, dt, apodis=False):
    
    if apodis:
        data *= window(("tukey", 0.4), data.shape)
    fk_raw= np.fft.fft2(data)
    fft_k = np.fft.fftfreq(fk_raw.shape[0], dx)
    fft_f = np.fft.fftfreq(fk_raw.shape[1], dt)
    
# =============================================================================
#     plt.pcolormesh(np.fft.fftshift(fft_f[np.abs(fft_f)<=20]), np.fft.fftshift(fft_k),
#                    np.fft.fftshift(np.real(fk_raw)[:, np.abs(fft_f)<=20]-np.mean(np.real(fk_raw)[:, np.abs(fft_f)<=20])),
#                    cmap='seismic')
#     plt.show()
# =============================================================================
    
# =============================================================================
#     fk_res = fk_raw[fft_k>=0, :]
#     fk_res[1:, :] += fk_raw[fft_k<0, :][-2+np.shape(fft_k)[0]%2::-1, :]
#     fk_res[1:, :] /= 2
# =============================================================================
    fk_res = fk_raw[fft_k>=0, :][:, fft_f>=0]
    #fk_res[1:, 1:] += fk_raw[fft_k<0, :][:, fft_f<0][-2+np.shape(fft_k)[0]%2::-1, -2+np.shape(fft_f)[0]%2::-1]
    #fk_res[1:, 1:] += fk_raw[fft_k<0, :][:, fft_f>0][-2+np.shape(fft_k)[0]%2::-1, :]
    #fk_res[1:, 1:] += fk_raw[fft_k>0, :][:, fft_f<0][:, -2+np.shape(fft_f)[0]%2::-1]
    #fk_res[1:, 1:] /= 2
    
    fft_k = fft_k[fft_k>=0]
    fft_f = fft_f[fft_f>=0]
    
# =============================================================================
#     plt.pcolormesh(fft_f[fft_f<=20], fft_k,
#                    np.real(fk_res)[:, fft_f<=20]-np.mean(np.real(fk_res)[:, fft_f<=20]),
#                    cmap='seismic')
#     plt.show()
# =============================================================================

# =============================================================================
#     fft_k = np.fft.fftshift(fft_k)
#     fft_f = np.fft.fftshift(fft_f)
#     fk_res = np.fft.fftshift(fk_raw) #fk_res, axes=1)
# =============================================================================
    
    fk_res = fk_res #on renvoie pas abs pcq on veut éventuellement stacker dans le map_fv
    
    return fk_res, fft_f, fft_k
    

def map_fv(data, x_axis, t_axis, dx, dt, freqs, vels, norm=False, apodis=False, mod=True):
    nscanv = np.size(vels)

    if norm:
        data = data / np.linalg.norm(data, axis=-1, keepdims=True, ord=1)
    
    
    #fk_res, fft_f, fft_k = fk_new(data, dx, dt)
    
    backward_prop, forward_prop, acausal, causal = False, False, False, False
    if x_axis[x_axis<0].shape[0] >1:
        backward_prop = True
    if x_axis[x_axis>0].shape[0] >1:
        forward_prop = True
    if t_axis[t_axis<0].shape[0] >1:
        acausal = True
    if t_axis[t_axis>0].shape[0] >1:
        causal = True
    
    if forward_prop and acausal:
        vsg_forw = data[x_axis>=0, :][:, t_axis<=0] #on peut inverser deux fois ou ne pas inverser dans ce cas,
        #pas besoin d'inverser deux fois si on stack les deux quadrants principaux de toute façon
        fk_forw_acaus, fft_f, fft_k = fk_new(vsg_forw, dx, dt, apodis)
        #fk_forw = fk_forw[::-1, :] #si on stack tout les quartiers on a pas besoin de ça, ah bon ??
    if forward_prop and causal:
        vsg_forw = data[x_axis>=0, :][:, t_axis>=0][::-1, :]
        fk_forw_caus, fft_f, fft_k = fk_new(vsg_forw, dx, dt, apodis)
    if backward_prop and acausal:
        vsg_back = data[x_axis<=0, :][:, t_axis<=0][:, ::-1]
        fk_back_acaus, fft_f, fft_k = fk_new(vsg_back, dx, dt, apodis)
    if backward_prop and causal:
        vsg_back = data[x_axis<=0, :][:, t_axis>=0]
        fk_back_caus, fft_f, fft_k = fk_new(vsg_back, dx, dt, apodis)
        
    #checking every corner combination
    if backward_prop and not forward_prop:
        if acausal and not causal:
            fk_res = fk_back_acaus
        elif causal and not acausal:
            fk_res = fk_back_caus
        else:
            fk_res = fk_back_acaus + fk_back_caus
            
    elif forward_prop and not backward_prop:
        if acausal and not causal:
            fk_res = fk_forw_acaus
        elif causal and not acausal:
            fk_res = fk_forw_caus
        else:
            fk_res = fk_forw_acaus + fk_forw_caus
            
    else: #both
        if acausal and not causal:
            fk_res = fk_forw_acaus + fk_back_acaus
        elif causal and not acausal:
            fk_res = fk_forw_caus + fk_back_caus
        else:
            fk_res = fk_forw_acaus + fk_forw_caus + fk_back_acaus + fk_back_caus

    if mod:
        fk_res = np.abs(fk_res)
    else:
        fk_res = np.real(fk_res)
    
    interp_fun = RegularGridInterpolator((fft_k, fft_f), fk_res, bounds_error=False)
    
    ones_arr = np.ones(nscanv)
    fv_map = np.zeros((len(freqs), len(vels)), dtype=np.float32)
    for ind, fr in enumerate(freqs):
        #test = fr*ones_arr/vels
        #print(test[0], test[-1])
        xx, yy = np.meshgrid(fr*ones_arr/vels, fr)
        fv_map[ind, :] = np.squeeze(interp_fun((xx, yy)))
        #print('n nans:', fv_map[ind, :][np.isnan(fv_map[ind, :])].shape)
    fv_map = savgol_filter(fv_map, 25, 4, axis=0)
# =============================================================================
#     fv_map = savgol_filter(fv_map, 25, 4, axis=1)
# =============================================================================

    return fv_map.T

def plot_fv_map(fv_map, freqs, vels, norm=True, fig_dir="Fig/", fig_name=None, ax=None, pclip=100, fontsize=24, tickfont=20, **kwargs):

    norm = True
    if norm:
        row_sums = np.amax(fv_map, axis=0)
        fv_map = fv_map / row_sums
    if not ax:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8,10)))

    pclip = 90
    vmax = np.percentile(np.abs(fv_map), pclip)
    vmin = np.percentile(np.abs(fv_map), 100-pclip)

    ax.imshow(fv_map, aspect="auto",
              extent=[freqs[0], freqs[-1], vels[0], vels[-1]],
              cmap="jet",
              vmax=vmax,
              vmin=vmin)

    ax.grid()
    ax.set_xscale(kwargs.get('set_xscale', 'linear'))

    ax.set_xlabel("Frequency [Hz]", fontsize=fontsize)
    ax.set_ylabel("Phase velocity [m/s]", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tickfont)
    plt.tight_layout()
    if fig_name:
        fig_path = os.path.join(fig_dir, fig_name)
        print(f'saving {fig_path}...')
        plt.savefig(f"{fig_path}")
        plt.close()
    else:
        plt.show()


class Dispersion:
    def __init__(self, data, x_axis, t_axis, dx, dt, freqs, vels, norm=False, apodis=False, mod=True, compute_fv=True):
        self.data = data
        self.x_axis = x_axis
        self.t_axis = t_axis
        self.dx = dx
        self.dt = dt
        self.freqs = freqs
        self.vels = vels
        self.norm = norm
        self.apodis = apodis
        self.mod = mod
        if compute_fv:
            self._map_fv()

    def save_to_npz(self, fname, fdir='./'):
        np.savez(os.path.join(fdir, fname), freqs=self.freqs, vels=self.vels, fv_map=self.fv_map)

    @classmethod
    def get_dispersion_obj(cls, fname, fdir='./'):
        file = np.load(os.path.join(fdir, fname))
        obj = Dispersion(data=None, dx=None, dt=None, freqs=file['freqs'], vels=file['vels'], compute_fv=False)
        obj.fv_map = file['fv_map']
        return obj

    def _map_fv(self):
        self.fv_map = map_fv(self.data, self.x_axis, self.t_axis, self.dx, self.dt, freqs=self.freqs, vels=self.vels, norm=self.norm, apodis=self.apodis, mod=self.mod)

    def plot_image(self, fig_dir=None, fig_name=None, norm=False, **kwargs):
        norm = norm or self.norm
        print(fig_name)
        plot_fv_map(self.fv_map, self.freqs, self.vels, norm, fig_dir, fig_name, **kwargs)

    def __add__(self, other):
        sum_ = Dispersion(self.data, self.dx, self.dt, self.freqs, self.vels, compute_fv=False)
        sum_.fv_map = self.fv_map + other.fv_map
        return sum_

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __truediv__(self, other: float):
        div_ = copy.deepcopy(self)
        div_.fv_map /= other
        return div_