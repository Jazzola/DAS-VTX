import os
from multiprocessing import Pool
from datetime import datetime, timedelta
import numpy as np
from scipy.signal import savgol_filter, resample_poly, butter, sosfiltfilt, welch
from scipy.interpolate import RegularGridInterpolator
from scipy.fft import fft2, fftfreq, fftshift, ifft2, ifftshift
import matplotlib.pyplot as plt
from VSG_class import VirtualShotGather
from obspy.signal.filter import lowpass_cheby_2

from copy import deepcopy
from skimage.filters import window as taper_window
from scipy.signal.windows import tukey

from utils import get_file_section


#Ajouter la fonction de SNR
#Boucler sur les paramètres et enregistrer l'array de SNR à chaque fois et un dictionnaire des paramètres
#Enregistrer les VSGs
# >> On refait tourner le complet pour faire une sauvegarde ??
#On fait le test pour un stack pas trop gros pour aller vite (500 max)
# >> On peut faire tourner différents jobs (plutôt que d'écrire une fonction multi) et ne pas boucler pour aller plus vite aussi

#Avec l'array de SNR on fait un code à part pour plotter la figure en fonction parametres


def calculate_SNR(VSG):
    # Create meshgrid of time (yy) and space (xx) coordinates
    yy, xx = np.meshgrid(VSG.t_axis, VSG.x_axis)
 
    # Prevent division by zero by replacing zero values in yy
    yy_safe = np.where(yy == 0, 1e-6, yy)  # Small value instead of zero
    ratio = np.abs(xx/yy_safe)
 
    signal = VSG.XCF_out[(ratio < 1500)&(ratio > 300)].copy()
    signal = np.sqrt(np.mean(np.square(signal)))
    # noise = VSG.XCF_out[(ratio >= 1500)|(ratio <= 300)].copy()
    noise = VSG.XCF_out[(np.abs(yy) < 10)&(np.abs(yy) > 2.5)].copy()
    noise = np.sqrt(np.mean(np.square(noise)))
 
    # Avoid division errors
    if noise == 0:
        return np.inf  # Infinite SNR if noise is zero
 
    # Compute SNR in dB
    SNR = 20*np.log10(signal / noise)
 
    return SNR
    

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

    
def new_whiten_signals(X, fmin, fmax, fs=1.0, alpha=0.2, nperseg=256, noverlap=None):
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
        
        #NEW: tapering the spectre before ifft and cliping to freq of interest
        ifmin = np.argmin(np.abs(freqs-fmin))
        ifmax = np.argmin(np.abs(freqs-fmax))
        nidx_taper = int((ifmax+1-ifmin)*alpha/2)
        
        #ATTENTION
        #On a un probl�me avec le slicing li� au taper
        #Si on choisi une freq dans le taper on a une erreur
        
        X_fft[:ifmin -nidx_taper] = 0
        X_fft[ifmax +1 +nidx_taper:] = 0
        X_fft[ifmin -nidx_taper: ifmax +1 +nidx_taper] *= tukey(ifmax +1 -ifmin +2*nidx_taper, alpha=alpha)
        
        X_fft_whitened = X_fft
        
        # Convert back to time domain
        X_whitened[i] = np.real(np.fft.ifft(X_fft_whitened))
    
    return X_whitened

    


#base_dir = '/home/johannes/TEST_THOMAS/Workflow_vehicles_NewVersion/outputs/detects'
from config import PROCESSED_DIR
base_dir = PROCESSED_DIR / 'detects'

start_ch, end_ch = (15200, 15900) #(9400, 10150)  ##(9390, 10190)  (14600, 15300) (12330, 13750)
filelist = []
for day in os.scandir(base_dir):
    # if day.name[5] != '3':
    #     print(day.name[5])
    #     continue
    for hour in os.scandir(day.path):
        for file in os.scandir(hour.path):
            file_section = get_file_section(file) #ATTENTION, fonctionnne pour notre format de nom sp�cifique
            if start_ch == file_section[0] and end_ch == file_section[1]:
                filelist.append(file.path)
            # else:
            #     print(file.name)
import random
random.shuffle(filelist)
#filelist.sort()

# parameters={'overlap':0,
#             'delta_t':2,
#             'wlen':28,
#             'time_win':30.5}

#IL FAUDRA LE REFAIRE EN ACTIVANT LE ONEBIT
for o in [0.9]:
    for dt in [0.5]:
        for w in [30]: #On ne pourra pas les comparer car impacte direct sur le SNR ???
            for tw in [50]:
                parameters = {key:value for key, value in zip(['overlap', 'delta_t', 'wlen', 'time_win'], [o, dt, w, tw])}
                overlap = parameters['overlap']
                delta_t = parameters['delta_t']
                wlen = parameters['wlen']
                time_win = parameters['time_win']
                
                # for i in range(9):
                #     filechunk = filelist[i*100: i*100+200]
                
                parameters_str = f'large152-159_1000_o:{overlap}, dt:{delta_t}, w:{wlen}, twin:{time_win}' #{(start_ch+end_ch)//2}
                stack = []
                SNR_values = []
                
                n=0
                for file in filelist: #filechunk
                    if n >= 1000:
                       break
                
                    #print(hour.name, file.name)
                    sw = np.load(file, allow_pickle=True).item()
                    direction = np.sign(np.mean(np.sign(np.diff(sw.veh_state_x))))
                    if direction < 0:
                        print('Detection in opposite direction')
                        continue
                    
                    sos_t = butter(5, [0.5, 40], btype='bandpass', output='sos', fs=200)
                    sw.data = sosfiltfilt(sos_t, sw.data, axis=1)
                    sw.data = new_whiten_signals(sw.data, 0.7, 20, fs=200, nperseg=500, alpha=0.05)
                    
                    #sw.data = np.sign(sw.data) #on ne le met pas par defaut !!!
                    
                    
                
                    try:
                        # vsg=VirtualShotGather(sw, start_x=14600, pivot=14950, end_x=15300, wlen=wlen,
                        #                   overlap=overlap, delta_t=delta_t, time_window_to_xcorr=time_win,
                        #                   include_other_side=True, norm=False, new_xcorr=True)
                        
                        # vsg=VirtualShotGather(sw, start_x=12330, pivot=13040, end_x=13750, wlen=wlen,
                        #                   overlap=overlap, delta_t=delta_t, time_window_to_xcorr=time_win,
                        #                   include_other_side=True, norm=False, new_xcorr=True)
                        
                        # vsg=VirtualShotGather(sw, start_x=9390, pivot=9790, end_x=10190, wlen=wlen,
                        #                       overlap=overlap, delta_t=delta_t, time_window_to_xcorr=time_win,
                        #                       include_other_side=True, norm=False, norm_amp=True, new_xcorr=True)

                        # vsg=VirtualShotGather(sw, start_x=8400, pivot=9775, end_x=11150, wlen=wlen,
                        #                       overlap=overlap, delta_t=delta_t, time_window_to_xcorr=time_win,
                        #                       include_other_side=True, norm=False, norm_amp=True, new_xcorr=True)

                        vsg=VirtualShotGather(sw, start_x=14150, pivot=15550, end_x=17050, wlen=wlen,
                                              overlap=overlap, delta_t=delta_t, time_window_to_xcorr=time_win,
                                              include_other_side=True, norm=False, norm_amp=True, new_xcorr=True)
                        
                    except ValueError:
                        continue
                    
                    if np.isnan(vsg.XCF_out).any():
                        continue

                    n+=1
                    
                    vsg.norm()
                    
                    if not stack:
                        stack.append(vsg)
                    else:
                        stack[0]+=vsg
                
                    SNR = calculate_SNR(stack[0])
                    SNR_values.append(SNR)
                
                
                print('Number of individual VSGs:', n)
                vsg_stack = stack[0]
                
                
                # fig, ax = plt.subplots(figsize=(5, 5))
                # ax.plot(SNR_values, 'k')
                # ax.scatter(len(SNR_values), SNR_values[-1], c='r')
                # ax.set_xlabel('Number of VSGs in the stack', fontsize='x-large')
                # ax.set_ylabel('SNR [-]', fontsize='x-large')
                # plt.grid(visible=True)
                # plt.savefig('/pfs/data6/home/ka/ka_agw/ka_wb2462/Vehicles_test_trains/figs_xcorr/SNR_'+parameters_str+'_.png')
                # plt.close()
                
                
                # vsg_stack.plot_image(x_lim=[-1420, 1420])
                # plt.title(parameters_str)
                # #plt.show()
                # plt.savefig('/pfs/data6/home/ka/ka_agw/ka_wb2462/Vehicles_test_trains/figs_xcorr/stack_raw_'+parameters_str+'_.png')
                # plt.close()
                
                np.savez('/pfs/data6/home/ka/ka_agw/ka_wb2462/Vehicles_test_trains/stacks/stack_raw_'+parameters_str, parameters=parameters, SNR=SNR_values, stack=vsg_stack, allow_pickle=True)


                
                # vsg_stack.norm()
                
                # # vsg_stack.plot_image(x_lim=[-1420, 1420])
                # # plt.title(parameters_str)
                # # #plt.show()
                # # plt.savefig('/pfs/data6/home/ka/ka_agw/ka_wb2462/Vehicles_test_trains/figs_xcorr/stack_norm_'+parameters_str+'_.png')
                # # plt.close()
                
                
                # # vsg_stack.compute_disp_image(freqs=np.linspace(1, 25, 24*wlen), vels=5/np.linspace(1/(2*9.6), 1/2800, 5*140)) #1400, 70
                # # test_disp = vsg_stack.disp.fv_map
                # # pclip = 99
                # # vmax = np.percentile(np.abs(test_disp[~np.isnan(test_disp)]), pclip)
                # # vmin = np.percentile(np.abs(test_disp[~np.isnan(test_disp)]), 100-pclip)
                # # plt.pcolormesh(vsg_stack.disp.freqs,
                # #                vsg_stack.disp.vels,
                # #                test_disp,
                # #               cmap="jet",
                # #               vmax=vmax,
                # #               vmin=vmin)
                # # plt.xscale('log')
                # # plt.xlim(1, 20)
                # # plt.yscale('log')
                # # plt.ylim(200, 2000)
                # # plt.title(parameters_str)
                # # #plt.show()
                # # plt.savefig('/pfs/data6/home/ka/ka_agw/ka_wb2462/Vehicles_test_trains/figs_xcorr/disp_norm_'+parameters_str+'_.png')
                # # plt.close()
                
                
                
                # vsg_stack.XCF_out, _, _, _ = k_manual_filt(vsg_stack.x_axis, vsg_stack.t_axis, vsg_stack.XCF_out, npts=2)
                
                
                
                # #NOUVELLE FONCTION DE SEMBLANCE CORRIGEE
                # def semb(slw):
                #     sem = np.empty((vsg_stack.XCF_out.shape[0], vsg_stack.XCF_out.shape[1]//decim))
                #     causal_t = vsg_stack.t_axis[vsg_stack.t_axis>=0]
                #     acausal_t = vsg_stack.t_axis[vsg_stack.t_axis<0]
                #     for i in range(vsg_stack.XCF_out.shape[0]):
                #         Gtosum = 0
                #         GHtosum = 0
                #         for x in range(max(0, i-xwin), min(i+xwin+1, vsg_stack.XCF_out.shape[0])):
                #             t_interp_caus = causal_t + slw*(np.abs(vsg_stack.x_axis[x]) - np.abs(vsg_stack.x_axis[i])) #on enl�ve les abs si on veut regarder les slowness negatifs
                #             t_interp_acaus = acausal_t - slw*(np.abs(vsg_stack.x_axis[x]) - np.abs(vsg_stack.x_axis[i]))
                #             t_interpolation_axis = np.hstack((t_interp_acaus, t_interp_caus)) #est ce qu'on passe de l'autre cot� du z�ro avec le t shift ??
                #             og, tg = np.meshgrid(vsg_stack.x_axis[x], t_interpolation_axis, indexing='ij') #regular grid
                #             interpolated_data = interpolator((og,tg))[0,:]
                #             Gtosum += interpolated_data
                #             GHtosum += interpolated_data**2
                
                #         numerateur = Gtosum**2
                #         denominateur = GHtosum
                
                #         for semj in range(0, vsg_stack.XCF_out.shape[1]//decim):
                #             j=semj*decim
                #             num_win = numerateur[max(0, j-twin):min(j+twin+1, vsg_stack.XCF_out.shape[1])]
                #             num_win = num_win[~np.isnan(num_win)]
                #             num = np.real(np.sum(num_win))
                #             deno_win = denominateur[max(0, j-twin):min(j+twin+1, vsg_stack.XCF_out.shape[1])]
                #             deno_win = deno_win[~np.isnan(deno_win)]
                #             deno = np.real(np.sum(deno_win)) #sum(denominateur) >> ERREUR ??????
                #             sem[i, semj] = num/deno
                #     return np.abs(sem)/(2*xwin+1)
                
                # slow2test = np.linspace(1/300, 1/1500, 16)
                # #slow2test = 1/slow2test
                # twin=int(200*0.1)
                # xwin=int(50/9.6)
                # decim = 10
                
                # interpolator = RegularGridInterpolator((vsg_stack.x_axis, vsg_stack.t_axis), vsg_stack.XCF_out,
                #                                         bounds_error=False, fill_value=0)
                # with Pool(8) as p:
                #     results = p.map(semb, list(slow2test))
                # sem = 2*np.array(sum(results))/len(slow2test)
                # if vsg_stack.t_axis.shape[0]%decim==0:
                #     shift=vsg_stack.t_axis[::decim]
                # else:
                #     shift=vsg_stack.t_axis[:-decim:decim]
                # sem_interp = RegularGridInterpolator((vsg_stack.x_axis, shift), sem, #am�liorer le slicing
                #                                      bounds_error=False, fill_value=0)
                # xx, yy = np.meshgrid(vsg_stack.x_axis, vsg_stack.t_axis, indexing='ij')
                # sem = sem_interp((xx, yy))
                # vsg_stack.XCF_out *= sem
                
                # np.savez('/pfs/data6/home/ka/ka_agw/ka_wb2462/Vehicles_test_trains/stacks/stack_semb_'+parameters_str, parameters=parameters, stack=vsg_stack, allow_pickle=True)
                
                # # vsg_stack.plot_image(x_lim=[-1420, 1420])
                # # #plt.show()
                # # plt.title(parameters_str)
                # # plt.savefig('/pfs/data6/home/ka/ka_agw/ka_wb2462/Vehicles_test_trains/figs_xcorr/stack_semb_'+parameters_str+'_.png')
                # # plt.close()
                
                
                # vsg_stack.compute_disp_image(freqs=np.linspace(0.7, 25, 24*wlen), vels=1/np.linspace(1/(2*9.6), 1/3200, wlen*2*80))
                # #vels=1/np.linspace(1/(2*9.6), 1/3200, wlen*2*80))
                # #vels=1/np.linspace(1/(2*9.6), 1/5680, wlen*2*142))
                # test_disp = vsg_stack.disp.fv_map
                # pclip = 99
                # vmax = np.percentile(np.abs(test_disp[~np.isnan(test_disp)]), pclip)
                # vmin = np.percentile(np.abs(test_disp[~np.isnan(test_disp)]), 100-pclip)
                # plt.pcolormesh(vsg_stack.disp.freqs,
                #                vsg_stack.disp.vels,
                #                test_disp,
                #               cmap="jet",
                #               vmax=vmax,
                #               vmin=vmin)
                # plt.xscale('log')
                # plt.xlim(0.7, 20)
                # plt.yscale('log')
                # plt.ylim(200, 2000)
                # plt.title(parameters_str)
                # #plt.show()
                # plt.savefig('/pfs/data6/home/ka/ka_agw/ka_wb2462/Vehicles_test_trains/figs_xcorr/disp_semb_'+parameters_str+'_.png')
                # plt.close()