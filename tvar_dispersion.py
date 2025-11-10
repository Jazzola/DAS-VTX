#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing  import Pool
from copy import deepcopy

from scipy.signal.windows import tukey
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.signal import find_peaks
from scipy.signal import correlate, correlation_lags

from skimage.filters import window

import meteostat as ms
from datetime import datetime, timedelta

from config import freqs, vels, xcorr_parameters



disp_dir = '/home/johannes/THOMAS/DAS-VTX_WIP/out_test/DISPs' #On importe les dispersions préalablement enregistrées
piezo_file ='/home/johannes/THOMAS/DATA/GW_2023.csv'

filelist = []
for file in os.scandir(disp_dir):
    filelist.append(file.path)
filelist.sort()


date_axis = np.empty(len(filelist), dtype='datetime64[m]')
f = [10] #[2, 4, 6, 7, 8, 9, 10]
spectrum_amp = np.empty((len(date_axis), len(vels), len(f)))


for i, file in enumerate(filelist):
    disp = np.load(file, allow_pickle=True)['disp']
    
    for p, fvalue in enumerate(f):
        c_test = disp[:, np.argmin(np.abs(freqs - fvalue))]
        c_var = c_test.reshape(vels.shape)
        spectrum_amp[i, :, p] = c_var

    st = file.split('_st_')[1][0:13]
    end = file.split('-end_')[1][0:13]
    #Attention, ici date prise au start. On peut envisager de changer si besoin
    st = st[:4]+'-'+st[4:6]+'-'+st[6:8]+'T'+st[9:11]+':'+st[11:]
    date_axis[i] = st
    

#data meteo
start = datetime(2023, 1, 1)
end = datetime(2023, 10, 31)
data_bruch = ms.Daily('D0731', start, end)
data_bruch = data_bruch.fetch()
data_rulz = ms.Daily('D4310', start, end)
data_rulz = data_rulz.fetch()

prcp_bruch = data_bruch['prcp'].to_numpy()
prcp_rulz = data_rulz['prcp'].to_numpy()
av_prcp = (prcp_bruch+prcp_rulz)/2
meteo_date = np.arange(start, end+timedelta(days=1), timedelta(days=1)).astype(datetime)



piezo_data = np.loadtxt(piezo_file, dtype=str, delimiter=',', skiprows=1)

gw_base = piezo_data[:, 3].astype(int)
gw_var = piezo_data[:, 4].astype(int)
gw_lvl = gw_base + gw_var/100
gw_date = piezo_data[:, 1]

for q, k in enumerate(gw_date):
    gw_date[q] = datetime.strptime(k,'%d.%m.%Y %H:%M')
gw_date = gw_date.astype('datetime64[m]')



x, y = np.meshgrid(date_axis, vels, indexing='ij')

ylowlim = 700 #530 pour 4Hz
if not np.any(np.isnan(spectrum_amp[(y>=ylowlim)&(y<=1200)])):
    spectrum_amp = spectrum_amp[(y>=ylowlim)&(y<=1200)]
    x = x[(y>=ylowlim)&(y<=1200)]
    y = y[(y>=ylowlim)&(y<=1200)]
    
    spectrum_amp = spectrum_amp.reshape((len(date_axis), len(x)//len(date_axis), len(f)))
    y = y.reshape((len(date_axis), len(x)//len(date_axis)))
    x = x.reshape((len(date_axis), len(x)//len(date_axis)))
    
    for j in range(len(f)):
        fig = plt.figure(figsize=(8,10))
        gs = fig.add_gridspec(3, 10)
        ax1 = fig.add_subplot(gs[0:2,:9])
        ax2 = fig.add_subplot(gs[2,:9], sharex=ax1)
        cax = fig.add_subplot(gs[: ,9])
        
        max_pos = vels[vels>=ylowlim][np.argmax(spectrum_amp[:,:,j], axis=1)]
        
        im = ax1.pcolormesh(x, y, spectrum_amp[:,:,j], cmap='jet',
                       vmax=np.percentile(spectrum_amp[:,:,j], 99),
                       vmin=np.percentile(spectrum_amp[:,:,j], 10))
        ax1.plot(date_axis, max_pos, 'white', linewidth=3)
        
        ax2.bar(meteo_date, av_prcp)
        ax2.plot(gw_date, gw_lvl*np.max(av_prcp)/np.max(gw_lvl), 'g') #on retire le correlation lag car on veut la moyenne avant le jour meme
        
        ax1.set(title=f'{f[j]}Hz', yscale='linear')
        ax2.set_xlim(date_axis[0], date_axis[-1])
        ax1.tick_params(labelbottom=False)
        ax2.tick_params(axis='x', labelrotation=-45)
        
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.tight_layout()
        plt.show()
        
        
# =============================================================================
#     # EN INTERPOLANT LA VITESSE
#     velocity_interpolator = interp1d(date_axis.astype(int), max_pos)
#     meteo_date64 = meteo_date.astype('datetime64[m]')
#     meteo_date_int = meteo_date64.astype(int)
#     dd_new = meteo_date_int[(meteo_date_int>=date_axis.astype(int)[0])&
#                             (meteo_date_int<=date_axis.astype(int)[-1])]
#     max_pos_interp = velocity_interpolator(dd_new)
#     v_der = np.diff(max_pos)/np.diff(date_axis).astype(int)
#     v_der_interp = np.diff(max_pos_interp)/np.diff(dd_new)
#     dd = (date_axis[:-1].astype(int) + date_axis[1:].astype(int))/2
#     dd_new = (dd_new[:-1] + dd_new[1:])/2
# =============================================================================
    # EN INTERPOLANT LA DERIVE DE LA VITESSE
    v_der = np.diff(max_pos)/np.diff(date_axis).astype(int)
    #ATTENTION, on interpole avant correl car position absolue en temps
    dd = (date_axis[:-1].astype(int) + date_axis[1:].astype(int))/2 #Est-ce que c'est mieux de prendre au milieu ??
    velocity_interpolator = interp1d(dd, v_der)
    meteo_date64 = meteo_date.astype('datetime64[m]')
    meteo_date_int = meteo_date64.astype(int)
    dd_new = meteo_date_int[(meteo_date_int>=dd[0])&
                            (meteo_date_int<=dd[-1])] #On interpole sur un jour donc on réduit le sampling pour la majorité de la période
# =============================================================================
#     dd_new = np.arange(dd[0], dd[-1]+1)
#     #on a les minutes en integer donc seulement besoin d'un arange avec step de 1
# =============================================================================
    v_der_interp = velocity_interpolator(dd_new)
    
    plt.plot(dd.astype('datetime64[m]'), v_der, 'r', linewidth=0.7)
    plt.plot(dd_new.astype('datetime64[m]'), v_der_interp, 'k', alpha=0.7)
    plt.show()
    v_der_interp -= np.mean(v_der_interp)
    v_der_interp /= np.max(np.abs(v_der_interp))
    
    gw_date_int = gw_date.astype(int)
    gw_interpolator = interp1d(gw_date_int, gw_lvl)
    gw_lvl_interp = gw_interpolator(dd_new)
    gw_lvl_interp -= np.min(gw_lvl_interp)
    gw_lvl_interp /= np.max(np.abs(gw_lvl_interp))
    
    for run_win in [1]:
        running_prcp = correlate(av_prcp, np.ones(run_win), mode='same')/run_win
        #running_prcp -= np.mean(running_prcp)
        running_prcp /= np.max(running_prcp)
        
# =============================================================================
#         meteo_interpolator = interp1d(meteo_date_int, running_prcp)
#         dd_new = np.arange(meteo_date_int[0], meteo_date_int[-1]+1)
#         running_prcp_interp = meteo_interpolator(dd_new)
#         running_prcp = running_prcp_interp
# =============================================================================
        
        vel_corr = correlate(running_prcp,
                             v_der_interp,
                             mode='valid')
        time_diff = dd_new.astype('datetime64[m]')[0]-meteo_date64[0] #date_axis[0]
        time_diff = time_diff.astype(int)
        vel_corr_lags = correlation_lags(len(running_prcp), len(v_der_interp),
                                         mode='valid') - time_diff/(24*60) - (run_win-1)//2
        
        plt.plot(vel_corr_lags, vel_corr)
        plt.show()
        
        retard = int(-1*vel_corr_lags[np.argmin(vel_corr[vel_corr_lags<=0])]) #On veut le minimum pour vérifier l'anti-corr
        #plt.plot(date_axis[:-1], v_der/np.max(v_der), 'k')
        anti_corr2plot = (-1)*v_der_interp #- np.mean((-1)*v_der_interp)
        plt.plot(dd_new.astype('datetime64[m]'), anti_corr2plot, 'k') #on plot l'anti-corr
        plt.plot(meteo_date+timedelta(days=retard),
                 running_prcp, 'magenta')
        plt.xlim(date_axis[0], date_axis[-1])
        plt.show()
        
        
        gw_corr = correlate(running_prcp, gw_lvl_interp, mode='valid')
        gw_corr_lags = correlation_lags(len(running_prcp), len(gw_lvl_interp),
                                        mode='valid') - time_diff/(24*60) - (run_win-1)//2
        plt.plot(gw_corr_lags, gw_corr)
        plt.show()
        
        retard = int(-1*gw_corr_lags[np.argmax(gw_corr[gw_corr_lags<=0])])
        plt.plot(dd_new.astype('datetime64[m]'), gw_lvl_interp, 'g')
        plt.plot(meteo_date+timedelta(days=retard),
                 running_prcp, 'magenta')
        plt.xlim(date_axis[0], date_axis[-1])
        plt.show()
        
        gw_lvl_interp -= np.mean(gw_lvl_interp)
        gw_lvl_interp /= np.max(np.abs(gw_lvl_interp))
        max_pos2plot = max_pos - np.mean(max_pos)
        max_pos2plot /= np.max(np.abs(max_pos2plot))
        max_pos2plot *= -1
        plt.plot(dd_new.astype('datetime64[m]'), gw_lvl_interp, 'g')
        #plt.plot(dd_new.astype('datetime64[m]'), anti_corr2plot, 'k', alpha=0.5)
        plt.plot(date_axis+(0*24*60), max_pos2plot, 'k')
        plt.show()
        

