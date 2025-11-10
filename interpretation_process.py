import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from copy import deepcopy

from config import PROCESSED_DIR, dx
from config import offsets_to_keep, lags_to_keep, stack_norm, coherence_enhancing
from config import slw_list, twin, xwin, decim, n_processes
from config import freqs, vels
from config import disp_masking, max_cut_dist, min_cut_dist, step_factor
from utils import Coherence_Enhancement

from scipy.interpolate import RegularGridInterpolator



def interpretation_process(stack_file):

    if not os.path.exists(PROCESSED_DIR / 'FIG'):
                    os.mkdir(PROCESSED_DIR / 'FIG')
    if not os.path.exists(PROCESSED_DIR / 'FIG' / 'Stack'):
                    os.mkdir(PROCESSED_DIR / 'FIG' / 'Stack')
    if not os.path.exists(PROCESSED_DIR / 'FIG' / 'Disp'):
                    os.mkdir(PROCESSED_DIR / 'FIG' / 'Disp')
    if not os.path.exists(PROCESSED_DIR / 'FIG' / 'SNR'):
                    os.mkdir(PROCESSED_DIR / 'FIG' / 'SNR')
                    
    stack_name = stack_file.split('/')[-1]
    if not os.path.exists(PROCESSED_DIR / 'FIG' / 'Disp' / stack_name):
                    os.mkdir(PROCESSED_DIR / 'FIG' / 'Disp' / stack_name)

    content = np.load(stack_file, allow_pickle=True)
    parameters = content['parameters'].item()
    stack_full = content['stack'].item()
    SNR_values = content['SNR']


    fig, ax = plt.subplots()
    pclip = np.percentile(np.abs(stack_full.XCF_out), 99)
    ax.pcolormesh(stack_full.t_axis, stack_full.x_axis, stack_full.XCF_out,
                    cmap='seismic', vmax=pclip, vmin=-pclip)
    ax.set_xlabel('Timelag [s]', fontsize=12)
    ax.set_ylabel('Inter-station distance [m]', fontsize=12)
    plt.savefig(PROCESSED_DIR / 'FIG' / 'Stack' / stack_name.replace('.npz', '.png'))
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(SNR_values, 'k')
    ax.set_xlabel('Number of VSGs in the stack', fontsize=12)
    ax.set_ylabel('SNR [-]', fontsize=12)
    ax.tick_params(labelsize=10)
    plt.grid(visible=True)
    plt.savefig(PROCESSED_DIR / 'FIG' / 'SNR' /
                stack_name.replace('.npz', '.png'))
    plt.close()


    if offsets_to_keep == 'backward':
        stack_full.XCF_out = stack_full.XCF_out[stack_full.x_axis<=0, :]
        stack_full.x_axis = stack_full.x_axis[stack_full.x_axis<=0]
    elif offsets_to_keep == 'forward':
        stack_full.XCF_out = stack_full.XCF_out[stack_full.x_axis>=0, :]
        stack_full.x_axis = stack_full.x_axis[stack_full.x_axis>=0]
    elif offsets_to_keep == 'both':
        if np.abs(stack_full.x_axis[0]) > np.abs(stack_full.x_axis[-1]):
            stack_full.x_axis = stack_full.x_axis[1:]
            stack_full.XCF_out = stack_full.XCF_out[1:]
        elif np.abs(stack_full.x_axis[-1]) > np.abs(stack_full.x_axis[0]):
            stack_full.x_axis = stack_full.x_axis[:-1]
            stack_full.XCF_out = stack_full.XCF_out[:-1]

    if lags_to_keep == 'causal':
        stack_full.XCF_out = stack_full.XCF_out[:, stack_full.t_axis>=0]
        stack_full.t_axis = stack_full.t_axis[stack_full.t_axis>=0]
    elif lags_to_keep == 'acausal':
        stack_full.XCF_out = stack_full.XCF_out[:, stack_full.t_axis<=0]
        stack_full.t_axis = stack_full.t_axis[stack_full.t_axis<=0]
    elif lags_to_keep == 'both':
        pass
    

    if stack_norm:
        stack_full.norm()

    if coherence_enhancing:
        interpolator = RegularGridInterpolator((stack_full.x_axis, stack_full.t_axis), stack_full.XCF_out,
                                                bounds_error=False, fill_value=0)

        enhancer = Coherence_Enhancement(stack_full, interpolator, xwin, twin, list(slw_list), decim, n_processes)
        results = enhancer.calculate_enhanced_stack()
        sem = 2*np.array(sum(results))/len(slw_list)

        if stack_full.t_axis.shape[0]%decim==0:
            shift = stack_full.t_axis[::decim]
        else:
            shift = stack_full.t_axis[:-decim:decim]
        sem_interp = RegularGridInterpolator((stack_full.x_axis, shift), sem,
                                            bounds_error=False, fill_value=0)
        xx, yy = np.meshgrid(stack_full.x_axis, stack_full.t_axis, indexing='ij')
        sem = sem_interp((xx, yy))

        stack_full.XCF_out *= sem**2   #Est ce qu'on donne le choix de ne pas prendre le carré ? dans ce cas on define coherence_enhancing = 0,1,2

    stack_full.compute_disp_image(freqs=freqs, vels=vels)
    disp_full = stack_full.disp.fv_map
    
    
    fig, ax = plt.subplots(1, 2, figsize=(6.4*2, 6.4))
    pclip = np.percentile(np.abs(stack_full.XCF_out), 99)
    ax[0].pcolormesh(stack_full.t_axis, stack_full.x_axis, stack_full.XCF_out,
                    cmap='seismic', vmax=pclip, vmin=-pclip)

    ax[0].set_xlabel('Timelag [s]', fontsize='x-large')
    ax[0].set_ylabel('Inter-station distance [m]', fontsize='x-large')
    pclip = 99.
    vmax = np.percentile(np.abs(disp_full[(stack_full.disp.vels>=250) & (stack_full.disp.vels>=2000)]), pclip)*5 #[~np.isnan(disp)]
    vmin = np.percentile(np.abs(disp_full[(stack_full.disp.vels>=250) & (stack_full.disp.vels>=2000)]), 50)#100-pclip)
    ax[1].pcolormesh(stack_full.disp.freqs,
                stack_full.disp.vels,
                disp_full,
                cmap="jet",
                vmax=vmax,
                vmin=vmin)
    ax[1].set(xscale='log', xlim=(0.7, 20), yscale='log', ylim=(250, 2000))
    ax[1].set_xlabel('Frequency [Hz]', fontsize='x-large')
    ax[1].set_ylabel('Phase Velocity [m/s]', fontsize='x-large')
    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / 'FIG' / 'Disp' / stack_name / f'Disp_without_masking.png', dpi=300)
    plt.close()
    

    if disp_masking:
        final_mask = np.zeros((len(vels), len(freqs)))
    
        for k in range(int((max_cut_dist-min_cut_dist)//(step_factor*dx))):
    
            cut_dist = max_cut_dist - ((k+1)*step_factor*dx) + 1
            
            stack_cut = deepcopy(stack_full)
            stack_cut.XCF_out = stack_cut.XCF_out[np.abs(stack_cut.x_axis) <= cut_dist]
            stack_cut.x_axis = stack_cut.x_axis[np.abs(stack_cut.x_axis) <= cut_dist]

    
            stack_cut.compute_disp_image(freqs=freqs, vels=vels, apodis=True, mod=False)
            masking_disp = stack_cut.disp.fv_map
            
            
            masking_disp[np.isnan(masking_disp)] = 0
            masking_disp[masking_disp<0] = 0
            final_mask += masking_disp
            
            
            fig, ax = plt.subplots(1, 2, figsize=(6.4*2, 6.4))
            pclip = np.percentile(np.abs(stack_cut.XCF_out), 99)
            ax[0].pcolormesh(stack_cut.t_axis, stack_cut.x_axis, stack_cut.XCF_out,
                            cmap='seismic', vmax=pclip, vmin=-pclip)
            
            ax[0].set_xlabel('Timelag [s]', fontsize='x-large')
            ax[0].set_ylabel('Inter-station distance [m]', fontsize='x-large')
            pclip = 99.
            vmax = np.percentile(np.abs(masking_disp[(stack_cut.disp.vels>=250) & (stack_cut.disp.vels>=2000)]), pclip)*3 #[~np.isnan(disp)]
            vmin = np.percentile(np.abs(masking_disp[(stack_cut.disp.vels>=250) & (stack_cut.disp.vels>=2000)]), 50)#100-pclip)
            ax[1].pcolormesh(stack_cut.disp.freqs,
                        stack_cut.disp.vels,
                        masking_disp,
                        cmap="jet",
                        vmax=vmax,
                        vmin=vmin)
            ax[1].set(xscale='log', xlim=(0.7, 20), yscale='log', ylim=(250, 2000))
            ax[1].set_xlabel('Frequency [Hz]', fontsize='x-large')
            ax[1].set_ylabel('Phase Velocity [m/s]', fontsize='x-large')
            plt.tight_layout()
            plt.savefig(PROCESSED_DIR / 'FIG' / 'Disp' / stack_name / f'Masking_disp_{k}_cut-dist={round(cut_dist, 1)}.png', dpi=300)
            plt.close()
    
    
        final_mask /= np.max(final_mask)
    
    
        pclip = 99.
        vmax = np.percentile(np.abs(final_mask), pclip)*1 #[~np.isnan(disp)]
        vmin = np.percentile(np.abs(final_mask), 50)#100-pclip)
        plt.pcolormesh(freqs, vels, final_mask, cmap="jet", vmax=vmax, vmin=vmin)
        plt.xscale('log')
        plt.xlim((0.7, 20))
        plt.yscale('log')
        plt.ylim((150, 2000))
        plt.savefig(PROCESSED_DIR / 'FIG' / 'Disp' / stack_name / f'Final_mask.png', dpi=300)
        plt.close()


        disp_full *= final_mask
    
    
        fig, ax = plt.subplots(1, 2, figsize=(6.4*2, 6.4))
        pclip = np.percentile(np.abs(stack_full.XCF_out), 99)
        ax[0].pcolormesh(stack_full.t_axis, stack_full.x_axis, stack_full.XCF_out,
                        cmap='seismic', vmax=pclip, vmin=-pclip)
    
        ax[0].set_xlabel('Timelag [s]', fontsize='x-large')
        ax[0].set_ylabel('Inter-station distance [m]', fontsize='x-large')
        pclip = 95
        vmax = np.percentile(np.abs(disp_full[(stack_full.disp.vels>=250) & (stack_full.disp.vels>=2000)]), pclip)#*5 #[~np.isnan(disp)]
        vmin = np.percentile(np.abs(disp_full[(stack_full.disp.vels>=250) & (stack_full.disp.vels>=2000)]), 50)#100-pclip)
        ax[1].pcolormesh(stack_full.disp.freqs,
                    stack_full.disp.vels,
                    disp_full,
                    cmap="jet",
                    vmax=vmax,
                    vmin=vmin)
        ax[1].set(xscale='log', xlim=(0.7, 20), yscale='log', ylim=(250, 2000))
        ax[1].set_xlabel('Frequency [Hz]', fontsize='x-large')
        ax[1].set_ylabel('Phase Velocity [m/s]', fontsize='x-large')
        plt.tight_layout()
        plt.savefig(PROCESSED_DIR / 'FIG' / 'Disp' / stack_name / f'Masked_disp.png', dpi=300)
        plt.close()


    if not os.path.exists(PROCESSED_DIR / 'DISPs'):
        os.mkdir(PROCESSED_DIR / 'DISPs')
    savedir = PROCESSED_DIR / 'DISPs'
    disp_name = 'Disp_' + stack_name
    if disp_masking:
        np.savez(savedir / disp_name, parameters=parameters, disp=disp_full, vels=vels, freqs=freqs, mask=final_mask, allow_pickle=True)
    else:
        np.savez(savedir / disp_name, parameters=parameters, disp=disp_full, vels=vels, freqs=freqs, allow_pickle=True)
