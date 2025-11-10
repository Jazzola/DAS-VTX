import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey
import os
import copy
from random import shuffle

from VSG_class import VirtualShotGather
from config import PROCESSED_DIR
from config import xcorr_start_date, xcorr_end_date, xcorr_sections, n_vsg_per_stack
from config import freq_lo, freq_hi, wlen_sw, taper, sw_bp_filt, sw_whiten
from config import xcorr_parameters, include_other_side
from utils import whiten_signals, calculate_SNR, generate_date_range, get_file_section


#il faut r��crire la corr�lation pour avoir des fails safe lorsque NaN
#notamment lorsqu'on a pas de data � corr�ler � cause de la traj

def xcorr_process(section, start_date, end_date):
    
    stack_list = []
    start_ch, pivot_ch, end_ch = section

    if not os.path.exists(PROCESSED_DIR / 'VSGs'):
            os.mkdir(PROCESSED_DIR / 'VSGs')
            os.mkdir(PROCESSED_DIR / 'VSGs' / 'Individuals')
    
    date_range = generate_date_range(start_date, end_date)

    dayly_dir_list = []
    for date in date_range:
        if os.path.exists(PROCESSED_DIR/'detects' / date):
            dayly_dir_list.append(PROCESSED_DIR/'detects' / date)
        else:
            print('No detections on:', date)
    dayly_dir_list.sort()

    detect_list = []
    for output_day in dayly_dir_list:
        
        if not os.path.exists(PROCESSED_DIR / 'VSGs' / 'Individuals' / output_day.name):
            os.mkdir(PROCESSED_DIR / 'VSGs' / 'Individuals' / output_day.name)
                
        for detection_hour in os.scandir(output_day):
            for detect in os.scandir(detection_hour.path):

                file_section = get_file_section(detect)
                if start_ch>= file_section[0] and end_ch <= file_section[1]:
                    detect_list.append(detect.path)

                    output_path = PROCESSED_DIR / 'VSGs' / 'Individuals' / output_day.stem / detection_hour.name
                    if not os.path.exists(output_path):
                        os.mkdir(output_path)
    

    stack_files_list = []
    if n_vsg_per_stack is None:
        shuffle(detect_list)

    else:
        detect_list.sort()
        
    for parameters in xcorr_parameters:
        wlen = parameters.get('wlen', 1)
        overlap = parameters.get('overlap', 0.8)
        delta_t = parameters.get('delta_t', 0.5)
        time_window_to_xcorr = parameters.get('time_window_to_xcorr', 5)
        norm = parameters.get('norm', True)
        norm_amp = parameters.get('norm_amp', True)
        #Nom associé aux paramètres
        parameters_str = f'st:{start_date}-end:{end_date}_o={overlap};dt={delta_t};w={wlen};twin={time_window_to_xcorr}'

        n_vsg = 0
        subset = 0
        stack = []
        SNR_values = []
        for detect in detect_list:
                    
            sw = np.load(detect, allow_pickle=True).item()


            for i in range(sw.data.shape[0]):
                #Taper for filter
                sw.data[i, :] = sw.data[i, :]*tukey(sw.data.shape[1],
                                                    taper/wlen_sw)
                
            dt = sw.t_axis[1]-sw.t_axis[0]
            if sw_bp_filt:
                sos = butter(5, [freq_lo, freq_hi], btype='bandpass', output='sos', fs=1/dt)
                data_filt = sosfiltfilt(sos, sw.data)
            
            if sw_whiten:
                data_filt = whiten_signals(data_filt, freq_lo, freq_hi, fs=1/dt)
            else:
                data_filt = sw.data
            
            sw_filt = copy.deepcopy(sw)
            sw_filt.data = data_filt

            try:
                vsg=VirtualShotGather(sw_filt, start_x=start_ch,
                                        end_x=end_ch, pivot=pivot_ch,
                                        wlen=wlen, overlap=overlap, delta_t=delta_t,
                                        time_window_to_xcorr=time_window_to_xcorr,
                                        norm=norm, norm_amp=norm_amp,
                                        include_other_side=include_other_side,
                                        new_xcorr=True) #le rajouter dans config ?
            except ValueError:
                print('SW skipped, VSG calculation error') #ajouter des détails?
                continue

            if np.isnan(vsg.XCF_out).any():
                print('SW skipped, NaNs in VSG, SW window boundary issue ??') #ajouter des détails?
                continue
            
            #Est ce qu'on enregistre les individuals ???????????
            # si on ne le fait pas ce n'est pas la peine de créer le repertoire
            
            if not stack:
                stack.append(vsg)
            else:
                stack[0]+=vsg

            SNR = calculate_SNR(stack[0])
            SNR_values.append(SNR)


            n_vsg += 1
            if n_vsg == n_vsg_per_stack: #fixed size stacks for studying time variations
                stack = stack[0]
                subset += 1
                if not os.path.exists(PROCESSED_DIR / 'VSGs' / 'STACKs'):
                    os.mkdir(PROCESSED_DIR / 'VSGs' / 'STACKs')
                out_name = f'Stack_n={n_vsg}_Subset={subset}_' + parameters_str
                np.savez(PROCESSED_DIR / 'VSGs' / 'STACKs' / out_name, parameters=parameters, SNR=SNR_values, stack=stack, allow_pickle=True)
                stack_files_list.append(str(PROCESSED_DIR / 'VSGs' / 'STACKs' / out_name)+'.npz')
                n_vsg = 0
                stack = []
                SNR_values = []
        
        if n_vsg_per_stack is None: #full stack for studying parameters
            stack = stack[0]
            if not os.path.exists(PROCESSED_DIR / 'VSGs' / 'STACKs'):
                os.mkdir(PROCESSED_DIR / 'VSGs' / 'STACKs')
            out_name = f'Stack_n={n_vsg}_' + parameters_str
            np.savez(PROCESSED_DIR / 'VSGs' / 'STACKs' / out_name, parameters=parameters, SNR=SNR_values, stack=stack, allow_pickle=True)
            stack_files_list.append(str(PROCESSED_DIR / 'VSGs' / 'STACKs' / out_name)+'.npz')

    return stack_files_list
