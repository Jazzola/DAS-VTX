import os
import numpy as np
import matplotlib.pyplot as plt

from Tracker_class import Tracker
from config import dx
from config import DATE_FORMAT, PROCESSED_DIR, DAS_FILE_FORMAT
from config import tracking_args
from config import wlen_sw, length_sw, temporal_spacing
from utils import get_date_from_file_path

import logging
logger = logging.getLogger('tests_trains')


def tracking_process(DataLoader, section):
    start_ch, pivot_ch, end_ch = section
    
    output_name = 'from_offset'+str(start_ch)+'to'+str(end_ch)+'_'

    for k in range(DataLoader.nfiles):

        try:
            data, distance, t_axis, data_attributes, file_path, decimated_files = DataLoader.get_next_data()
        except ValueError:
            #print('File skipped')
            logger.info('File skipped')
            continue
        
        date = get_date_from_file_path(file_path, DAS_FILE_FORMAT)
        #print('Processing file: '+ file_path.stem)
        logger.info('Processing file: '+ file_path.stem)
        
        day_str = date.strftime(DATE_FORMAT)
        if not os.path.exists(PROCESSED_DIR / 'detects' / day_str):
            os.mkdir(PROCESSED_DIR / 'detects' / day_str)

        detect_list = []
        for decimated_file in decimated_files:
            
            tracker = Tracker(decimated_file, dx)
            tracker.track_cars(start_ch, end_ch, tracking_args, show_plot=False)
            
            tracker.select_surface_wave_windows(data, distance, t_axis, pivot_ch,
                                                wlen_sw=wlen_sw, length_sw=length_sw,
                                                temporal_spacing=temporal_spacing)
            curt_veh_num = len(tracker.sw_selector)
            #print('Detections: ', len(tracker.sw_selector))
            logger.info(f'Detections: {len(tracker.sw_selector)}')
            if curt_veh_num == 0:
                continue

            
            for sw in tracker.sw_selector:
                #sw.save_fig()
                #plt.show()
                detect_list.append(sw)
            #faudra modifier les plots pour avoir les values dans la config

            
        if detect_list:
            if not os.path.exists(PROCESSED_DIR / 'detects' / day_str/ file_path.stem):
                os.mkdir(PROCESSED_DIR / 'detects' / day_str/ file_path.stem)
                
            for current_detect in detect_list:
                detect_name = output_name + 't_'+str(round(current_detect.t_axis[0]))
                np.save(PROCESSED_DIR / 'detects' / day_str/ file_path.stem/ detect_name,
                        current_detect,
                        allow_pickle=True)
        else:
            #print('No detections for the hour')
            logger.info('No detections for the hour')

    del data, distance, t_axis, data_attributes, file_path, decimated_files, tracker, detect_list