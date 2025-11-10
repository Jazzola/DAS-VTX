import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import shutil

config_to_use = 'voiture'
if config_to_use == 'voiture' or config_to_use == 'train':
    shutil.copy(os.getcwd()+'/config_'+config_to_use+'.py', os.getcwd()+'/config.py')
    
from data_loader import DataLoader
from tracking_process import tracking_process
from xcorr_process import xcorr_process
from interpretation_process import interpretation_process
from config import PROCESSED_DIR
from config import RUN_TRACKING, RUN_XCORR, RUN_INTERPRETATION, n_processes
from config import tracking_start_date, tracking_end_date, tracking_sections, start_hour, end_hour
from config import xcorr_start_date, xcorr_end_date, xcorr_sections
from config import stack_files_list
from utils import multiprocess_iterable_on_dates, multiprocess_iterable_on_sections

import logging
logger = logging.getLogger('todefine')


def run_tracking_analysis(args):
    section, start_date, end_date = args
    loader = DataLoader(section)
    loader.scan_data(start_date, end_date, start_hour, end_hour)
    tracking_process(loader, section)
    
def run_xcorr_analysis(args):
    section, start_date, end_date = args
    return xcorr_process(section, start_date, end_date)


def main():
    logging.basicConfig(filename='test.log', level=logging.INFO)
    logger.info('Started')


    if RUN_TRACKING:
        if not os.path.exists(PROCESSED_DIR / 'detects'):
            os.mkdir(PROCESSED_DIR / 'detects')
        for section in tracking_sections:
            args = multiprocess_iterable_on_dates(tracking_start_date,
                                                tracking_end_date,
                                                n_processes,
                                                section)

            with Pool(n_processes) as p:
                p.map(run_tracking_analysis, args)
    
    
    if RUN_XCORR:

        args = multiprocess_iterable_on_sections(xcorr_start_date,
                                            xcorr_end_date,
                                            xcorr_sections)

        with Pool(n_processes) as p:
            sfl = p.map(run_xcorr_analysis, args)
            sfl = [item for sublist in sfl for item in sublist]


    if RUN_INTERPRETATION:
        if RUN_XCORR:
            for stack_file in sfl:
                interpretation_process(stack_file)
        elif stack_files_list is not None:
            if type(stack_files_list) == list:
                for stack_file in stack_files_list:
                    interpretation_process(stack_file)
            else:
                for stack_file in os.scandir(stack_files_list):
                    interpretation_process(stack_file.path)
        else:
            print('No stack files found, skipping interpretation')

            

if __name__=='__main__':
    main()