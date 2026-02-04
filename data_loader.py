import os
from datetime import timedelta
from pathlib import Path

from func_data_imports import data_matrix_import
from config import datapath, decimateddatapath, DAS_FILE_FORMAT, DECIM_FILE_FORMAT, DATE_FORMAT, dx, length_sw, tracking_data_decimation_factor
from utils import generate_date_range, get_date_from_file_path, create_npz_data

class DataLoader:
    def __init__(self, section):
        self.data_dir = datapath
        self.decimated_data_dir = decimateddatapath
        self.pivot_channel = section[1]
        #on a fait la modif ici
    
    def scan_data(self, start_date: str, end_date: str, start_hour: int, end_hour: int):
        """Scan data files for the given date range."""
        print(f"Scanning data files from {start_date} to {end_date}...")
        date_range = generate_date_range(start_date, end_date)

        self.file_paths = []
        for date in date_range:
            file_day_dir = Path(self.data_dir) / date
            if not file_day_dir.exists():
                print(f"Repertory not found for date {date} and path {file_day_dir}")
                continue
                
            for file in os.scandir(file_day_dir):
                if file.is_file():
                    fp = file_day_dir / file.name
                    file_date = get_date_from_file_path(fp, DAS_FILE_FORMAT)
                    if file_date.hour >= start_hour and file_date.hour < end_hour:
                        self.file_paths.append(fp)
            
        self.file_paths.sort()
        print(f'Found {len(self.file_paths)} data files between {start_hour}h and {end_hour}h.')

        self.nfiles = len(self.file_paths)

    def get_next_data(self):
        fp=self.file_paths.pop(0)
        file_date = get_date_from_file_path(fp, DAS_FILE_FORMAT)
        
        decimated_day_rep = Path(self.decimated_data_dir) / file_date.strftime(DATE_FORMAT)
        
        corresponding_decimated_files = []
        
        if decimated_day_rep.exists():
            corresponding_decimated_files = [f.path for f in os.scandir(decimated_day_rep) 
                                             if get_date_from_file_path(f, DECIM_FILE_FORMAT) >= file_date
                                             and get_date_from_file_path(f, DECIM_FILE_FORMAT) < file_date + timedelta(hours=1)]
            corresponding_decimated_files.sort()
        else:
            os.mkdir(decimated_day_rep)
            
        if len(corresponding_decimated_files) > 0: #si il n'y a qu'un seul file on peut avoir un soucis...
            print('Using existing decimated data')
            StrainRate, distance, time, data_attributes = data_matrix_import(fp, ch1=self.pivot_channel-length_sw//2, ch2=self.pivot_channel+length_sw//2)

        else:
            print('No decimated files')
            print('Creating decimated data...')
            create_npz_data(fp, decimated_day_rep, decimation_factor=tracking_data_decimation_factor, return_full=False)
            
            StrainRate, distance, time, data_attributes = data_matrix_import(fp, ch1=self.pivot_channel-length_sw//2, ch2=self.pivot_channel+length_sw//2)
            corresponding_decimated_files = [f.path for f in os.scandir(decimated_day_rep) 
                                             if get_date_from_file_path(f, DECIM_FILE_FORMAT) >= file_date
                                             and get_date_from_file_path(f, DECIM_FILE_FORMAT) < file_date + timedelta(hours=1)]
            corresponding_decimated_files.sort()
        return StrainRate, distance, time, data_attributes, fp, corresponding_decimated_files
