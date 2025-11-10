from pathlib import Path
import numpy as np


#=======================#
### MAIN PARAMETERS ###
#=======================#
RUN_TRACKING= False
RUN_XCORR= True
RUN_INTERPRETATION= True
n_processes = 7


#=======================#
### TRACKING PROCESS ###
#=======================#
tracking_sections = [(1750, 2100, 2450)] #(start, pivot, end)
tracking_start_date = '20230404'
tracking_end_date = '20230404'
start_hour = 5 # UTC >= 
end_hour = 12 # <
# Defines the time of the day to process, for entire day set start_hour=0 and end_hour=24


#=======================#
### XCORR PROCESS ###
#=======================#
xcorr_sections = [(1750, 1950, 2150)] #(start, pivot, end)
xcorr_start_date='20230404'
xcorr_end_date='20230404'
# included date limits

n_vsg_per_stack = None #fixed size stacks for studying time variations

#list of stack files paths 
#OR string of stack files directory path
#OR None if followup of xcorr process
stack_files_list = None #['/home/johannes/THOMAS/0versions_wr/Updated_xcorr_test_ws/workflow_fig/stack_raw_New_500-rand_SNRnew_o_0, dt_0.2, w_2.3, twin_2.6.npz'] #


#=================#
### DATA ###
#=================#
dx=9.6


#=================#
### DIRECTORIES ###
#=================#
datapath = '/home/johannes/DATA/LSDF_DAS/2023_CN_DAS'
decimateddatapath = '/home/johannes/DATA/LSDF_DAS/2023_CN_DAS/DECIMATE'
#Output directory path
PROCESSED_DIR = Path('/home/johannes/THOMAS/DAS-VTX_WIP/out_test')



#=================#
### FORMATS ###
#=================#
DATE_FORMAT = '%Y%m%d'
DATE_TIME_FORMAT = '%Y%m%d_%H%M%S'
DAS_FILE_FORMAT = 'SR_DS_%Y-%m-%d_%H-%M-%S_UTC.h5'
DECIM_FILE_FORMAT = '%Y%m%d_%H%M%S.npz'


#=======================#
### PREPROCESSING ###
#=======================#
default_preprocessing_dict = {'smoothing':(21,15), #21,15
                              'x_inter': (), #240, 25
                              'FK':{}, #'slope_hi':3.6/30,'slope_lo':3.6/80 # 20-70?
                              'BP':{'freq_hi':1, #1
                                    'freq_lo':0.2}, #0.2
                              'SQRT':True, #True
                              'spatial_av_vel': 50/3.6, #velocity in m/s used to shift data before spatial averaging, put False if not used
                              'av_win': 5, #spatial averaging window size in number of data points
                              'oversampling_factor': 5 #preprocessed data oversamling before tracking
                              }
preprocessing_dict = default_preprocessing_dict


#==============#
### TRACKING ###
#==============#
tracking_args = {
    "detect":{
            "minprominence": 0.3, #0.3, 
            "minseparation": 1, #1, 
            "prominenceWindow": 600, 
            }
}
sigma_a = 0.1 #0.1 
R = 5 #10 #5
reverse_amp=True
nx_init = 3 #3

#Preselection parameters, use False in the dictionnary to deactivate a criteria
preselection_dict = {'max_adjacent_nan': 50,#number of adjacent nan
                     'max_total_nan': 0.2, #portion of nan
                     'average_speed': (30/3.6, 80/3.6), # (minspeed, maxspeed) in m/s #40-100
                     'curve_break': (5, 1.8, 0.1, 25), # (slinding mean window size, factor greater than percentile, max portion of points greater, specific quantile)
                     'speed_fluctuations': (1.5, 0.1) # (factor greater than average_speed, max portion of points greater)
                     }

#SW window output size
taper = 4  #Used to apodise before BP
wlen_sw = 80 #250  #seconds
wlen_sw += taper
length_sw= 740 #2800 #meters
temporal_spacing = wlen_sw - taper #minimum spacing between detection windows
#spatial_ratio ?


#===========#
### XCORR ###
#===========#
xcorr_parameters = [{'wlen':2.3,
                    'overlap':0.8,
                    'delta_t':0.5,
                    'time_window_to_xcorr':5,
                    'norm':True,
                    'norm_amp':True}] #list of parameters dictionnaries if multiple stacks are to be calculated
#est ce qu'on retire cette possibilité au final ??
include_other_side = True
sw_bp_filt = True
sw_whiten = True
freq_lo = 0.5
freq_hi = 40

#MODIFIER LA CORREL POUR GERER LE WARNING: invalid value lorsqu'on normalise le VSG ligne 217


#==============#
### INTERPRETATION ###
#==============#
#coherence enhancement parameters
coherence_enhancing = True
slw_list = np.linspace(1/250, 1/1200, 16)
#slw_list = 1/slw_list
twin = int(500*0.1)
xwin = int(100/9.6)
decim = 20


freqs = np.linspace(0.7, 25, int((25-0.7)*2.3*2)) #*xcorr_parameters['wlen'] on peut pas le fetch car on a implémenté une liste de combinaison de parametres....
vels = np.linspace(2*9.6, 2100, int(2.3*35*3))#1/np.linspace(1/(2*9.6), 1/2100, int(wlen*35*3))


stack_norm = False
offsets_to_keep = 'both'
lags_to_keep = 'causal'


disp_masking = False
max_cut_dist = 500 #np.max(np.abs(stack_full.x_axis))
min_cut_dist = 150
step_factor = 4 #Used to get a multiple of dx as cutting distance step

