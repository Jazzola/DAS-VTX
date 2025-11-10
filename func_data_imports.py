import sys
import numpy as np
#import DASpy_A1_process as dasP
#import DASpy_A1_read as dasR
from obspy.core import UTCDateTime
from febus_optics_lib.reader import H5ReaderDas


def read_das_npz(fname, **kwargs):
    try:
        data_file = np.load(fname)
    except:
        raise Exception(f"fname: {fname}")
    data = data_file["data"]
    x_axis = data_file["x_axis"]

    t_axis = data_file["t_axis"]
    ch1 = kwargs.get('ch1', x_axis[0])
    ch2 = kwargs.get('ch2', x_axis[-1])

    ch1_idx = np.argmax(x_axis >= ch1)
    ch2_idx = np.argmax(x_axis >= ch2)
    data = data[ch1_idx:ch2_idx+1, :] #+1 pour bien prendre la ch2, est ce qu'on a la bonne taile ????
    x_axis = x_axis[ch1_idx:ch2_idx+1]

    return data, x_axis, t_axis


def data_matrix_import(filepath, spatial_sampling=9.6,
                       starttime=None, endtime=None,
                       maxdist=25000, mindist=40, #Values for Dataset of campus North
                       modePro = 'StrainRate',
                       **kwargs):
    
    ch1 = kwargs.get('ch1', mindist)
    ch2 = kwargs.get('ch2', maxdist)
    if ch1<mindist or ch2>maxdist:
        raise KeyError(f'ch1 or ch2 out of our boundaries: {mindist}, {maxdist}. \nPlease change those values.')
    
    instance = H5ReaderDas(filepath)
    first_zone = instance.list_zones[0] # extracting the zones name
    
    concat_results = instance.extract_concat(from_time=starttime,
                                             to_time=endtime,
                                             time_type="relative",
                                             from_dist=ch1,
                                             to_dist=ch2,
                                             dist_type="meter",
                                             zones=first_zone)
    
    StrainRate = concat_results[first_zone]['data']
    StrainRate = np.transpose(StrainRate)
    distance = concat_results[first_zone]['distance_vect']
    dt = instance.param_dict[first_zone]['derivation_time']; dt = dt/1000 #ms to s
    
    data_attributes = [instance.param_dict[first_zone]['spacing'][1],
                       instance.param_dict[first_zone]['spacing'][0],
                       instance.param_dict[first_zone]['gauge_length'],
                       instance.param_dict[first_zone]['derivation_time'],
                       instance.param_dict[first_zone]['pulse_width'],
                       instance.param_dict[first_zone]['sampling_res'],
                       instance.param_dict[first_zone]['pulse_rate_freq'],
                       UTCDateTime(concat_results[first_zone]['time_vect'][0])] #absolute starttime
    
    time = np.arange(StrainRate.shape[1])*dt
    print('H5 data attributes', data_attributes)
    print('Verify spatial sampling =', spatial_sampling)


    return StrainRate, distance, time, data_attributes
