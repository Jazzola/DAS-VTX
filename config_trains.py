from pathlib import Path
import numpy as np

# ============================================================
#                    MAIN EXECUTION FLAGS
# ============================================================
# Control which parts of the DAS-VTX workflow are executed.
# This allows running detection, correlation, or interpretation
# independently without recomputing intermediate products.
RUN_TRACKING = True          # Detect and track moving sources (trains)
RUN_XCORR = False            # Compute Virtual Shot Gathers (VSGs)
RUN_INTERPRETATION = False   # Dispersion analysis and interpretation

n_processes = 15             # Number of parallel processes (CPU cores)


# ============================================================
#                TRACKING CONFIGURATION (if RUN_TRACKING)
# ============================================================
# Definition of fiber sections used for detecting and tracking
# moving sources. Each tuple defines (start, pivot, end) offsets
# along the cable in meters.
tracking_sections = [(15200, 15550, 15900)]

# Time span over which train passages are processed
tracking_start_date = '20250401'
tracking_end_date   = '20250630'

# Daily UTC time window to restrict processing to periods
# with significant train traffic
start_hour = 2     # inclusive
end_hour   = 22    # exclusive

# Decimation factor applied to raw DAS data before tracking
# (1 = no decimation, higher values reduce data volume)
tracking_data_decimation_factor = 1


# ============================================================
#              CROSS-CORRELATION (if RUN_XCORR)
# ============================================================
# Defines the spatial section used to compute cross-correlations
# for VSG retrieval.
xcorr_sections = [(9400, 9775, 10150)]

# Time span over which possible detections are processed
xcorr_start_date = '20230404'
xcorr_end_date   = '20230404'

# VSG stacking options
randomize_vsg = True         # Randomize VSG selection to avoid bias
n_vsg_per_stack = None       # Fixed stack size (None = use all available)
stack_files_list = None      # Optional: use precomputed stack files (to run correlation alone)


# ============================================================
#                   DAS ACQUISITION PARAMETERS
# ============================================================
dx = 9.6  # Inter-channel spacing [m], used to convert channel index to offset


# ============================================================
#                       PATHS AND I/O
# ============================================================
# Input DAS data directories
datapath = '//' #str: path to the input DAS data files
# Output directory path
decimateddatapath = ('//DECIMATE') #str: path to the files created from the raw DAS datasets after decimation
PROCESSED_DIR = Path('//RESULTS/outputs') #str: path where are stored the outputs of the script
 

# ============================================================
#                     FILE NAMING FORMATS
# ============================================================
DATE_FORMAT = '%Y%m%d'
DATE_TIME_FORMAT = '%Y%m%d_%H%M%S'
DAS_FILE_FORMAT = 'SR_DS_%Y-%m-%d_%H-%M-%S_UTC.h5'
DECIM_FILE_FORMAT = '%Y%m%d_%H%M%S.npz'


# ============================================================
#            PREPROCESSING PRIOR TO TRACKING
# ============================================================
# This preprocessing enhances coherent surface-wave energy
# while suppressing incoherent noise before detection and tracking.
default_preprocessing_dict = {
    'smoothing': (21, 15),        # (time, space) smoothing window sizes
    'x_inter': (),                # Optional spatial interpolation (disabled)
    'FK': {},                     # FK-domain filtering (disabled for trains)
    'BP': {
        'freq_lo': 0.2,           # Lower corner frequency [Hz]
        'freq_hi': 1.0            # Upper corner frequency [Hz]
    },
    'SQRT': True,                 # Square-root amplitude scaling
    'spatial_av_vel': 70 / 3.6,   # Reference velocity [m/s] for alignment
    'av_win': 5,                  # Spatial averaging window (channels)
    'oversampling_factor': 5      # Temporal oversampling factor
}

preprocessing_dict = default_preprocessing_dict


# ============================================================
#                  DETECTION AND TRACKING
# ============================================================
# Parameters controlling peak detection in the DAS amplitude
# used to identify moving sources.
tracking_args = {
    "detect": {
        "minprominence": 0.3,     # Minimum peak prominence
        "minseparation": 1,       # Minimum separation between peaks
        "prominenceWindow": 600,  # Window for prominence estimation
    }
}
# Kalman filter parameters for trajectory estimation
sigma_a = 0.1    # Process noise (controls smoothness of trajectory)
R = 10           # Measurement noise covariance
reverse_amp = True
nx_init = 3      # Number of detections needed to initialize a track


# ============================================================
#                TRAJECTORY PRESELECTION
# ============================================================
# Criteria used to reject spurious or non-physical trajectories.
preselection_dict = {
    'max_adjacent_nan': 50,               # Max allowed consecutive gaps
    'max_total_nan': 0.2,                 # Max fraction of missing data
    'average_speed': (40/3.6, 100/3.6),   # Accepted speed range [m/s]
    'curve_break': (
        5, 1.8, 0.1, 25                   # Parameters for curvature checks
    ),
    'speed_fluctuations': (1.5, 0.1)      # Limits on speed variability
}


# ============================================================
#           SURFACE-WAVE WINDOW EXTRACTION
# ============================================================
# Definition of the spatio-temporal windows extracted around
# detected train passages.
taper = 4                 # Taper length [s] before filtering
wlen_sw = 200              # Window length [s]
wlen_sw += taper
length_sw = 2800           # Spatial aperture [m]
temporal_spacing = (
    wlen_sw - taper        # Minimum spacing between windows
)


# ============================================================
#            CROSS-CORRELATION / VSG PARAMETERS
# ============================================================
xcorr_parameters = [{
    'wlen': 5.8,                   # Correlation window length [s]
    'overlap': 0.0,                # Overlap between windows
    'delta_t': 0.5,                # At a pivot channel, time shift between the correlation window and the estimated trajectory [s]
    'time_window_to_xcorr': 6.5,   # Lag window length [s]
    'norm': True,                  # Temporal normalization
    'norm_amp': True               # Amplitude normalization
}]

include_other_side = True          # Use both sides of the pivot
sw_bp_filt = True                  # Band-pass filter SW windows
sw_whiten = True                   # Spectral whitening
freq_lo = 0.5                      # Lower SW frequency [Hz]
freq_hi = 40.0                     # Upper SW frequency [Hz]


# ============================================================
#       INTERPRETATION / DISPERSION ANALYSIS PARAMETERS
# ============================================================
# Parameters controlling dispersion imaging and enhancement.
coherence_enhancing = True         # Apply coherence-based enhancement
slw_list = np.linspace(1/250, 1/1200, 16)  # Slowness grid [s/m]
twin = int(500 * 0.1)              # Temporal smoothing window
xwin = int(100 / dx)               # Spatial smoothing window (channels)
decim = 20                         # Decimation for dispersion analysis

# Frequency and velocity grids for dispersion spectra
freqs = np.linspace(0.7, 25, int((25 - 0.7) * xcorr_parameters[0]['wlen'] * 2))
vels = np.linspace(2 * dx, 2100, int(xcorr_parameters[0]['wlen'] * 35 * 3))

stack_norm = False                 # Normalize stacked VSGs
offsets_to_keep = 'both'           # Use causal + acausal offsets
lags_to_keep = 'causal'            # Restrict to causal lags


# ============================================================
#             DISPERSION MASKING / APERTURE CONTROL
# ============================================================
# Recursive masking used to stabilize dispersion estimates
# by progressively reducing VSG aperture.
disp_masking = True
max_cut_dist = 500                 # Initial max offset [m]
min_cut_dist = 150                 # Minimum retained offset [m]
step_factor = 4                    # Step size in multiples of dx
