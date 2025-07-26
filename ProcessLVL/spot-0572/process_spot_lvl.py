#!/usr/bin/python -u
# Author: Xavier Chartrand
# Email : x.chartrand@protonmail.me
#         xavier.chartrand@ec.gc.ca

'''
Copy and parse Spotter data 'lvl0', 'lvl1' and 'lvl2' to NetCDF.

Directional moments (a1,b1,a2,b2) are computed from raw 3D displacement datas.

Directional spectra are computed for the specified date, hour and minutes,
usin a weigthed Fourier series.

A Butterworth filter may be applied to filter low or high frequencies, or
equivalently, low or high scales using the linear dispersion relation for
surface waves to associate an equivalent frequency.
'''

# Custom utilities
from spot_utils import *

## BEGIN STREAM EDITOR
# Information about the buoy
# buoy:         Buoy station to process (spot-[0572,1082])
# year:         Year to process
# H:            Water depth underneath the platform

# Corrected date for good values
# cbd:          Begin date (timestamp) for which values start to be good
# ced:          End date (timestmap) for which values end to be good

# 'filt_p'
# filt_bool:    Flag to apply ButterWorth filter
# filt_type:    ButterWorth filter type (lowpass 'lp' or highpass 'hp')
# filt_data:    Cutoff type (frequency 'freq' or length 'length')
# C0:           Cutoff parameter depending on 'filt_data'

# Data files and directories
# rdata_dir:    Raw data directory
# odata_dir:    OGSL data directory
# lvl0_dir:     'lvl0' data directory
# lvl1_dir:     'lvl1' data directory
# lvl2_dir:     'lvl2' data directory
# rdata_file:   Raw data file
# odata_file:   OGSL data file
# lvl0_file:    'lvl0' file to write
# lvl1_file:    'lvl1' file to write
# lvl2_file:    'lvl2' file to write
# ---------- #
buoy       = 'spot-0572'
year       = 2023
H          = 4

cbd        = '2023-06-06T00:00:00'
ced        = '2023-06-09T21:30:00'

filt_bool  = True
filt_type  = 'hp'
filt_data  = 'length'
C0         = 2.E2

rdata_dir  = '../../RawData/'
odata_dir  = '../../OGSL/statistics/'
lvl0_dir   = '../../lvl0/'
lvl1_dir   = '../../lvl1/'
lvl2_dir   = '../../lvl2/'
rdata_file = 'D.txt'
ldata_file = 'L.txt'
odata_file = 'iml-4_statistics.csv'
lvl0_file  = 'lvl0_displacement.nc'
lvl1_file  = ['lvl1_spectralvariables.nc','lvl1_wave.nc']
# ---------- #
## END STREAM EDITOR

## MAIN
# Physical parameters
rho_0 = 1000                                    # reference density [kg/m3]
g     = 9.81                                    # gravity acceleration [m/s2]

# Wave monitor parameter
fs       = 2.5                                  # sampling frequency
afac     = 3*[1]                                # amplitude factor
freq_min = 0.03                                 # minimal frequency resolved
freq_max = 0.8                                  # maximal frequency resolved
xpos     = 0.0                                  # x position
ypos     = 0.0                                  # y position
zpos     = 0.0                                  # z position

# Compute frequency cutoff
fcut = getFrequency(2*pi/C0,H)

# Magnetic declination relative to true north
magdec = 0

# X-Y-Z and Longitude-Latitude index
xyz_cartesian_index = [-3,-2,-1]

# Update raw data directory with buoy and year
rdata_dir = '%s/%s/%d/'%(rdata_dir,buoy,year)

# Update files with directories, buoy and year
file_fmt     = '%s%s_%s_%d.nc'
buoy_str     = buoy.replace('-','')
rdata_file   = os.popen('ls %s*/*/*%s'%(rdata_dir,rdata_file)).read()\
                       .rstrip('\n').split('\n')
ldata_file   = os.popen('ls %s*/*/*%s'%(rdata_dir,ldata_file)).read()\
                       .rstrip('\n').split('\n')
lvl0_file    = file_fmt%(lvl0_dir,buoy_str,lvl0_file.split('.nc')[0],year)
lvl1_file[0] = file_fmt%(lvl1_dir,buoy_str,lvl1_file[0].split('.nc')[0],year)
lvl1_file[1] = file_fmt%(lvl1_dir,buoy_str,lvl1_file[1].split('.nc')[0],year)

# Create output directories if inexistant
[sh('mkdir -p %s 2>/dev/null'%d) for d in [lvl0_dir,lvl1_dir,lvl2_dir]]

## OGSL STATISTICS
# Initialize outputs
bwp_rmin,bwp_rmax,bwp_std = [[] for _ in range(3)]

# Selection and tickers
bwp_sel     = ['Wave Significant Height',
               'Wave Period',
               'Wave Period',
               'Wave Period',
               'Wave Mean Direction',
               'Wave Mean Direction',
               'Wave Mean Spreading',
               'Wave Mean Spreading']
bwp_tickers = ['Hm0',
               'Tm-10',
               'Tm01',
               'Tm02',
               'Theta_Mean',
               'Theta_Peak',
               'Sigma_Mean',
               'Sigma_Peak']
bwp_eps     = [0.02,1/2/fs,1/2/fs,1/2/fs,1,1,1,1]

# Read and append data
DSogsl = pd.read_csv(odata_dir+odata_file,
                     delimiter=',',
                     skipinitialspace=True,
                     skiprows=3)
bwp    = np.array([p.rstrip(' ') for p in DSogsl['Parameter']])
for key in bwp_sel:
    min_var = DSogsl['min'][np.where(bwp==key)[0][0]]
    std_var = DSogsl['std'][np.where(bwp==key)[0][0]]
    if key=='Wave Period':
        n_std   = 2
        max_var = max(DSogsl['max'][np.where(bwp==key)[0][0]],1/fcut)
    else:
        n_std   = 1
        max_var = DSogsl['max'][np.where(bwp==key)[0][0]]
    bwp_rmin.append(min_var)
    bwp_rmax.append(max_var)
    bwp_std.append(n_std*std_var)

# LEVEL GLOBAL PARAMETERS
lvl_d = {'Info':{'Id':buoy,
                 'Corrected_Date_Begin':cbd,
                 'Corrected_Date_End':ced,
                 'Magnetic_Declination':magdec,
                 'Sampling_Frequency':fs,
                 'Wave_Record_Length':30*60,
                 'Aux_Record_Length':30*60,
                 'Wave_Regular_Length':30*60,
                 'Aux_Regular_Length':30*60},
         'Input':{'Loc_File_List':ldata_file,
                  'Raw_File_List':rdata_file,
                  'Raw_Header_Rows':7},
         'Output':{'LVL0_File':lvl0_file,
                   'LVL1_File':lvl1_file},
         'Physics_Parameters':{'Ref_Density':rho_0,
                               'Gravity':g,
                               'Water_Depth':H},
         'Wave_Monitor':{'Amplitude_Factor':afac,
                         'Freq_Min':freq_min,
                         'Freq_Max':freq_max,
                         'X_Position':xpos,
                         'Y_Position':ypos,
                         'Z_Position':zpos,
                         'XYZ_Cartesian_Index':xyz_cartesian_index},
         'Filtering':{'Filter':filt_bool,
                      'F_Type':filt_type,
                      'D_Type':filt_data,
                      'C0':C0,
                      'H':H}}

## QUALITY FLAG PARAMETERS
testinit   = []
test_order = ['18','14','15','20','19','16','17','13','12','10','9','11']
l2v_order  = ['hm0','tmn10','tm01','tm02','fp','tm','tp','sm','sp']

# Short-term
qfst_d = {'Test_9':{'Do_Test':True,
                    'N':3,
                    'QF':testinit,
                    'Type':'hv',
                    'Update_Data':True},
          'Test_10':{'Do_Test':True,
                     'N':5,
                     'm':3,
                     'p':3,
                     'thrs':0.01,
                     'QF':testinit,
                     'Type':'hv',
                     'Update_Data':True},
          'Test_11':{'Do_Test':True,
                     'imin':-50,
                     'imax':50,
                     'lmin':-50,
                     'lmax':50,
                     'QF':testinit,
                     'Type':'hv',
                     'Update_Data':True},
          'Test_12':{'Do_Test':True,
                     'm':int(np.ceil(2*fs/fcut)),
                     'delta':0.1,
                     'QF':testinit,
                     'Type':'hv',
                     'Update_Data':False},
          'Test_13':{'Do_Test':False,
                     'N':np.nan,
                     'QF':testinit,
                     'Type':'hv',
                     'Update_Data':False},
          'Test_Order':test_order}

# Long-term
qflt_d = {'Test_14':{'Do_Test':False,
                     'wnum':'to_update',
                     'freq':'to_update',
                     'H':H,
                     'fv':'to_update',
                     'bw':0.1,
                     'QF':testinit},
          'Test_15':{'Do_Test':True,
                     'N':5,
                     'QF':testinit},
          'Test_16':{'Do_Test':True,
                     'Ns':3,
                     'Nf':5,
                     'eps':bwp_eps,
                     'QF':testinit},
          'Test_17':{'Do_Test':False,
                     'freq':'to_update',
                     'csd_dep':'to_update',
                     'imin':freq_min,
                     'imax':freq_max,
                     'lmin':freq_min,
                     'lmax':freq_max,
                     'eps':1.E-8,
                     'QF':testinit},
          'Test_18':{'Do_Test':False,
                     'QF':testinit},
          'Test_19':{'Do_Test':True,
                     'rmin':bwp_rmin,
                     'rmax':bwp_rmax,
                     'set_flag':0,
                     'prev_qf':[],
                     'QF':testinit},
          'Test_20':{'Do_Test':True,
                     'rmin':bwp_rmin,
                     'rmax':bwp_rmax,
                     'eps':bwp_std,
                     'QF':testinit},
          'Tickers_Order':bwp_tickers,
          'LVL2_Vars_Order':l2v_order,
          'Test_Order':test_order}

## OUTPUTS
# Write 'lvl0' displacement
writeLvl0(lvl_d,qfst_d)

# Write 'lvl1' spectral variables and bulk wave parameters
writeLvl1(lvl_d,qflt_d)

# END
