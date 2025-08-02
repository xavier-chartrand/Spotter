#!/usr/bin/python -u
# Author: Xavier Chartrand
# Email : x.chartrand@protonmail.me
#         xavier.chartrand@ec.gc.ca
#         xavier.chartrand@uqar.ca

'''
Copy and parse Spotter data 'lvl0', 'lvl1' and 'lvl2' to NetCDF.

Directional moments (a1,b1,a2,b2) are computed from raw 3D displacement data.

A Butterworth filter may be applied to filter low or high frequencies, or
equivalently, low or high wavenumbers using the linear dispersion relation for
surface waves to associate an equivalent frequency.
'''

# Custom utilities
from spot_utils import *

## BEGIN STREAM EDITOR
#  CHECK FILE good_timestamps.txt for "cbd","ced"
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
# filt_data:    Cutoff type (frequency 'freq' or wavenumber 'wnum')
# C0:           Cutoff parameter depending on 'filt_data'

# Data files and directories
# ldata_dir:    Geoposition data directory
# odata_dir:    OGSL data directory
# rdata_dir:    Raw data directory
# lvl0_dir:     'lvl0' data directory
# lvl1_dir:     'lvl1' data directory
# lvl2_dir:     'lvl2' data directory
# ldata_file:   Geoposition data file
# odata_file:   OGSL data file
# rdata_file:   Raw data file
# lvl0_file:    'lvl0' file to write
# lvl1_file:    'lvl1' file to write
# lvl2_file:    'lvl2' file to write
# ---------- #
buoy       = 'spot-0572'
year       = 2023
H          = 10

cbd        = '2023-06-06T00:00:00'
ced        = '2023-06-09T21:30:00'

filt_bool  = False
filt_type  = 'hp'
filt_data  = 'wnum'
C0         = 2.E2

ldata_dir  = '../../RawData/'
odata_dir  = '../../OGSL/statistics/'
rdata_dir  = '../../RawData/'
lvl0_dir   = '../../lvl0/'
lvl1_dir   = '../../lvl1/'
lvl2_dir   = '../../lvl2/'
ldata_file = 'L.txt'
odata_file = 'iml-4_statistics.csv'
rdata_file = 'D.txt'
lvl0_file  = 'lvl0_displacements.nc'
lvl1_file  = 'lvl1_wavespectra.nc'
lvl2_file  = 'lvl2_waveparameters.nc'
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
rmg      = -1

# Compute frequency cutoff
fcut = getFrequency(2*pi/C0,H)

# XYZ index
xyz_cartesian_index = [-3,-2,-1]

# Get magnetic declination
magdec = 0

# Update data directories with buoy and year
ldata_dir = '%s/%s/%d/'%(ldata_dir,buoy,year)
rdata_dir = '%s/%s/%d/'%(rdata_dir,buoy,year)

# Update files with directory, buoy and year
file_fmt   = '%swavebuoy_%s_%s_%d.nc'
buoy_name  = buoy.replace('-','')
rdata_file = os.popen('ls %s*/*/*%s'%(rdata_dir,rdata_file)).read()\
                     .rstrip('\n').split('\n')
ldata_file = os.popen('ls %s*/*/*%s'%(rdata_dir,ldata_file)).read()\
                     .rstrip('\n').split('\n')
lvl0_file  = file_fmt%(lvl0_dir,buoy_name,lvl0_file.split('.nc')[0],year)
lvl1_file  = file_fmt%(lvl1_dir,buoy_name,lvl1_file.split('.nc')[0],year)
lvl2_file  = file_fmt%(lvl2_dir,buoy_name,lvl2_file.split('.nc')[0],year)

# Define level 0, 1 and 2 variables
lvl0_vars = ['x','y','z']
lvl1_vars = ['sxx','syy','szz','cxy','qxz','qyz','a1','b1','a2','b2']
lvl2_vars = ['hm0','tmn10','tm01','tm02','fp','wp','tm','tp','sm','sp']

# Create output directories if inexistant
[sh('mkdir -p %s 2>/dev/null'%d) for d in [lvl0_dir,lvl1_dir,lvl2_dir]]

## OGSL STATISTICS
# Initialize outputs
# For frequency peak, bounds are calculated as the inverse of 'Wave Period'
# latter on in the quality control procedure
bwp_rmin,bwp_rmax,bwp_mean,bwp_std = [[] for _ in range(4)]

# Selection and tickers
bwp_sel     = ['Wave Significant Height',
               'Wave Period',
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
               'Frequency_Peak',
               'Theta_Mean',
               'Theta_Peak',
               'Sigma_Mean',
               'Sigma_Peak']

# Read and append data
DSogsl = pd.read_csv(odata_dir+odata_file,
                     delimiter=',',
                     skipinitialspace=True,
                     skiprows=3)
bwp    = np.array([p.rstrip(' ') for p in DSogsl['Parameter']])
for key in bwp_sel:
    mean_var = DSogsl['mean'][np.where(bwp==key)[0][0]]
    min_var  = DSogsl['min'][np.where(bwp==key)[0][0]]
    std_var  = DSogsl['std'][np.where(bwp==key)[0][0]]
    if key=='Wave Period':
        n_std   = 2
        max_var = max(DSogsl['max'][np.where(bwp==key)[0][0]],1/fcut)
    else:
        n_std   = 1
        max_var = DSogsl['max'][np.where(bwp==key)[0][0]]
    bwp_rmin.append(min_var)
    bwp_rmax.append(max_var)
    bwp_mean.append(mean_var)
    bwp_std.append(n_std*std_var)

# Get 'eps' for test 16
bwp_dt  = 1800
bwp_eps = hstack([bwp_mean[0]/bwp_dt,4*[bwp_mean[1]/bwp_dt],4*[1/bwp_dt]])

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
                  'LVL0_Vars':lvl0_vars,
                  'LVL1_Vars':lvl1_vars,
                  'LVL2_Vars':lvl2_vars,
                  'Raw_File_List':rdata_file,
                  'Raw_Header_Rows':8},
         'Output':{'LVL0_File':lvl0_file,
                   'LVL1_File':lvl1_file,
                   'LVL2_File':lvl2_file},
         'Physics_Parameters':{'Ref_Density':rho_0,
                               'Gravity':g,
                               'Water_Depth':H},
         'Wave_Monitor':{'Amplitude_Factor':afac,
                         'Freq_Min':freq_min,
                         'Freq_Max':freq_max,
                         'X_Position':xpos,
                         'Y_Position':ypos,
                         'Z_Position':zpos,
                         'XYZ_Cartesian_Index':xyz_cartesian_index,
                         'Remove_Gravity':rmg},
         'Filtering':{'Filter':filt_bool,
                      'F_Type':filt_type,
                      'D_Type':filt_data,
                      'C0':C0,
                      'H':H}}

## QUALITY FLAG PARAMETERS
testinit      = []
test_order_st = ['13','10','11','9','12']
test_order_lt = ['18','14','15','20','19','16','17','9','12']
l2v_order     = ['hm0','tmn10','tm01','tm02','fp','tm','tp','sm','sp']

# Short-term
qfst_d = {'Test_9':{'Do_Test':True,
                    'N':4,
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
          'Test_11':{'Do_Test':False,
                     'imin':[-50,-50,-10],
                     'imax':[50,50,10],
                     'lmin':[-50,-50,-10],
                     'lmax':[50,50,10],
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
          'Test_Order':test_order_st}

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
          'Test_17':{'Do_Test':True,
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
          'Test_Order':test_order_lt}

## OUTPUTS
# Write 'lvl0' displacement data
writeLvl0(lvl_d,qfst_d)

# Write 'lvl1' wave spectra
writeLvl1(lvl_d,qfst_d)

# Write 'lvl2' bulk wave parameters
writeLvl2(lvl_d,qflt_d)

# END
