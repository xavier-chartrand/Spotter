#!/usr/bin/python -u
# Author: Xavier Chartrand
# Email : xavier.chartrand@ec.gc.ca
#         xavier.chartrand@proton.me
#         xavier.chartrand@uqar.ca

'''
Retrieve a subset of Spotter data.
'''

# Module
import numpy as np
import pandas as pd
import os
import xarray as xr

## MAIN
# Buoy information
buoy    = 'spot-1082'
year    = '2023'
cbd     = '2023-08-01T00:00:00'
ced     = '2023-08-31T23:59:59'
lvl     = '2'
lvl_id  = 'waveparameters'

# Make input directory and file, and output file
lvl_dir  = '../lvl%s/%s/'%(lvl,buoy)
lvl_file = 'wavebuoy_%s_lvl%s_%s_%s.nc'%(buoy.replace('-',''),lvl,lvl_id,year)
out_dir  = 'lvl%s_subsetted/'%lvl
out_file = 'wavebuoy_%s_lvl%s_%s_%s_%s_subsetted.nc'\
           %(buoy.replace('-',''),lvl,lvl_id,cbd.split('T')[0],ced.split('T')[0])

# Load data for given buoy and level
DS = xr.open_dataset(lvl_dir+lvl_file,engine='netcdf4')

# Retrieve cropped time indices
time = np.array([pd.Timestamp(t).timestamp() for t in DS.Time.values])
i0   = abs(time-pd.Timestamp(cbd).timestamp()).argmin()
i1   = abs(time-pd.Timestamp(ced).timestamp()).argmin()

# Output cropped dataset
os.system("bash -c '%s'"%('mkdir -p %s 2>/dev/null'%out_dir))
DSo = DS.isel(Time=slice(i0,i1))
DSo.to_netcdf(out_dir+out_file,engine='netcdf4')

# END
