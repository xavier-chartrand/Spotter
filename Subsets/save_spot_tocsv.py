#!/usr/bin/python -u
# Author: Xavier Chartrand
# Email : x.chartrand@protonmail.me
#         xavier.chartrand@ec.gc.ca

'''
Save AZMP wave parameters data to CSV format.
'''

# Module
import numpy as np
import pandas as pd
import os
import xarray as xr

# Shell commands in python
def sh(s): os.system("bash -c '%s'"%s)

## MAIN
# Buoy information
buoy    = 'spot-1082'
year    = '2023'
cbd     = '2023-01-01T00:00:00'
ced     = '2023-12-31T23:59:59'
lvl     = '2'
lvl_id  = 'waveparameters'

# Make input directory and file, and output file
str_cbd  = cbd.split('T')[0]
str_ced  = ced.split('T')[0]
lvl_dir  = '../lvl%s/'%lvl
lvl_file = 'wavebuoy_%s_lvl%s_%s_%s.nc'%(buoy.replace('-',''),lvl,lvl_id,year)
out_dir  = 'csv_files/lvl%s/%s/'%(lvl,buoy)
out_file = 'wavebuoy_%s_lvl%s_%s_%s_%s-%s.csv'\
           %(buoy.replace('-',''),lvl,lvl_id,year,str_cbd,str_ced)

# Load data for given buoy and level
ncvars = ['Hm0','Tm02','Theta_Mean']
DS     = xr.open_dataset(lvl_dir+lvl_file,engine='netcdf4')
units  = [DS[v].attrs['Units'] for v in ncvars]
header = 'Time (UTC), '\
       + ', '.join(['%s (%s)'%(ncvars[i],units[i]) for i in range(len(units))])

# Retrieve cropped time indices
time = np.array([pd.Timestamp(t).timestamp() for t in DS.Time.values])
i0   = abs(time-pd.Timestamp(cbd).timestamp()).argmin()
i1   = abs(time-pd.Timestamp(ced).timestamp()).argmin()

# Output cropped dataset
os.system("bash -c '%s'"%('mkdir -p %s 2>/dev/null'%out_dir))
DSo   = DS.isel(Time=slice(i0,i1))
timeo = DSo.Time.values
dim   = len(timeo)
sh('echo "%s" > %s 2>/dev/null'%(header,out_dir+out_file))
for i in range(dim):
    vtime     = ('%s'%DSo.Time[i].values).rstrip('0').rstrip('.')
    vdata     = ['%.8f'%DSo[v][i].values for v in ncvars]
    data_line = '%s, '%vtime + ', '.join(vdata)
    sh('echo "%s" >> %s 2>/dev/null'%(data_line,out_dir+out_file))

# END
