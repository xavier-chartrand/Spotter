#!/usr/bin/python -u
# Author: Xavier Chartrand
# Email : x.chartrand@protonmail.me
#         xavier.chartrand@ec.gc.ca

'''
Plot wind provenance versus wave provenance for a given AZMP buoy and year.
'''

# Modules
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Matplotlib settings and TeX environment
plt.rcParams.update({
    "pgf.texsystem":"pdflatex",
    "text.usetex":True,
    "font.family":"sans-serif",
    "font.sans-serif":"DejaVu Sans",
    "font.size":16,
    "savefig.dpi":240,
})

## MAIN
# Parameters
buoy      = 'spot-1082'
year      = 2023
month     = range(5,13)
lvl2_dir  = '../../lvl2/'
lvl0_file = 'ncfiles/wavebuoy_iml4_lvl0_auxiliaryvariables_%s.nc'%year
lvl2_file = 'wavebuoy_%s_lvl2_waveparameters.nc'%buoy.replace('-','')

# Update files and directories with buoy and year
lvl2_file = '%s%s_%d.nc'%(lvl2_dir,lvl2_file.split('.nc')[0],year)

# Load data
DS0   = xr.open_dataset(lvl0_file,engine='netcdf4')
DS2   = xr.open_dataset(lvl2_file,engine='netcdf4')
time0 = np.array([pd.Timestamp(t).timestamp() for t in DS0.Time.values])
time2 = np.array([pd.Timestamp(t).timestamp() for t in DS2.Time.values])
wd    = (DS0.Wind_Provenance.values+16.5)%360
ws    = DS0.Wind_Speed.values
hm0   = DS2.Hm0.values
tm    = DS2.Theta_Mean.values

## FIGURES
# Figure 1: Wind speed versus significant wave height
colors  =[[217/255,83/255,25/255],
          [31/255,119/255,180/255]]
fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(15,5),
                       gridspec_kw={'width_ratios':[1,0.5]},
                       constrained_layout=True)
ax0     = axs[0]
ax0t    = ax0.twinx()
ax1     = axs[1]

# Plots
ax0.plot(time0,ws,'-',color=colors[0],lw=0.5)
ax0t.plot(time2,hm0,'-',color=colors[1],lw=1)
#ax1.plot(hm0,ws[1:],'k.',ms=1)

# Axis properties
# x axis
sx0_scl    = 86400
sx1_scl    = 0.1
x0_scl     = None
x1_scl     = 0.5
x0_lim_0   = pd.Timestamp('%04g-%02g-01T00:00:00'%(year,month[0])).timestamp()
x0_lim_1   = pd.Timestamp('%04g-%02g-01T00:00:00'%(year,month[-1])).timestamp()
x1_lim_0   = 0
x1_lim_1   = np.ceil(np.nanmax(hm0/x1_scl))*x1_scl
nx1        = int((x1_lim_1-x1_lim_0)//x1_scl) + 1
x0_tcks    = [pd.Timestamp('%04g-%02g-01T00:00:00'%(year,m)).timestamp()\
              for m in month]
x1_tcks    = np.linspace(x1_lim_0,x1_lim_1,nx1)
x0_labtcks = ['%04g-%02g'%(year,m) for m in month]
x1_labtcks = [('%.2f'%t).rstrip('0').rstrip('.') for t in x1_tcks]
ax0.set_xlabel(r'Date (UTC)',fontsize=20)
ax1.set_xlabel(r'$H_{m0}$ (m)',fontsize=20)
ax0.set_xticks(x0_tcks,color=colors[0])
ax1.set_xticks(x1_tcks,color=colors[1])
ax0.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(sx0_scl))
ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(sx1_scl))
ax0.set_xticklabels(x0_labtcks)
ax1.set_xticklabels(x1_labtcks)
ax0.set_xlim(x0_lim_0-sx0_scl//2,x0_lim_1+sx0_scl//2)
ax1.set_xlim(x1_lim_0-sx1_scl//2,x1_lim_1+sx1_scl//2)
# y axis
sy0_scl     = 1
sy0t_scl    = 0.1
sy1_scl     = 1
y0_scl      = 5
y0t_scl     = 0.5
y1_scl      = 5
y0_lim_0    = 0
y0_lim_1    = np.ceil(np.nanmax(ws/y0_scl))*y0_scl
y0t_lim_0   = 0
y0t_lim_1   = np.ceil(np.nanmax(hm0/y0t_scl))*y0t_scl
y1_lim_0    = 0
y1_lim_1    = np.ceil(np.nanmax(ws/y1_scl))*y1_scl
ny0         = int((y0_lim_1-y0_lim_0)//y0_scl) + 1
ny0t        = int((y0t_lim_1-y0t_lim_0)//y0t_scl) + 1
ny1         = int((y1_lim_1-y1_lim_0)//y1_scl) + 1
y0_tcks     = np.linspace(y0_lim_0,y0_lim_1,ny0)
y0t_tcks    = np.linspace(y0t_lim_0,y0t_lim_1,ny0t)
y1_tcks     = np.linspace(y1_lim_0,y1_lim_1,ny1)
y0_labtcks  = [('%.2f'%t).rstrip('0').rstrip('.') for t in y0_tcks]
y0t_labtcks = [('%.2f'%t).rstrip('0').rstrip('.') for t in y0t_tcks]
y1_labtcks  = [('%.2f'%t).rstrip('0').rstrip('.') for t in y1_tcks]
ax0.set_ylabel(r'Wind speed (m/s)',color=colors[0],fontsize=20)
ax0t.set_ylabel(r'$H_{m0}$ (m)',color=colors[1],fontsize=20)
ax1.set_ylabel(r'Wind speed (m/s)',fontsize=20)
ax0.set_yticks(y0_tcks,color=colors[0])
ax0t.set_yticks(y0t_tcks,color=colors[1])
ax1.set_yticks(y1_tcks)
ax0.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(sy0_scl))
ax0t.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(sy0t_scl))
ax1.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(sy1_scl))
ax0.set_yticklabels(y0_labtcks,color=colors[0])
ax0t.set_yticklabels(y0t_labtcks,color=colors[1])
ax1.set_yticklabels(y1_labtcks)
ax0.set_ylim(y0_lim_0-sy0_scl//2,y0_lim_1+sy0_scl//2)
ax0t.set_ylim(y0t_lim_0-sy0t_scl//2,y0t_lim_1+sy0t_scl//2)
ax1.set_ylim(y1_lim_0-sy1_scl//2,y1_lim_1+sy1_scl//2)

# Title
fig.suptitle('%s %d'%(buoy,year),fontsize=30)
plt.savefig('figures/windwave_magnitude_%s_%d.png'%(buoy,year),dpi=120)

# Figure 2: Wind direction versus wave direction
fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(15,5),
                       gridspec_kw={'width_ratios':[1,0.5]},
                       constrained_layout=True)
ax0     = axs[0]
ax1     = axs[1]

# Plots
ax0.plot(time2,tm,'.',color=colors[0],ms=1,label='wave provenance')
ax0.plot(time0,wd,'.',color=colors[1],ms=1,label='wind provenance')
#ax1.plot(tm,wd[1:],'k.',ms=1)
ax1.plot([0,360],[0,360],'k-',lw=1)

# Axis properties
# x axis
sx0_scl    = 86400
sx1_scl    = 10
x0_scl     = None
x1_scl     = 90
x0_lim_0   = pd.Timestamp('%04g-%02g-01T00:00:00'%(year,month[0])).timestamp()
x0_lim_1   = pd.Timestamp('%04g-%02g-01T00:00:00'%(year,month[-1])).timestamp()
x1_lim_0   = 0
x1_lim_1   = 360
nx1        = int((x1_lim_1-x1_lim_0)//x1_scl) + 1
x0_tcks    = [pd.Timestamp('%04g-%02g-01T00:00:00'%(year,m)).timestamp()\
              for m in month]
x1_tcks    = np.linspace(x1_lim_0,x1_lim_1,nx1)
x0_labtcks = ['%04g-%02g'%(year,m) for m in month]
x1_labtcks = ['%d'%t for t in x1_tcks]
ax0.set_xlabel(r'Date (UTC)',fontsize=20)
ax1.set_xlabel('Wave provenance (T.N.)',fontsize=20)
ax0.set_xticks(x0_tcks)
ax1.set_xticks(x1_tcks)
ax0.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(sx0_scl))
ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(sx1_scl))
ax0.set_xticklabels(x0_labtcks)
ax1.set_xticklabels(x1_labtcks)
ax0.set_xlim(x0_lim_0-sx0_scl//2,x0_lim_1+sx0_scl//2)
ax1.set_xlim(x1_lim_0-sx1_scl//2,x1_lim_1+sx1_scl//2)
# y axis
sy_scl    = 10
y_scl     = 90
y_lim_0   = 0
y_lim_1   = 360
ny        = int((y_lim_1-y_lim_0)//y_scl) + 1
y_tcks    = np.linspace(y_lim_0,y_lim_1,ny)
y_labtcks = ['%d'%t for t in y_tcks]
ax0.set_ylabel('Provenance (T.N.)',fontsize=20)
ax1.set_ylabel('Wind provenance (T.N.)',fontsize=20)
ax0.set_yticks(y_tcks)
ax1.set_yticks(y_tcks)
ax0.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(sy_scl))
ax1.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(sy_scl))
ax0.set_yticklabels(y_labtcks)
ax1.set_yticklabels(y_labtcks)
ax0.set_ylim(y_lim_0-sy_scl//2,y_lim_1+sy_scl//2)
ax1.set_ylim(y_lim_0-sy_scl//2,y_lim_1+sy_scl//2)

# Title
fig.suptitle('%s %d'%(buoy,year),fontsize=30)
plt.savefig('figures/windwave_direction_%s_%d.png'%(buoy,year),dpi=120)

# END
