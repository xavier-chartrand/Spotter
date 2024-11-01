#!/usr/bin/python -u
# Author: Xavier Chartrand
# Email : x.chartrand@protonmail.me
#         xavier.chartrand@ec.gc.ca

'''
Plot wind provenance versus wave provenance for a given AZMP buoy and year.
'''

# Custom utilities
from spot_utils import *

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
buoy       = 'spot-0572'
year       = 2023
month      = range(6,9)
lvl2_dir   = 'lvl2/'
lvl20_file = 'lvl2_bulkparameters.nc'

# Update files and directories with buoy and year
lvl20_file = '%s%s_%s_%d.nc'%(lvl2_dir,buoy,lvl20_file.split('.nc')[0],year)

# Load data
DS20 = xr.open_dataset(lvl20_file,engine='netcdf4')
tm   = DS20.Theta_Mean.values
tm_t = np.array([pd.Timestamp(t).timestamp() for t in DS20.Time.values])

# Find and remove "NaN" index
inan_tm = np.where(np.invert(np.isnan(tm)))[0]
tm,tm_t = tm[inan_tm],tm_t[inan_tm]

# Figure
nrows  = 1
ncols  = 1
width  = 8
height = 8
fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(width,height),
                      constrained_layout=True)

# Plots
ax.plot(tm_t,tm,'b.',ms=1,label='wave provenance')

# Axis properties
# x axis
x_tcks    = [pd.Timestamp('%04g-%02g-01T00:00:00'%(year,m)).timestamp()\
             for m in month]
xlab_tcks = ['%04g-%02g'%(year,m) for m in month]
ax.set_xlabel(r'Date',fontsize=20)
ax.set_xticks(x_tcks)
ax.set_xticklabels(xlab_tcks)
# y axis
y_tcks    = np.linspace(0,360,5)
ylab_tcks = ['%d'%t for t in y_tcks]
ax.set_ylabel('Provenance [TN ${}^\circ$]',fontsize=20)
ax.set_yticks(y_tcks)
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
ax.set_yticklabels(ylab_tcks)

# Legend
ax.legend(handlelength=1,handletextpad=1,markerscale=10,loc='upper right')

# Title
fig.suptitle('%s %d'%(buoy,year),fontsize=30)
plt.savefig('windwave_%s_%d.png'%(buoy,year),dpi=120)

# END
