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
buoy      = 'spot-1082'
year      = 2023
month     = range(6,9)
lvl_file  = '../../lvl2/%s_lvl2_bulkparameters_2023.nc'%buoy
pars_file = '../../ParsedData/%s/2023/bulkparameters.csv'%buoy

# Load processed data
DS1 = xr.open_dataset(lvl_file,engine='netcdf4')
x1  = np.array([pd.Timestamp(t).timestamp() for t in DS1.Time.values])
y1  = DS1.Theta_Mean.values

# Load parsed data
DS2 = pd.read_csv(pars_file,delimiter=',',skipinitialspace=True)
x2  = []
y2  = DS2['MeanDirection']
fmt = '%d-%02g-%02gT%02g:%02g:%02g.%d'
for i in range(len(DS2)):
    Y   = DS2['# year'][i]
    M   = DS2['month'][i]
    D   = DS2['day'][i]
    HH  = DS2['hour'][i]
    MM  = DS2['min'][i]
    SS  = DS2['sec'][i]
    MSS = int(np.ceil(DS2['milisec'][i]/100)*100)
    x2.append(pd.Timestamp(fmt%(Y,M,D,HH,MM,SS,MSS)).timestamp())

# Figure
nrows  = 1
ncols  = 1
width  = 8
height = 8
fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(width,height),
                      constrained_layout=True)

# Plots
ax.plot(x1,y1,'b.',ms=1,label='processed')
ax.plot(x2,y2,'r.',ms=1,label='parsed')

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
plt.savefig('comparison_%s_%d.png'%(buoy,year),dpi=120)

# END
