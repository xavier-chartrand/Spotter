#!/usr/bin/python -u
# Author: Xavier Chartrand
# Email : x.chartrand@protonmail.me
#         xavier.chartrand@ec.gc.ca

'''
Plot the quality flag histogram for windwave data
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
buoy      = 'spot1082'
year      = 2023
month     = range(5,12)
lvl2_dir  = '../../lvl2/'
lvl2_file = 'wavebuoy_%s_lvl2_waveparameters.nc'%buoy.replace('-','')

# Update files and directories with buoy and year
lvl2_file = '%s%s_%d.nc'%(lvl2_dir,lvl2_file.split('.nc')[0],year)

# Load data
DS2       = xr.open_dataset(lvl2_file,engine='netcdf4')
time      = np.array([pd.Timestamp(t).timestamp() for t in DS2.Time.values])
hm0_qf    = DS2.Hm0.QC
tmn10_qf  = DS2['Tm-10'].QC
tm01_qf   = DS2.Tm01.QC
tm02_qf   = DS2.Tm01.QC
freqp_qf  = DS2.Frequency_Peak.QC
wnump_qf  = DS2.Wavenumber_Peak.QC
thetam_qf = DS2.Theta_Mean.QC
thetap_qf = DS2.Theta_Peak.QC
sigmam_qf = DS2.Sigma_Mean.QC
sigmap_qf = DS2.Sigma_Peak.QC
dim       = len(time)

# Count quality flag occurences
hm0_qf_lab,hm0_qf_cnt       = np.unique(hm0_qf,return_counts=True)
tmn10_qf_lab,tmn10_qf_cnt   = np.unique(tmn10_qf,return_counts=True)
tm01_qf_lab,tm01_qf_cnt     = np.unique(tm01_qf,return_counts=True)
tm02_qf_lab,tm02_qf_cnt     = np.unique(tm02_qf,return_counts=True)
freqp_qf_lab,freqp_qf_cnt   = np.unique(freqp_qf,return_counts=True)
wnump_qf_lab,wnump_qf_cnt   = np.unique(wnump_qf,return_counts=True)
thetam_qf_lab,thetam_qf_cnt = np.unique(thetam_qf,return_counts=True)
thetap_qf_lab,thetap_qf_cnt = np.unique(thetap_qf,return_counts=True)
sigmam_qf_lab,sigmam_qf_cnt = np.unique(sigmam_qf,return_counts=True)
sigmap_qf_lab,sigmap_qf_cnt = np.unique(sigmap_qf,return_counts=True)

labels   = [hm0_qf_lab,
            tm01_qf_lab,
            freqp_qf_lab,
            thetam_qf_lab,
            sigmam_qf_lab,
            tmn10_qf_lab,
            tm02_qf_lab,
            wnump_qf_lab,
            thetap_qf_lab,
            sigmap_qf_lab]
counts   = [hm0_qf_cnt,
            tm01_qf_cnt,
            freqp_qf_cnt,
            thetam_qf_cnt,
            sigmam_qf_cnt,
            tmn10_qf_cnt,
            tm02_qf_cnt,
            wnump_qf_cnt,
            thetap_qf_cnt,
            sigmap_qf_cnt]
ticks    = [range(len(hm0_qf_cnt)),
            range(len(tm01_qf_cnt)),
            range(len(freqp_qf_cnt)),
            range(len(thetam_qf_cnt)),
            range(len(sigmam_qf_cnt)),
            range(len(tmn10_qf_cnt)),
            range(len(tm02_qf_cnt)),
            range(len(wnump_qf_cnt)),
            range(len(thetap_qf_cnt)),
            range(len(sigmap_qf_lab))]
ax_title = [r'$\mathbf{H_{m0}}$',
            r'$\mathbf{T_{m_{0,1}}}$',
            r'$\mathbf{f_{peak}}$',
            r'$\mathbf{\theta_{mean}}$',
            r'$\mathbf{\sigma_{mean}}$',
            r'$\mathbf{T_{m_{-1,0}}}$',
            r'$\mathbf{T_{m_{0,2}}}$',
            r'$\mathbf{k_{peak}}$',
            r'$\mathbf{\theta_{peak}}$',
            r'$\mathbf{\sigma_{peak}}$']

## FIGURES
# Wave bulk parameters qf
fig,axs = plt.subplots(nrows=2,ncols=5,figsize=(20,10),
                       gridspec_kw={'width_ratios':[1,1,1,1,1]},
                       constrained_layout=True)
ax00    = axs[0,0]
ax01    = axs[0,1]
ax02    = axs[0,2]
ax03    = axs[0,3]
ax04    = axs[0,4]
ax10    = axs[1,0]
ax11    = axs[1,1]
ax12    = axs[1,2]
ax13    = axs[1,3]
ax14    = axs[1,4]

# Plots
for i in range(len(ticks)):
    # plot
    ax = axs.ravel()[i]
    ax.bar(ticks[i],counts[i],align='center')
    ax.set_xticks(ticks[i],labels[i])
    # text
    str1 = r'%d good (%.2f\%%)'%(counts[i][0],counts[i][0]/dim*100)
    str2 = r'%d missing (%.2f\%%)'%(counts[i][-1],counts[i][-1]/dim*100)
    str3 = r'%d bad or suspect (%.2f\%%)'\
            %(sum(counts[i][1:-1]),sum(counts[i][1:-1])/dim*100)
    ax.text(0.98,0.925,ax_title[i],ha='right',va='center',
            transform=ax.transAxes,fontsize=24)
    ax.text(0.98,0.80,str1,ha='right',va='center',
            transform=ax.transAxes,fontsize=12)
    ax.text(0.98,0.75,str2,ha='right',va='center',
            transform=ax.transAxes,fontsize=12)
    ax.text(0.98,0.70,str3,ha='right',va='center',
            transform=ax.transAxes,fontsize=12)

# Axis properties
# x axis
ax10.set_xlabel(r'qf code',fontsize=24)
ax11.set_xlabel(r'qf code',fontsize=24)
ax12.set_xlabel(r'qf code',fontsize=24)
ax13.set_xlabel(r'qf code',fontsize=24)
ax14.set_xlabel(r'qf code',fontsize=24)
# y axis
yscl      = np.floor(np.log10(dim))
ylim_0    = 0
ylim_1    = int(np.ceil(dim/10**yscl)*10**yscl)
ny        = int(ylim_1/10**yscl+1)
y_tcks    = np.linspace(ylim_0,ylim_1,ny)
ylab_tcks = ['%d'%t for t in y_tcks]
ax00.set_ylabel(r'count',fontsize=24)
ax10.set_ylabel(r'count',fontsize=24)
[ax.set_yticks(y_tcks) for ax in axs.ravel()]
[ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10**(yscl-1)))\
 for ax in axs.ravel()]
[ax.set_yticklabels(ylab_tcks) for ax in axs[:,0]]
[ax.set_yticklabels([]) for ax in axs[:,1:].ravel()]

# Title
fig.suptitle('%s %d quality flag overview'%(buoy,year),fontsize=30)
plt.savefig('figures/qf_overview_%s_%d.png'%(buoy,year),dpi=120)
plt.close(fig)

# END
