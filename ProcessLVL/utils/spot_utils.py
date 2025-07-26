#!/usr/bin/python -u
# Author: Xavier Chartrand
# Email : x.chartrand@protonmail.me
#         xavier.chartrand@ec.gc.ca

'''
SPOT buoy utilities.
'''

# Module
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import warnings
import xarray as xr
# Functions
from numpy import array,invert,hstack,transpose,where
from scipy.signal.windows import get_window as getWindow
from scipy.special import erfcinv
from scipy.stats import median_abs_deviation as mad
from scipy.stats import pearsonr
# Constants
from scipy.constants import pi
# Custom utilities
from qf_utils import *

### Shell commands in python
# -------------------------------------------------- #
def sh(s): os.system("bash -c '%s'"%s)

### "spot_utils" utilities
# -------------------------------------------------- #
def getSPOTLocFromFileList(f_list,hrows=7):
    '''
    Retrieve timestamp, longitude and latitude from a SPOT location file list.
    '''

    # Initialize outputs
    ts,lon,lat = [[] for _ in range(3)]

    # Open and read file
    for f_name in f_list:
        DS = array(pd.read_csv(f_name,
                   delimiter=',',
                   skipinitialspace=True,
                   skiprows=hrows,
                   header=None))

        # Get year, mont, day, hour, minute, second, millisecond
        Y   = [int(y) for y in DS[:,0]]
        M   = [int(m) for m in DS[:,1]]
        D   = [int(d) for d in DS[:,2]]
        HH  = [int(hh) for hh in DS[:,3]]
        MM  = [int(mm) for mm in DS[:,4]]
        SS  = [int(ss) for ss in DS[:,5]]
        MSS = [int(np.ceil(mss/100)*100) for mss in DS[:,6]]

        # Calculate timestamps
        tfrmt = '%d-%02g-%02gT%02g:%02g:%02g.%d'
        tss   = [pd.Timestamp(tfrmt%(Y[i],M[i],D[i],HH[i],MM[i],SS[i],MSS[i]))\
                            .timestamp()\
                 for i in range(len(DS))]

        # Append data
        ts.append(tss)
        lat.append(DS[:,7])
        lon.append(DS[:,8])

    return hstack(ts),hstack(lon),hstack(lat)

# ---------- #
def getSPOTTimeFromFile(f_name,hrows=7):
    '''
    Retrieve date from a SPOT displacement data file.
    '''

    # Open and read file
    DS = array(pd.read_csv(f_name,
                           delimiter=',',
                           skipinitialspace=True,
                           skiprows=hrows,
                           header=None))

    # Get year, mont, day, hour, minute, second, millisecond
    Y   = [int(y) for y in DS[:,0]]
    M   = [int(m) for m in DS[:,1]]
    D   = [int(d) for d in DS[:,2]]
    HH  = [int(hh) for hh in DS[:,3]]
    MM  = [int(mm) for mm in DS[:,4]]
    SS  = [int(ss) for ss in DS[:,5]]
    MSS = [int(np.ceil(mss/100)*100) for mss in DS[:,6]]

    # Append to time list
    time_list = [(Y[i],M[i],D[i],HH[i],MM[i],SS[i],MSS[i])\
                 for i in range(np.shape(DS)[0])]

    return time_list

# ---------- #
def getBWFilter(freq,filt_p,n=5,g=9.81):
    '''
    Compute low-or-high pass ButtherWorth for frequencies 'freq'
    (default order: n=5).

    "F_Type"    is the filter type, low pass 'lp', high pass 'hp' ;
    "D_Type"    is the cutoff type, length scale 'length', frequency 'freq' ;
    "C0"        is the cutoff value.
    '''

    # Unpack "filt_p"
    bool_filt = filt_p['Filter']
    ftype     = filt_p['F_Type']
    dtype     = filt_p['D_Type']
    c0        = filt_p['C0']
    H         = filt_p['H'] if 'H' in filt_p.keys() else 0

    # Return ones if no filtering specified
    if not bool_filt: return np.ones(len(freq))

    # Raise error if water's depth is missing for 'dtype'=='length'
    if dtype=='length' and not H:
        raise TypeError("'H' missing for data type 'length'")

    # Sign of low or high pass filter
    fsgn = 1 if ftype=='lp' else -1 if ftype=='hp' else 0

    # Swap cutoff length to frequency if specified
    c0 = (np.tanh(2*pi*H/c0)*g/2/pi/c0)**(0.5) if dtype=='length' else c0

    # Compute Butterworth filter
    filt = fsgn/(1 + (freq/freq[abs(freq-c0).argmin()])**(2*n))**(0.5)\
         + (1 - fsgn)/2

    return filt

# ---------- #
def getCSD(X1,X2,dsmp,win='hann'):
    '''
    Compute the cross-covariance power density (cross-spectrum) between
    "X1" and "X2" time series.

    "dsmp"      is the sampling period ;
    "win"       is the windowing function.
    '''

    # Compute series length and total time
    lX = len(X1)                                # length of series
    T  = lX*dsmp                                # total time

    # Get window
    try:W=getWindow(win,lX)
    except:W=np.ones(lX)

    # Right-sided, normalized Fourier transform of windowed series
    F1 = np.fft.rfft(W*X1)/lX
    F2 = np.fft.rfft(W*X2)/lX

    # Cross-spectral density times windowing normalization factors
    # (normalization complies with 'scipy.signals.csd')
    qfac = 2*hstack([1/2,np.ones(lX//2)]) if lX%2 else\
           2*hstack([1/2,np.ones(lX//2-1),1/2]) # right-sided norm factor
    wfac = 1/np.mean(W**2)                      # window norm factor
    nfac = T*qfac*wfac                          # CSD norm factor

    # Compute cross-spectrum
    S = nfac*F1*np.conj(F2)

    return S

# ---------- #
def getDirMoments(CS,weight=False,Ef=None,fs=None):
    '''
    Compute directional moments (Fourier coefficients) up to 2nd order.

    Directional moments may be normalized by the total available energy,
    then the variance wave spectrum and the sampling frequency must be given.

    "CS" must be expressed as "CS=[CS_xx,CS_yy,CS_zz,CS_xy,CS_xz,CS_yz]".

    The convention for coincident (co) and quadrature (quad) spectra is
        cross = co - 1j*quad = real(cross) -1j*imag(cross)
    '''

    # Retrieve auto, coincident and quadrature spectra
    Axx = np.real(CS[0])                        # "xx" auto-spectra
    Ayy = np.real(CS[1])                        # "yy" auto-spectra
    Azz = np.real(CS[2])                        # "zz" auto-spectra
    Cxy = np.real(CS[3])                        # "xy" co-spectra
    Qxz = -np.imag(CS[4])                       # "xz" quad-spectra
    Qyz = -np.imag(CS[5])                       # "yz" quad-spectra

    # Normalize spectra if specified
    if weight and type(Ef)!=type(None) and type(fs)!=type(None):
        Axx = getWeightedParam(Axx,Ef,fs)       # "Axx" normalized
        Ayy = getWeightedParam(Ayy,Ef,fs)       # "Ayy" normalized
        Azz = getWeightedParam(Azz,Ef,fs)       # "Azz" normalized
        Cxy = getWeightedParam(Cxy,Ef,fs)       # "Cxy" normalized
        Qxz = getWeightedParam(Qxz,Ef,fs)       # "Qxz" normalized
        Qyz = getWeightedParam(Qyz,Ef,fs)       # "Qyz" normalized
    elif weight:
        raise TypeError("Specify variance spectrum and sampling frequency")

    # Compute directional moments (Fourier coefficients)
    a1 = Qxz/np.sqrt(Azz*(Axx+Ayy))             # "a1" directional moment
    b1 = Qyz/np.sqrt(Azz*(Axx+Ayy))             # "b1" directional moment
    a2 = (Axx - Ayy)/(Axx + Ayy)                # "a2" directional moment
    b2 = 2*Cxy/(Axx + Ayy)                      # "b2" directional moment

    return a1,b1,a2,b2

# ---------- #
def getFrequency(k,H,g=9.81):
    '''
    Estimate a frequency using the linear dispersion relation for surface
    waves, for a single wavenumber "k".

    "k"         is the wavenumber ;
    "H"         is the water depth.
    '''

    return (g*k*np.tanh(k*H))**(0.5)/2/pi

# ---------- #
def getFreqMoment(Ef,freq,n):
    '''
    Compute the nth order statistical frequency moment.

    "freq"      is the frequency constantly sampled.
    '''

    return iTrapz1o(Ef*freq**n,np.diff(freq)[0],0)[-1]

# ---------- #
def getOutlierIndex(F,n,method='mad'):
    '''
    Find outliers indices for a ND raveled array based on different methods.
    '''

    # Array dimensions
    dim = np.shape(F)                           # array dimension
    F   = F.ravel()                             # ravel F array to 1D
    inn = where(invert(isnan(F)))[0]            # non NaN F index

    if method=='mad':                           # median absolute deviation
        ioF  = F - np.nanmedian(F)
        thrs = -n/(2**(1/2)*erfcinv(3/2))*mad(F[inn])
    elif method=='mean':                        # mean
        ioF  = F - np.nanmean(F)
        thrs = n*np.nanstd(F)
    else:
        raise('Implement a new method or specify a valid one')

    # Return raveled outliers indices
    return abs(ioF)>=thrs

# ---------- #
def getSpreading(phi,a1,a2,b1,b2):
    '''
    Compute the spreading function from "a1,a2,b1,b2" directional moments
    using weighted Fourier series with positive coefficients
    (Longuet-Higgins 1963).
    '''

    # Smooth-positive coefficients
    cD = [1/2/pi,2/3/pi,1/6/pi]

    return array([cD[1]*(a1*np.cos(p)+b1*np.sin(p))+\
                  cD[2]*(a2*np.cos(2*p)+b2*np.sin(2*p))
                  for p in phi]) + cD[0]

# ---------- #
def getWavenumber(om,H,da1=1.E3,thrs=1.E-10,g=9.81):
    '''
    Estimate a wavenumber using the linear dispersion relation for surface
    waves, for a single angular velocity "om".

    "om"        is the angular velocity ;
    "H"         is the water depth.
    '''

    # Remove warnings
    warnings.filterwarnings('ignore')

    # Find roots with secant method
    a0 = om**2*H/g
    a1 = a0*np.tanh(a0**(3/4))**(-2/3)
    while abs(da1/a1)>thrs:
        da1 = (a0-a1*np.tanh(a1))/(a1*np.cosh(a1)**(-2)+np.tanh(a1))
        a1 += da1

    return a1/H

# ---------- #
def getWeightedParam(X,Ef,fs):
    '''
    Compute the spectrally weighted mean parameter.
    '''

    return iTrapz1o(Ef*X,fs,0)[-1]/iTrapz1o(Ef,fs,0)[-1]

# ---------- #
def iTrapz1o(f,dx,f0):
    '''
    Compute first order trapezoidal integral of "f(x)" on the full "x"
    interval (i.e. from "f[0]" to "f[-1]"). An initial value of "f0" can be
    specified as an integration constant.

    "dx"        is the differential supposed constant.
    '''

    return np.nancumsum(hstack([f0,(f[:-1]+f[1:])*dx/2]))

# ---------- #
def positiveCosAngle(arad):
    '''
    Swap an angle from "[0,2*pi]" to "[-pi,pi]" (i.e having positive cosine).
    '''

    return arad if arad<=pi else arad-2*pi

### WRITE LVL
# ---------- #
def writeLvl0(lvl_d,qfst_d):
    '''
    Write level 0 (surface motions) of Spotter buoy.
    '''

    ## SET
    # Unpack 'lvl_d'
    buoy      = lvl_d['Info']['Id']
    cdb       = lvl_d['Info']['Corrected_Date_Begin']
    cde       = lvl_d['Info']['Corrected_Date_End']
    rec_len   = lvl_d['Info']['Wave_Record_Length']
    wreg_dt   = lvl_d['Info']['Wave_Regular_Length']
    areg_dt   = lvl_d['Info']['Aux_Regular_Length']
    fs        = lvl_d['Info']['Sampling_Frequency']
    loc_list  = lvl_d['Input']['Loc_File_List']
    file_list = lvl_d['Input']['Raw_File_List']
    hrows     = lvl_d['Input']['Raw_Header_Rows']
    afac      = lvl_d['Wave_Monitor']['Amplitude_Factor']
    xyz_ci    = lvl_d['Wave_Monitor']['XYZ_Cartesian_Index']

    # Calculate record length in time index
    irec_len = int(rec_len*fs)

    # Get timestamp and date
    tfrmt   = '%d-%02g-%02gT%02g:%02g:%02g.%d'
    raw_ts  = array([pd.Timestamp(tfrmt%d).timestamp()\
                     for file in file_list\
                     for d in getSPOTTimeFromFile(file)])
    file_ts = array([pd.Timestamp(tfrmt%getSPOTTimeFromFile(file)[0])\
                                 .timestamp()\
                     for file in file_list])

    # Truncate with begin and end dates
    cdb_ts    = pd.Timestamp(cdb).timestamp()
    cde_ts    = pd.Timestamp(cde).timestamp()
    iraw_db   = where((raw_ts-cdb_ts)>=0)[0][0]
    iraw_de   = where((raw_ts-cde_ts)<=0)[0][-1] + 1
    ifile_db  = where((file_ts-cdb_ts)>=0)[0][0]
    ifile_de  = where((file_ts-cde_ts)<=0)[0][-1] + 1
    raw_ts    = raw_ts[iraw_db:iraw_de]
    file_list = file_list[ifile_db:ifile_de]

    # Get location information
    loc_ts,lon,lat = getSPOTLocFromFileList(loc_list)

    # Compute regular timestamp and date
    reg_tsi  = np.floor(raw_ts[0]/wreg_dt)*wreg_dt
    reg_tsf  = np.ceil(raw_ts[-1]/wreg_dt)*wreg_dt
    n_ts     = int((reg_tsf-reg_tsi)/wreg_dt) + 1
    reg_ts   = np.linspace(reg_tsi,reg_tsf,n_ts)
    reg_date = array([pd.Timestamp(t,unit='s') for t in reg_ts])

    # Initialize outputs
    time_range = np.arange(irec_len)/fs
    data       = np.nan*np.ones((len(reg_ts),3,len(time_range)))
    location   = np.nan*np.ones((len(reg_ts),2))
    dim        = len(reg_ts)
    nfiles     = len(file_list)

    ## PARSE LEVEL 0
    print('\nParsing level 0 for %s...'%buoy)
    progress = []

    # Iterate over files
    for i in range(nfiles):
        # Print progress
        iprog    = int(i/(nfiles-1)*20)
        progress = printProgress(iprog,progress)

        # Open displacement data
        DSo = array(pd.read_csv(file_list[i],
                                delimiter=',',
                                skipinitialspace=True,
                                skiprows=hrows,
                                header=None))

        # Get time information
        date = [tfrmt%t for t in getSPOTTimeFromFile(file_list[i])]
        ts   = [pd.Timestamp(d).timestamp() for d in date]

        # Get file timestamps
        cnt = -1
        for t in ts:
            # Get regular timestamp and time range indices
            cnt     += 1
            ireg_ts  = where(reg_ts-t<=0)[0][-1]
            irange_t = abs(time_range-abs(reg_ts[ireg_ts]-t)).argmin()

            # Append displacement data and apply amplitude factor
            data[ireg_ts,:,irange_t] = afac*DSo[cnt,xyz_ci]

    # Append location with nearest lon,lat values to 30-minutes regular series
    for i in range(n_ts):
        iloc_ts = abs(loc_ts-reg_ts[i]).argmin()
        if abs(loc_ts[iloc_ts]-reg_ts[i])<wreg_dt:
            location[i,0] = lon[iloc_ts]
            location[i,1] = lat[iloc_ts]
        else: continue

    ## QUALITY FLAGS
    # Retrieve 3D displacement from "data"
    xdata,ydata,zdata = data[:,0,:],data[:,1,:],data[:,2,:]
    lon,lat           = location[:,0],location[:,1]

    # Compute quality flag
    xdata,qf_xdata = getSTQF(xdata,'"Displacement_X"',qfst_d)
    ydata,qf_ydata = getSTQF(ydata,'"Displacement_Y"',qfst_d)
    zdata,qf_zdata = getSTQF(zdata,'"Displacement_Z"',qfst_d)

    ## OUTPUTS
    # Variables attributes
    xdisp_attrs = {'Description':'Eastward displacement of the platform '\
                                +'measured with GPS-based method',
                   'Units':'Meter'}
    ydisp_attrs = {'Description':'Northward displacement of the platform '\
                                +'measured with GPS-based method',
                   'Units':'Meter'}
    zdisp_attrs = {'Description':'Upward displacement of the platform '\
                                +'measured with GPS-based method',
                   'Units':'Meter'}
    lon_attrs   = {'Description':'Longitude of the buoy',
                   'Units':'Decimal degree North'}
    lat_attrs   = {'Description':'Latitude of the buoy',
                   'Units':'Decimal degree West'}

    # Quality flag attributes
    qf_xdisp_attrs = {'Description':'Quality flag for variable '\
                                   +'"Displacement_X"'}
    qf_ydisp_attrs = {'Description':'Quality flag for variable '\
                                   +'"Displacement_Y"'}
    qf_zdisp_attrs = {'Description':'Quality flag for variable '\
                                   +'"Displacement_Z"'}

    # "xarray" data output
    Dim1         = ['Time']
    Dim2         = ['Time','Time_Range']
    Crd1         = {'Time':reg_date}
    Crd2         = {'Time':reg_date,'Time_Range':time_range}
    xdisp_out    = xr.DataArray(xdata,dims=Dim2,coords=Crd2,attrs=xdisp_attrs)
    ydisp_out    = xr.DataArray(ydata,dims=Dim2,coords=Crd2,attrs=ydisp_attrs)
    zdisp_out    = xr.DataArray(zdata,dims=Dim2,coords=Crd2,attrs=zdisp_attrs)
    lon_out      = xr.DataArray(lon,dims=Dim1,coords=Crd1,attrs=lon_attrs)
    lat_out      = xr.DataArray(lat,dims=Dim1,coords=Crd1,attrs=lat_attrs)
    qf_xdisp_out = xr.DataArray(qf_xdata,
                                dims=Dim1,coords=Crd1,attrs=qf_xdisp_attrs)
    qf_ydisp_out = xr.DataArray(qf_ydata,
                                dims=Dim1,coords=Crd1,attrs=qf_ydisp_attrs)
    qf_zdisp_out = xr.DataArray(qf_zdata,
                                dims=Dim1,coords=Crd1,attrs=qf_zdisp_attrs)

    # Create output dataset
    DSout = xr.Dataset({'Longitude':lon_out,
                        'Latitude':lat_out,
                        'Displacement_X':xdisp_out,
                        'Displacement_Y':ydisp_out,
                        'Displacement_Z':zdisp_out,
                        'Displacement_X_QF':qf_xdisp_out,
                        'Displacement_Y_QF':qf_ydisp_out,
                        'Displacement_Z_QF':qf_zdisp_out})

    # Auxiliary variables
    DSout['Id']          = lvl_d['Info']['Id']
    DSout['Fs']          = lvl_d['Info']['Sampling_Frequency']
    DSout['Water_Depth'] = lvl_d['Physics_Parameters']['Water_Depth']

    # Auxiliary attributes
    DSout.Id.attrs          = {'Description':'Buoy ID'}
    DSout.Fs.attrs          = {'Description':'Sampling frequency of surface '\
                                            +'motions',
                               'Units':'Hertz'}
    DSout.Water_Depth.attrs = {'Description':'Water column depth',
                               'Units':'Meter'}
    DSout.Time.attrs        = {'Description':'Start timestamp of 30-minute '\
                                            +'records',
                               'Units':'Timestamp'}
    DSout.Time_Range.attrs  = {'Description':'Time range of regularly '\
                                            +'sampled 30-minute records',
                               'Units':'Second'}

    # Write NetCDF
    sh('rm %s'%lvl_d['Output']['LVL0_File'])
    DSout.to_netcdf(lvl_d['Output']['LVL0_File'],engine='netcdf4')

# ---------- #
def writeLvl1(lvl_d,qflt_d):
    '''
    Write level 1 (wave) of Spotter buoy.
    '''

    # Unpack 'lvl_d'
    buoy      = lvl_d['Info']['Id']
    rec_len   = lvl_d['Info']['Wave_Record_Length']
    reg_dt    = lvl_d['Info']['Aux_Regular_Length']
    fs        = lvl_d['Info']['Sampling_Frequency']
    lvl0_file = lvl_d['Output']['LVL0_File']
    H         = lvl_d['Physics_Parameters']['Water_Depth']
    zpos      = lvl_d['Wave_Monitor']['Z_Position']
    freq_min  = lvl_d['Wave_Monitor']['Freq_Max']
    freq_max  = lvl_d['Wave_Monitor']['Freq_Min']
    filt_bool = lvl_d['Filtering']['Filter']
    f_type    = lvl_d['Filtering']['F_Type']
    d_type    = lvl_d['Filtering']['D_Type']

    # Retrieve filtering information
    if d_type=='length':
        wcut = 2*pi/lvl_d['Filtering']['C0']
        fcut = getFrequency(wcut,H)
    elif d_type=='freq':
        fcut = lvl_d['Filtering']['C0']
        wcut = getWavenumber(fcut,H)
    if filt_bool:
        fpass = 'high pass' if f_type=='hp' else\
                'low pass'  if f_type=='lp' else\
                ''
    else: fpass = 'no'

    # Calculate record length in time index
    irec_len = int(rec_len*fs)

    # Compute frequencies, angular velocities and theoretical wavenumbers
    freq = np.fft.rfftfreq(irec_len,d=1/fs)
    om   = 2*pi*freq
    wnum = hstack([0,[getWavenumber(o,H) for o in om[1:]]])

    # Compute ButtherWorth filter if specified
    bwfilt = getBWFilter(freq,lvl_d['Filtering'])

    # Open level 0 acceleration data
    DS0       = xr.open_dataset(lvl0_file,engine='netcdf4')
    lvl0_date = DS0.Time.values
    xdisp     = DS0.Displacement_X.values
    ydisp     = DS0.Displacement_Y.values
    zdisp     = DS0.Displacement_Z.values
    qf_x      = DS0.Displacement_X_QF.values
    qf_y      = DS0.Displacement_Y_QF.values
    qf_z      = DS0.Displacement_Z_QF.values
    dim       = len(lvl0_date)

    # Initialize outputs
    sxx,qf_sxx             = [[] for _ in range(2)]
    syy,qf_syy             = [[] for _ in range(2)]
    szz,qf_szz             = [[] for _ in range(2)]
    cxy,qf_cxy             = [[] for _ in range(2)]
    qxz,qf_qxz             = [[] for _ in range(2)]
    qyz,qf_qyz             = [[] for _ in range(2)]
    a1,qf_a1               = [[] for _ in range(2)]
    b1,qf_b1               = [[] for _ in range(2)]
    a2,qf_a2               = [[] for _ in range(2)]
    b2,qf_b2               = [[] for _ in range(2)]
    freq_peak,qf_freq_peak = [[] for _ in range(2)]
    wnum_peak,qf_wnum_peak = [[] for _ in range(2)]
    hm0,tmn10,tm01,tm02    = [[] for _ in range(4)]
    theta_mean,theta_peak  = [[] for _ in range(2)]
    sigma_mean,sigma_peak  = [[] for _ in range(2)]

    ## MAIN ITERATION
    print('\nParsing Level 1 for %s...'%buoy)
    progress = []

    # Iterate over "lvl0" data
    for i in range(dim):
        # Progress
        iprog    = int(i/(dim-1)*20)
        progress = printProgress(iprog,progress)

        # Compute cross-spectral densities by averaging cross quantities from
        # two previous rectified accelerations
        CSxx = getCSD(xdisp[i],xdisp[i],1/fs)
        CSyy = getCSD(ydisp[i],ydisp[i],1/fs)
        CSzz = getCSD(zdisp[i],zdisp[i],1/fs)
        CSxy = getCSD(xdisp[i],ydisp[i],1/fs)
        CSxz = getCSD(xdisp[i],zdisp[i],1/fs)
        CSyz = getCSD(ydisp[i],zdisp[i],1/fs)

        # Apply filter
        CSxx*= bwfilt
        CSyy*= bwfilt
        CSzz*= bwfilt
        CSxy*= bwfilt
        CSxz*= bwfilt
        CSyz*= bwfilt

        # Compute and apply transfer function for accelerations
        h_x  = 1
        h_y  = 1
        h_z  = 1
        CSxx/= h_x*np.conj(h_x)
        CSyy/= h_y*np.conj(h_y)
        CSzz/= h_z*np.conj(h_z)
        CSxy/= h_x*np.conj(h_y)
        CSxz/= h_x*np.conj(h_z)
        CSyz/= h_y*np.conj(h_z)

        # Crop to operator specification
        i_crop       = where((freq>=freq_min)*(freq<=freq_max))[0]
        CSxx[i_crop] = 0
        CSyy[i_crop] = 0
        CSzz[i_crop] = 0
        CSxy[i_crop] = 0
        CSxz[i_crop] = 0
        CSyz[i_crop] = 0

        # Pack cross-spectral densities to a list
        CS = [CSxx,CSyy,CSzz,CSxy,CSxz,CSyz]

        # Floor zero values of the cross-spectral densities
        for cs in CS: cs[where(abs(cs)<=0)] = 0

        # Compute the wave variance spectrum
        Ef = np.abs(CSzz)

        # Compute directional moments
        A1,B1,A2,B2 = getDirMoments(CS)

        # Append spectral variables
        sxx.append(np.abs(CSxx))
        syy.append(np.abs(CSyy))
        szz.append(np.abs(CSzz))
        cxy.append(np.real(CSxy))
        qxz.append(-np.imag(CSxz))
        qyz.append(-np.imag(CSyz))
        a1.append(A1)
        b1.append(B1)
        a2.append(A2)
        b2.append(B2)

        ## QUALITY FLAGS FOR SPECTRAL VARIABLES
        qf_xy  = getQFCombined(qf_x[i],qf_y[i])
        qf_xz  = getQFCombined(qf_x[i],qf_z[i])
        qf_yz  = getQFCombined(qf_y[i],qf_z[i])
        qf_xyz = getQFCombined(qf_x[i],getQFCombined(qf_y[i],qf_z[i]))
        qf_sxx.append(qf_x[i])
        qf_syy.append(qf_y[i])
        qf_szz.append(qf_z[i])
        qf_cxy.append(qf_xy)
        qf_qxz.append(qf_xz)
        qf_qyz.append(qf_yz)
        qf_a1.append(qf_xyz)
        qf_b1.append(qf_xyz)
        qf_a2.append(qf_xy)
        qf_b2.append(qf_xy)
        qf_freq_peak.append(qf_z[i])
        qf_wnum_peak.append(qf_z[i])

        ## BULK WAVE VARIABLES
        # Compute -1,0,1,2 frequency moments
        mn1 = getFreqMoment(Ef,freq,-1)
        m0  = getFreqMoment(Ef,freq,0)
        m1  = getFreqMoment(Ef,freq,1)
        m2  = getFreqMoment(Ef,freq,2)

        # Compute weighted directional moments
        A1_mean,B1_mean,A2_mean,B2_mean =\
        getDirMoments(CS,weight=True,Ef=Ef,fs=np.diff(freq)[0])

        # Compute peak variables and swap angles to true north provenance
        if ~isnan(abs(Ef)).all():
            ifmax  = np.nanargmax(abs(Ef))
            A1_max = A1[ifmax]
            B1_max = B1[ifmax]
            fp     = freq[ifmax]
            wp     = wnum[ifmax]
            tp     = (3*pi/2-np.arctan2(B1_max,A1_max))%(2*pi)
            sp     = (2*(1-(A1_max**2+B1_max**2)**(1/2)))**(1/2)
        else: fp,wp,tp,sp = [np.nan for _ in range(4)]

        # Compute mean variables and swap angles to true north provenance
        tm = (3*pi/2-np.arctan2(B1_mean,A1_mean))%(2*pi)
        sm = (2*(1-(A1_mean**2+B1_mean**2)**(1/2)))**(1/2)

        # Append bulk wave variables
        hm0.append(4*m0**(1/2))                 # significant wave height
        tmn10.append(mn1/m0)                    # energy wave period
        tm01.append(m0/m1)                      # mean wave period
        tm02.append(np.sqrt(m0/m2))             # absolute mean wave period
        freq_peak.append(fp)                    # peak frequency
        wnum_peak.append(wp)                    # peak wavenumber
        theta_mean.append(180/pi*tm)            # mean direction
        theta_peak.append(180/pi*tp)            # peak direction
        sigma_mean.append(180/pi*sm)            # mean directional spreading
        sigma_peak.append(180/pi*sp)            # peak directional spreading

    ## QUALITY FLAGS FOR BULK PARAMETERS
    # Update "qflt_d" for tests 14 and 17
    # 'fval' for test 14 defines frequencies to validate, here we choose mean
    # wave frequency and peak frequency
    qflt_d['Test_14']['wnum'] = wnum
    qflt_d['Test_14']['freq'] = freq
    qflt_d['Test_14']['CS']   = transpose([array(sxx),array(syy),array(szz)])
    qflt_d['Test_14']['fval'] = transpose([1/array(tm01),array(freq_peak)])
    qflt_d['Test_17']['freq'] = freq
    qflt_d['Test_17']['CS']   = transpose([array(sxx),array(syy),array(szz)])

    # Compute quality flags for "Hm0" and update result for test 19
    hm0,qf_hm0,qflt_d = getLTQF(hm0,qf_z,'Hm0',qflt_d)

    # Compute quality flags for other wave bulk parameters
    tmn10,qf_tmn10,_           = getLTQF(tmn10,qf_z,'Tm-10',qflt_d)
    tm01,qf_tm01,_             = getLTQF(tm01,qf_z,'Tm01',qflt_d)
    tm02,qf_tm02,_             = getLTQF(tm02,qf_z,'Tm02',qflt_d)
    theta_mean,qf_theta_mean,_ = getLTQF(theta_mean,qf_a1,'Theta_Mean',qflt_d)
    theta_peak,qf_theta_peak,_ = getLTQF(theta_peak,qf_a1,'Theta_Peak',qflt_d)
    sigma_mean,qf_sigma_mean,_ = getLTQF(sigma_mean,qf_a1,'Sigma_Mean',qflt_d)
    sigma_peak,qf_sigma_peak,_ = getLTQF(sigma_peak,qf_a1,'Sigma_Peak',qflt_d)

    ## OUTPUTS
    # Spectral variables attributes
    sxx_attrs = {'Description':'"XX" auto cross-spectral density',
                 'Units':'Meter squared per Hertz'}
    syy_attrs = {'Description':'"YY" auto cross-spectral density',
                 'Units':'Meter squared per Hertz'}
    szz_attrs = {'Description':'"ZZ" auto cross-spectral density',
                 'Units':'Meter squared per Hertz'}
    cxy_attrs = {'Description':'"XY" coincident cross-spectral density',
                 'Units':'Meter squared per Hertz'}
    qxz_attrs = {'Description':'"XZ" quadrature cross-spectral density',
                 'Units':'Meter squared per Hertz'}
    qyz_attrs = {'Description':'"YZ" quadrature cross-spectral density',
                 'Units':'Meter squared per Hertz'}
    a1_attrs  = {'Description':'"a1" directional moment','Units':'None'}
    b1_attrs  = {'Description':'"b1" directional moment','Units':'None'}
    a2_attrs  = {'Description':'"a2" directional moment','Units':'None'}
    b2_attrs  = {'Description':'"b2" directional moment','Units':'None'}

    # Spectral variables quality flag attributes
    qf_sxx_attrs = {'Description':'Quality flag for variable '+\
                    '"Sxx"'}
    qf_syy_attrs = {'Description':'Quality flag for variable '+\
                    '"Syy"'}
    qf_szz_attrs = {'Description':'Quality flag for variable '+\
                    '"Szz"'}
    qf_cxy_attrs = {'Description':'Quality flag for variable '+\
                    '"Cxy"'}
    qf_qxz_attrs = {'Description':'Quality flag for variable '+\
                    '"Qxz"'}
    qf_qyz_attrs = {'Description':'Quality flag for variable '+\
                    '"Qyz"'}
    qf_a1_attrs  = {'Description':'Quality flag for variable '+\
                    '"A1"'}
    qf_b1_attrs  = {'Description':'Quality flag for variable '+\
                    '"B1"'}
    qf_a2_attrs  = {'Description':'Quality flag for variable '+\
                    '"A2"'}
    qf_b2_attrs  = {'Description':'Quality flag for variable '+\
                    '"B2"'}

    # "xarray" data output for spectral variables
    Dim1       = ['Time']
    Dim2       = ['Time','Frequency']
    Crd1       = {'Time':lvl0_date}
    Crd2       = {'Time':lvl0_date,'Frequency':freq}
    sxx_out    = xr.DataArray(sxx,dims=Dim2,coords=Crd2,attrs=sxx_attrs)
    syy_out    = xr.DataArray(syy,dims=Dim2,coords=Crd2,attrs=syy_attrs)
    szz_out    = xr.DataArray(szz,dims=Dim2,coords=Crd2,attrs=szz_attrs)
    cxy_out    = xr.DataArray(cxy,dims=Dim2,coords=Crd2,attrs=cxy_attrs)
    qxz_out    = xr.DataArray(qxz,dims=Dim2,coords=Crd2,attrs=qxz_attrs)
    qyz_out    = xr.DataArray(qyz,dims=Dim2,coords=Crd2,attrs=qyz_attrs)
    a1_out     = xr.DataArray(a1,dims=Dim2,coords=Crd2,attrs=a1_attrs)
    b1_out     = xr.DataArray(b1,dims=Dim2,coords=Crd2,attrs=b1_attrs)
    a2_out     = xr.DataArray(a2,dims=Dim2,coords=Crd2,attrs=a2_attrs)
    b2_out     = xr.DataArray(b2,dims=Dim2,coords=Crd2,attrs=b2_attrs)
    qf_sxx_out = xr.DataArray(qf_sxx,dims=Dim1,coords=Crd1,attrs=qf_sxx_attrs)
    qf_syy_out = xr.DataArray(qf_syy,dims=Dim1,coords=Crd1,attrs=qf_syy_attrs)
    qf_szz_out = xr.DataArray(qf_szz,dims=Dim1,coords=Crd1,attrs=qf_szz_attrs)
    qf_cxy_out = xr.DataArray(qf_cxy,dims=Dim1,coords=Crd1,attrs=qf_cxy_attrs)
    qf_qxz_out = xr.DataArray(qf_qxz,dims=Dim1,coords=Crd1,attrs=qf_qxz_attrs)
    qf_qyz_out = xr.DataArray(qf_qyz,dims=Dim1,coords=Crd1,attrs=qf_qyz_attrs)
    qf_a1_out  = xr.DataArray(qf_a1,dims=Dim1,coords=Crd1,attrs=qf_a1_attrs)
    qf_b1_out  = xr.DataArray(qf_b1,dims=Dim1,coords=Crd1,attrs=qf_b1_attrs)
    qf_a2_out  = xr.DataArray(qf_a2,dims=Dim1,coords=Crd1,attrs=qf_a2_attrs)
    qf_b2_out  = xr.DataArray(qf_b2,dims=Dim1,coords=Crd1,attrs=qf_b2_attrs)

    # Create output dataset for spectral variables
    DSout = xr.Dataset({'Sxx':sxx_out,
                        'Syy':syy_out,
                        'Szz':szz_out,
                        'Cxy':cxy_out,
                        'Qxz':qxz_out,
                        'Qyz':qyz_out,
                        'A1':a1_out,
                        'B1':b1_out,
                        'A2':a2_out,
                        'B2':b2_out,
                        'Sxx_QF':qf_sxx_out,
                        'Syy_QF':qf_syy_out,
                        'Szz_QF':qf_szz_out,
                        'Cxy_QF':qf_cxy_out,
                        'Qxz_QF':qf_qxz_out,
                        'Qyz_QF':qf_qyz_out,
                        'A1_QF':qf_a1_out,
                        'B1_QF':qf_b1_out,
                        'A2_QF':qf_a2_out,
                        'B2_QF':qf_b2_out})

    # Auxiliary variables
    DSout['Id']                = lvl_d['Info']['Id']
    DSout['Fs']                = fs
    DSout['Water_Depth']       = H
    DSout['Wavenumber']        = wnum
    DSout['Cutoff_Frequency']  = fcut
    DSout['Cutoff_Wavenumber'] = wcut

    # Auxiliary attributes
    DSout.Id.attrs                = {'Description':'Buoy ID'}
    DSout.Fs.attrs                = {'Description':'Sampling frequency of '\
                                                  +'surface motions',
                                     'Units':'Hertz'}

    DSout.Water_Depth.attrs       = {'Description':'Water column depth',
                                     'Units':'Meter'}
    DSout.Cutoff_Frequency.attrs  = {'Description':'Cutoff frequency for '\
                                                  +'%s filtering of '%fpass\
                                                  +'cross-spectral densities',
                                     'Units':'Hertz'}
    DSout.Cutoff_Wavenumber.attrs = {'Description':'Cutoff wavenumber for '\
                                                  +'%s filtering of '%fpass\
                                                  +'cross-spectral densities',
                                     'Units':'Radian per meter'}
    DSout.Time.attrs              = {'Description':'Start timestamp of '\
                                                  +'spectral variable '\
                                                  +'provided every 30 minutes',
                                     'Units':'Timestamp'}
    DSout.Frequency.attrs         = {'Description':'Frequency bins',
                                     'Units':'Hertz'}
    DSout.Wavenumber.attrs        = {'Description':'Wavenumber bins '\
                                                  +'calculated using the '\
                                                  +'dispersion relation for '\
                                                  +'surface gravity waves',
                                     'Units':'Radian per meter'}

    # Write NetCDF for spectral variables
    sh('rm %s'%lvl_d['Output']['LVL1_File'][0])
    DSout.to_netcdf(lvl_d['Output']['LVL1_File'][0],engine='netcdf4')

    # Bulk wave variables attributes
    hm0_attrs        = {'Description':'Significant wave height',
                        'Units':'Meter'}
    tmn10_attrs      = {'Description':'Energy wave period',
                        'Units':'Second'}
    tm01_attrs       = {'Description':'Wave mean period',
                        'Units':'Second'}
    tm02_attrs       = {'Description':'Absolute wave mean period',
                        'Units':'Second'}
    freq_peak_attrs  = {'Description':'Wave peak frequency',
                        'Units':'Hertz'}
    wnum_peak_attrs  = {'Description':'Wave peak wavenumber',
                        'Units':'Radian per meter'}
    theta_mean_attrs = {'Description':'Wave mean provenance',
                        'Units':'True north degree'}
    theta_peak_attrs = {'Description':'Wave peak provenance',
                        'Units':'True north degree'}
    sigma_mean_attrs = {'Description':'Mean directional spreading',
                        'Units':'Degree'}
    sigma_peak_attrs = {'Description':'Peak directional spreading',
                        'Units':'Degree'}

    # Bulk wave variables quality flag attributes
    qf_hm0_attrs        = {'Description':'Quality flag for variable '+\
                           '"Hm0"'}
    qf_tmn10_attrs      = {'Description':'Quality flag for variable '+\
                           '"Tm-10"'}
    qf_tm01_attrs       = {'Description':'Quality flag for variable '+\
                           '"Tm01"'}
    qf_tm02_attrs       = {'Description':'Quality flag for variable '+\
                           '"Tm02"'}
    qf_freq_peak_attrs  = {'Description':'Quality flag for variable '+\
                           '"Freq_Peak"'}
    qf_wnum_peak_attrs  = {'Description':'Quality flag for variable '+\
                           '"Wnum_Peak"'}
    qf_theta_mean_attrs = {'Description':'Quality flag for variable '+\
                           '"Theta_Mean"'}
    qf_theta_peak_attrs = {'Description':'Quality flag for variable '+\
                           '"Theta_Peak"'}
    qf_sigma_mean_attrs = {'Description':'Quality flag for variable '+\
                           '"Sigma_Mean"'}
    qf_sigma_peak_attrs = {'Description':'Quality flag for variable '+\
                           '"Sigma_Peak"'}

    # "xarray" data output for bulk wave variables
    Dim               = ['Time']
    Crd               = {'Time':lvl0_date}
    hm0_out           = xr.DataArray(hm0,dims=Dim,coords=Crd,
                                     attrs=hm0_attrs)
    tmn10_out         = xr.DataArray(tmn10,dims=Dim,coords=Crd,
                                     attrs=tmn10_attrs)
    tm01_out          = xr.DataArray(tm01,dims=Dim,coords=Crd,
                                     attrs=tm01_attrs)
    tm02_out          = xr.DataArray(tm02,dims=Dim,coords=Crd,
                                     attrs=tm02_attrs)
    freq_peak_out     = xr.DataArray(freq_peak,dims=Dim,coords=Crd,
                                     attrs=freq_peak_attrs)
    wnum_peak_out     = xr.DataArray(wnum_peak,dims=Dim,coords=Crd,
                                     attrs=wnum_peak_attrs)
    theta_mean_out    = xr.DataArray(theta_mean,dims=Dim,coords=Crd,
                                     attrs=theta_mean_attrs)
    theta_peak_out    = xr.DataArray(theta_peak,dims=Dim,coords=Crd,
                                     attrs=theta_peak_attrs)
    sigma_mean_out    = xr.DataArray(sigma_mean,dims=Dim,coords=Crd,
                                     attrs=sigma_mean_attrs)
    sigma_peak_out    = xr.DataArray(sigma_peak,dims=Dim,coords=Crd,
                                     attrs=sigma_peak_attrs)
    qf_hm0_out        = xr.DataArray(qf_hm0,dims=Dim,coords=Crd,
                                     attrs=qf_hm0_attrs)
    qf_tmn10_out      = xr.DataArray(qf_tmn10,dims=Dim,coords=Crd,
                                     attrs=qf_tmn10_attrs)
    qf_tm01_out       = xr.DataArray(qf_tm01,dims=Dim,coords=Crd,
                                     attrs=qf_tm01_attrs)
    qf_tm02_out       = xr.DataArray(qf_tm02,dims=Dim,coords=Crd,
                                     attrs=qf_tm02_attrs)
    qf_freq_peak_out  = xr.DataArray(qf_freq_peak,dims=Dim,coords=Crd,
                                     attrs=qf_freq_peak_attrs)
    qf_wnum_peak_out  = xr.DataArray(qf_wnum_peak,dims=Dim,coords=Crd,
                                     attrs=qf_wnum_peak_attrs)
    qf_theta_mean_out = xr.DataArray(qf_theta_mean,dims=Dim,coords=Crd,
                                     attrs=qf_theta_mean_attrs)
    qf_theta_peak_out = xr.DataArray(qf_theta_peak,dims=Dim,coords=Crd,
                                     attrs=qf_theta_peak_attrs)
    qf_sigma_mean_out = xr.DataArray(qf_sigma_mean,dims=Dim,coords=Crd,
                                     attrs=qf_sigma_mean_attrs)
    qf_sigma_peak_out = xr.DataArray(qf_sigma_peak,dims=Dim,coords=Crd,
                                     attrs=qf_sigma_mean_attrs)

    # Auxiliary variables
    DSout['Id'] = lvl_d['Info']['Id']

    # Auxiliary attributes
    DSout.Id.attrs = {'Description':'Buoy ID'}

    # Create output dataset for bulk wave variables
    DSout = xr.Dataset({'Hm0':hm0_out,
                        'Tm-10':tmn10_out,
                        'Tm01':tm01_out,
                        'Tm02':tm02_out,
                        'Frequency_Peak':freq_peak_out,
                        'Wavenumber_Peak':wnum_peak_out,
                        'Theta_Mean':theta_mean_out,
                        'Theta_Peak':theta_peak_out,
                        'Sigma_Mean':sigma_mean_out,
                        'Sigma_Peak':sigma_peak_out,
                        'Hm0_QF':qf_hm0_out,
                        'Tm-10_QF':qf_tmn10_out,
                        'Tm01_QF':qf_tm01_out,
                        'Tm02_QF':qf_tm02_out,
                        'Frequency_Peak_QF':qf_freq_peak_out,
                        'Wavenumber_Peak_QF':qf_wnum_peak_out,
                        'Theta_Mean_QF':qf_theta_mean_out,
                        'Theta_Peak_QF':qf_theta_peak_out,
                        'Sigma_Mean_QF':qf_sigma_mean_out,
                        'Sigma_Peak_QF':qf_sigma_peak_out})

    # Write NetCDF for bulk wave variables
    sh('rm %s'%lvl_d['Output']['LVL1_File'][1])
    DSout.to_netcdf(lvl_d['Output']['LVL1_File'][1],engine='netcdf4')

# END
