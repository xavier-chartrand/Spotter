#!/usr/bin/python -u
# Author: Xavier Chartrand
# Email : x.chartrand@protonmail.me
#         xavier.chartrand@ec.gc.ca
#         xavier.chartrand@uqar.ca

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
from copy import deepcopy
from numpy import arange,arctan2,array,complex128,ceil,conj,copy,cos,cosh,\
                  diff,exp,floor,hstack,imag,isnan,mean,nan,nanargmax,\
                  nancumsum,ones,real,sin,sinh,shape,sqrt,tan,tanh,where
from numpy.fft import rfft,rfftfreq
from scipy.signal.windows import get_window as getWindow
from scipy.special import erfcinv
from scipy.stats import median_abs_deviation as mad
from xarray import DataArray,Dataset,open_dataset
# Constants
from scipy.constants import pi
# Custom utilities
from qf_utils import *

### Shell commands in python
# ----------
def sh(s): os.system("bash -c '%s'"%s)

### "spot_utils" utilities
# ----------
def getSPOTLocFromFileList(f_list,hrows=8):
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
        MSS = [int(ceil(mss/100)*100) for mss in DS[:,6]]

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

# ----------
def getSPOTTimeFromFile(f_name,hrows=8):
    '''
    Retrieve date from a SPOT displacement or location data file.
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
    MSS = [int(ceil(mss/100)*100) for mss in DS[:,6]]

    # Append to time list
    time_list = [(Y[i],M[i],D[i],HH[i],MM[i],SS[i],MSS[i])\
                 for i in range(shape(DS)[0])]

    return time_list

# ----------
def getBWFilter(freq,filt_p,n=5,g=9.81):
    '''
    Compute low-or-high pass ButtherWorth for spectral variables
    (default order: n=5).

    "F_Type"    is the filter type, low pass 'lp', high pass 'hp' ;
    "D_Type"    is the cutoff type, wavenumber 'wnum', frequency 'freq' ;
    "C0"        is the cutoff value.
    '''

    # Unpack "filt_p"
    bool_filt = filt_p['Filter']
    ftype     = filt_p['F_Type']
    dtype     = filt_p['D_Type']
    c0        = filt_p['C0']
    H         = filt_p['H'] if 'H' in filt_p.keys() else 0

    # Return "ones" if no filtering specified
    if not bool_filt: return ones(len(freq))

    # Raise error if water's depth is missing for 'dtype'=='wnum'
    if dtype=='wnum' and not H:
        raise TypeError("'H' missing for data type 'wnum'")

    fsgn = 1 if ftype=='lp' else -1 if ftype=='hp' else 0

    # Swap cutoff wavenumner to frequency if specified, using the dispersion
    # relation for linear surface gravity waves
    c0 = (tanh(2*pi*H/c0)*g/2/pi/c0)**(0.5) if dtype=='wnum' else c0

    # Generate Butterworth filter
    filt = fsgn/(1 + (freq/freq[abs(freq-c0).argmin()])**(2*n))**(0.5)\
         + (1 - fsgn)/2

    return filt

# ----------
def getCSD(X1,X2,dsmp,win='hann'):
    '''
    Compute the cross-spectral density between time series "X1" and "X2".

    "dsmp"      is the sampling period ;
    "win"       is the windowing function.
    '''

    # Compute series length and total time
    lX = len(X1)
    T  = lX*dsmp

    # Get window
    try:W=getWindow(win,lX)
    except:W=ones(lX)

    # Right-sided, normalized Fourier transform of windowed series
    F1 = rfft(W*X1)/lX
    F2 = rfft(W*X2)/lX

    # Cross-spectral density times windowing normalization factors
    # (normalization complies with 'scipy.signals.csd')
    qfac = 2*hstack([1/2,ones(lX//2)]) if lX%2 else\
           2*hstack([1/2,ones(lX//2-1),1/2])    # right-sided norm factor
    wfac = 1/mean(W**2)                         # window norm factor
    nfac = T*qfac*wfac                          # CSD norm factor

    # Compute cross-spectrum
    S = nfac*F1*conj(F2)

    return S

# ----------
def getDirMoments(CS,weight=False,Ef=None,fs=None):
    '''
    Compute directional moments (Fourier coefficients) up to the 2nd order.
    Directional moments may be normalized by the total available energy, if the
    variance wave spectrum and the sampling frequency are given.

    "CS" must be expressed as "CS=[S_xx,S_yy,S_zz,C_xy,Q_xz,Q_yz]".
    The convention for coincident (co) and quadrature (quad) spectra is
        cross = co - 1j*quad = real(cross) -1j*imag(cross)
    '''

    # Retrieve auto, coincident and quadrature spectra
    Sxx,Syy,Szz,Cxy,Qxz,Qyz = [cs for cs in CS]

    # Normalize spectra if specified
    if weight and type(Ef)!=type(None) and type(fs)!=type(None):
        Sxx = getWeightedParam(Sxx,Ef,fs)
        Syy = getWeightedParam(Syy,Ef,fs)
        Szz = getWeightedParam(Szz,Ef,fs)
        Cxy = getWeightedParam(Cxy,Ef,fs)
        Qxz = getWeightedParam(Qxz,Ef,fs)
        Qyz = getWeightedParam(Qyz,Ef,fs)
    elif weight:
        raise TypeError("Specify variance spectrum and sampling frequency")

    # Compute 1st and 2nd directional moments (Fourier coefficients)
    a1 = Qxz/sqrt(Szz*(Sxx+Syy))
    b1 = Qyz/sqrt(Szz*(Sxx+Syy))
    a2 = (Sxx - Syy)/(Sxx + Syy)
    b2 = 2*Cxy/(Sxx + Syy)

    return a1,b1,a2,b2

# ----------
def getFrequency(k,H,g=9.81):
    '''
    Estimate a frequency using the linear dispersion relation for surface
    gravity waves, for a single wavenumber "k".

    "k"         is the wavenumber ;
    "H"         is the water depth.
    '''

    return (g*k*tanh(k*H))**(0.5)/2/pi

# ----------
def getFreqMoment(Ef,freq,n):
    '''
    Compute the nth order statistical frequency moment.

    "freq"      is the frequency sampled constantly.
    '''

    return iTrapz1o(Ef*freq**n,diff(freq)[0],0)[-1]

# ----------
def getDirectionalSpectrum(phi,a1,a2,b1,b2):
    '''
    Compute the directional wave spectrum from "a1,a2,b1,b2" directional
    moments using weighted Fourier series with positive coefficients
    (Longuet-Higgins 1963).
    '''

    # Smooth-positive coefficients
    cD = [1/2/pi,2/3/pi,1/6/pi]

    return array([cD[1]*(a1*cos(p)+b1*sin(p))+\
                  cD[2]*(a2*cos(2*p)+b2*sin(2*p))
                  for p in phi]) + cD[0]

# ----------
def getWavenumber(om,H,da1=1.E3,thrs=1.E-10,g=9.81):
    '''
    Estimate a wavenumber using the linear dispersion relation for surface
    gravity waves, for a single angular velocity "om".

    "om"        is the angular velocity ;
    "H"         is the water depth.
    '''

    # Remove warnings
    warnings.filterwarnings('ignore')

    # Find roots with secant method
    a0 = om**2*H/g
    a1 = a0*tanh(a0**(3/4))**(-2/3)
    while abs(da1/a1)>thrs:
        da1 = (a0-a1*tanh(a1))/(a1*cosh(a1)**(-2)+tanh(a1))
        a1 += da1

    return a1/H

# ----------
def getWeightedParam(X,Ef,fs):
    '''
    Compute the spectrally weighted mean parameter.
    '''

    return iTrapz1o(Ef*X,fs,0)[-1]/iTrapz1o(Ef,fs,0)[-1]

# ----------
def iTrapz1o(f,dx,f0):
    '''
    Compute first order trapezoidal integral of "f(x)" on the full "x"
    interval (i.e. from "f[0]" to "f[-1]"). An initial value of "f0" can be
    specified as an integration constant.

    "dx"        is the differential supposed constant.
    '''

    return nancumsum(hstack([f0,(f[:-1]+f[1:])*dx/2]))

### WRITE LVL
# ---------- #
def writeLvl0(lvl_d,qfst_d):
    '''
    Write level 0 (surface motions) of Spotter buoy.
    '''

    ## SET
    # Unpack 'lvl_d'
    buoy           = lvl_d['Info']['Id']
    cdb            = lvl_d['Info']['Corrected_Date_Begin']
    cde            = lvl_d['Info']['Corrected_Date_End']
    rec_len        = lvl_d['Info']['Wave_Record_Length']
    wreg_dt        = lvl_d['Info']['Wave_Regular_Length']
    areg_dt        = lvl_d['Info']['Aux_Regular_Length']
    fs             = lvl_d['Info']['Sampling_Frequency']
    loc_file_list  = lvl_d['Input']['Loc_File_List']
    disp_file_list = lvl_d['Input']['Raw_File_List']
    hrows          = lvl_d['Input']['Raw_Header_Rows']
    lvl0_vars      = lvl_d['Input']['LVL0_Vars']
    afac           = lvl_d['Wave_Monitor']['Amplitude_Factor']
    xyz_ci         = lvl_d['Wave_Monitor']['XYZ_Cartesian_Index']

    # Get raw timestamps and dates from displacement files
    tfrmt     = '%d-%02g-%02gT%02g:%02g:%02g.%d'
    disp_date = [tfrmt%d
                 for file in disp_file_list\
                 for d in getSPOTTimeFromFile(file,hrows=hrows)]
    disp_ts   = array([pd.Timestamp(d).timestamp() for d in disp_date])
    file_ts   = array([pd.Timestamp(tfrmt%getSPOTTimeFromFile(file,
                                                              hrows=hrows)[0])\
                                   .timestamp()\
                       for file in disp_file_list])

    # Truncate data to retain dates within begin and end date interval
    cdb_ts         = pd.Timestamp(cdb).timestamp()
    cde_ts         = pd.Timestamp(cde).timestamp()
    idb_disp       = where((disp_ts-cdb_ts)>=0)[0][0]
    ide_disp       = where((disp_ts-cde_ts)<=0)[0][-1] + 1
    idb_file       = where((file_ts-cdb_ts)>=0)[0][0]
    ide_file       = where((file_ts-cde_ts)<=0)[0][-1] + 1
    disp_ts        = disp_ts[idb_disp:ide_disp]
    disp_file_list = disp_file_list[idb_file:ide_file]
    loc_file_list  = loc_file_list[idb_file:ide_file]

    # Get location information
    loc_ts,lon,lat = getSPOTLocFromFileList(loc_file_list)

    # Compute 30-minute regular timestamps and dates for displacement data
    reg_tsi  = floor(disp_ts[0]/wreg_dt)*wreg_dt
    reg_tsf  = ceil(disp_ts[-1]/wreg_dt)*wreg_dt
    reg_ts   = arange(reg_tsi,reg_tsf+wreg_dt,wreg_dt)
    reg_date = array([pd.Timestamp(t,unit='s') for t in reg_ts])

    # Initialize outputs
    time_range = arange(int(fs*rec_len))/fs
    data       = nan*ones((len(reg_ts),3,len(time_range)))
    location   = nan*ones((len(reg_ts),2))
    nfiles     = len(disp_file_list)
    dim_reg    = len(reg_ts)

    ## PARSE LEVEL 0
    print('\nParsing level 0 for %s: %s to %s ...'%(buoy,cdb,cde))
    progress = []

    # Retrieve and resample displacement data on regular grids
    for i in range(nfiles):
        # Print progress
        iprog    = int(i/(nfiles-1)*20)
        progress = printProgress(iprog,progress)

        # Open displacement data
        DSo = array(pd.read_csv(disp_file_list[i],
                                delimiter=',',
                                skipinitialspace=True,
                                skiprows=hrows,
                                header=None))

        # Get time information
        date = [tfrmt%t for t in getSPOTTimeFromFile(disp_file_list[i])]
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

    # Retrieve and resample location data on regular grids
    for i in range(int((reg_tsf-reg_tsi)/wreg_dt)+1):
        iloc_ts = abs(loc_ts-reg_ts[i]).argmin()
        if abs(loc_ts[iloc_ts]-reg_ts[i])<wreg_dt:
            location[i,0] = lon[iloc_ts]
            location[i,1] = lat[iloc_ts]
        else: continue
    lon,lat = location[:,0],location[:,1]

    ## QUALITY CONTROL FOR SHORT-TERM ACCELERATION TIME SERIES
    # Loop over "lvl0_vars"
    for i in range(len(lvl0_vars)):
        # Get variable and test parameters
        v       = lvl0_vars[i]
        vname   = 'Displacement_%s'%(v.upper())
        imin_11 = qfst_d['Test_11']['imin'][i]
        imax_11 = qfst_d['Test_11']['imax'][i]
        lmin_11 = qfst_d['Test_11']['lmin'][i]
        lmax_11 = qfst_d['Test_11']['lmax'][i]

        # Create test parameter dictionnary
        exec(f"global qfst_{v}; qfst_{v}=deepcopy(qfst_d)",globals(),locals())

        # Remove precalculated 'QF' if 'Do_Test' is false
        for k in qfst_d.keys():
            if k!='Test_Order':
                exec(f"if not qfst_{v}[k]['Do_Test']: qfst_{v}[k]['QF']=[]",
                     globals(),locals())

        # Update test 11
        exec(f"qfst_{v}['Test_11']['imin']=imin_11",globals(),locals())
        exec(f"qfst_{v}['Test_11']['imax']=imax_11",globals(),locals())
        exec(f"qfst_{v}['Test_11']['lmin']=lmin_11",globals(),locals())
        exec(f"qfst_{v}['Test_11']['lmax']=lmax_11",globals(),locals())

        # Compute displacement quality flag
        exec(f"global {v}_disp; global qf_{v}_disp;"+\
             f"{v}_disp,qf_{v}_disp=getSTQF(data[:,i,:],vname,qfst_{v})",
             globals(),locals())

    # Combine quality codes
    qf_disp = [getQFCombined(qf_x_disp[i],
                             getQFCombined(qf_y_disp[i],qf_z_disp[i],
                                           qfst_d['Test_Order']),
                             qfst_d['Test_Order'])\
               for i in range(len(qf_x_disp))]

    ## OUTPUTS
    # Variable attributes
    x_disp_attrs = {"Description":"Eastward displacement of the platform "\
                                 +"measured with GPS-based method",
                    "Units":"meter",
                    "QC":qf_disp,
                    "QC_Description":"Short-term primary and secondary "\
                                    +"quality code for 'x' horizontal "\
                                    +"displacement time series"}
    y_disp_attrs = {"Description":"Northward displacement of the platform "\
                                 +"measured with GPS-based method",
                    "Units":"meter",
                    "QC":qf_disp,
                    "QC_Description":"Short-term primary and secondary "\
                                    +"quality code for 'y' horizontal "\
                                    +"displacement time series"}
    z_disp_attrs = {"Description":"Upward displacement of the platform "\
                                 +"measured with GPS-based method",
                    "Units":"meter",
                    "QC_Description":"Short-term primary and secondary "\
                                    +"quality code for 'z' vertical "\
                                    +"displacement time series"}
    lon_attrs    = {"Description":"Longitude of the buoy",
                    "Units":"decimal degree north",
                    "QC":len(qf_disp)*[2],
                    "QC_Description":"Not evaluated"}
    lat_attrs    = {"Description":"Latitude of the buoy",
                    "Units":"decimal degree west",
                    "QC":len(qf_disp)*[2],
                    "QC_Description":"Not evaluated"}

    # "xarray" outputs
    Dim1 = ['Time','Time_Range']
    Dim2 = ['Time']
    Crd1 = {'Time':reg_date,'Time_Range':time_range}
    Crd2 = {'Time':reg_date}
    # Level 0, type 'Displacements'
    for v in hstack([lvl0_vars]):
        exec(f"global {v}_disp_out;"+\
             f"{v}_disp_out=DataArray({v}_disp,dims=Dim1,coords=Crd1,"+\
             f"attrs={v}_disp_attrs)",globals(),locals())
    for v in hstack(['lon','lat']):
        exec(f"global {v}_out;"+\
             f"{v}_out=DataArray({v},dims=Dim2,coords=Crd2,"+\
             f"attrs={v}_attrs)",globals(),locals())

    # Create output dataset
    DSout = Dataset({'Displacement_X':x_disp_out,
                     'Displacement_Y':y_disp_out,
                     'Displacement_Z':z_disp_out,
                     'Latitude':lat_out,
                     'Longitude':lon_out})

    # Auxiliary variables
    DSout['Id']          = lvl_d['Info']['Id']
    DSout['Fs']          = lvl_d['Info']['Sampling_Frequency']
    DSout['Water_Depth'] = lvl_d['Physics_Parameters']['Water_Depth']

    # Auxiliary attributes
    DSout.Id.attrs          = {"Description":"Buoy ID"}
    DSout.Fs.attrs          = {"Description":"Sampling frequency",
                               "Units":"Hertz"}
    DSout.Water_Depth.attrs = {"Description":"Water column depth",
                               "Units":"meter"}
    DSout.Time.attrs        = {"Description":"Starting timestamp of each "\
                                            +"30-minute displacement "\
                                            +"records",
                               "Units":"UTC"}
    DSout.Time_Range.attrs  = {"Description":"Time range of regularly sampled "
                                            +"30-minute records",
                               "Units":"second"}

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
    lvl1_vars = lvl_d['Input']['LVL1_Vars']
    lvl0_file = lvl_d['Output']['LVL0_File']
    H         = lvl_d['Physics_Parameters']['Water_Depth']
    zpos      = lvl_d['Wave_Monitor']['Z_Position']
    filt_bool = lvl_d['Filtering']['Filter']
    f_type    = lvl_d['Filtering']['F_Type']
    d_type    = lvl_d['Filtering']['D_Type']

    # Open level 0 acceleration data
    DS0       = open_dataset(lvl0_file,engine='netcdf4')
    lvl0_date = DS0.Time.values
    x_disp    = DS0.Displacement_X.values
    y_disp    = DS0.Displacement_Y.values
    z_disp    = DS0.Displacement_Z.values
    qf_disp   = DS0.Displacement_X.QC
    dim       = len(lvl0_date)

    # Compute frequencies, angular velocities and theoretical wavenumbers
    freq = rfftfreq(int(fs*rec_len),d=1/fs)
    om   = 2*pi*freq
    wnum = hstack([0,[getWavenumber(o,H) for o in om[1:]]])

    # Retrieve filtering information
    if d_type=='wnum':
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

    # Compute ButtherWorth filter if specified
    bwfilt = getBWFilter(freq,lvl_d['Filtering'])

    # Initialize outputs
    csd_nanpad = nan*ones(len(freq),dtype=complex128)
    qf_csd     = []
    for v in lvl1_vars: exec(f"global {v}; {v}=[]",globals(),locals())

    ## PARSE LEVEL 1
    print('\nParsing level 1 for %s: %s to %s ...'\
          %(buoy,lvl0_date[0],lvl0_date[-1]))
    progress = []

    # Iterate over "lvl0" data
    for i in range(dim):
        # Progress
        iprog    = int(i/(dim-1)*20)
        progress = printProgress(iprog,progress)

        # Compute cross-spectral densities
        CSxx = getCSD(x_disp[i],x_disp[i],1/fs)
        CSyy = getCSD(y_disp[i],y_disp[i],1/fs)
        CSzz = getCSD(z_disp[i],z_disp[i],1/fs)
        CSxy = getCSD(x_disp[i],y_disp[i],1/fs)
        CSxz = getCSD(x_disp[i],z_disp[i],1/fs)
        CSyz = getCSD(y_disp[i],z_disp[i],1/fs)

        # Apply filter
        CSxx*= bwfilt
        CSyy*= bwfilt
        CSzz*= bwfilt
        CSxy*= bwfilt
        CSxz*= bwfilt
        CSyz*= bwfilt

        # Compute and apply transfer function for displacements
        h_x  = 1
        h_y  = 1
        h_z  = 1
        CSxx/= h_x*conj(h_x)
        CSyy/= h_y*conj(h_y)
        CSzz/= h_z*conj(h_z)
        CSxy/= h_x*conj(h_y)
        CSxz/= h_x*conj(h_z)
        CSyz/= h_y*conj(h_z)

        # Pack cross-spectral densities to a list
        Sxx = np.abs(CSxx)
        Syy = np.abs(CSyy)
        Szz = np.abs(CSzz)
        Cxy = real(CSxy)
        Qxz = -imag(CSxz)
        Qyz = -imag(CSyz)
        CS  = [Sxx,Syy,Szz,Cxy,Qxz,Qyz]

        # Floor 0 or "NaN" values of cross-spectral densities to zeros
        for cs in CS:
            cs[where(abs(cs)<=0)] = 0
            if not all(isnan(cs)):
                cs[where(isnan(cs))] = 0

        # Compute the wave variance spectrum
        Ef = abs(CSzz)

        # Compute directional moments
        A1,B1,A2,B2 = getDirMoments(CS)

        # Append spectral variables
        sxx.append(Sxx)
        syy.append(Syy)
        szz.append(Szz)
        cxy.append(Cxy)
        qxz.append(Qxz)
        qyz.append(Qyz)
        a1.append(A1)
        b1.append(B1)
        a2.append(A2)
        b2.append(B2)

        ## QUALITY CONTROL FOR SHORT-TERM WAVE SPECTRA TIME SERIES
        # Quality control is inherited from Level 0, Type 'Displacements'
        qf_csd.append(qf_disp[i])

    ## OUTPUTS
    # Spectral variable attributes
    sxx_attrs = {"Description":"Auto cross-spectral density 'xx'",
                 "Units":"meter squared per Hertz",
                 "QC":qf_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'xx' cross-spectral density"}
    syy_attrs = {"Description":"Auto cross-spectral density 'yy'",
                 "Units":"meter squared per Hertz",
                 "QC":qf_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'yy' cross-spectral density"}
    szz_attrs = {"Description":"Auto cross-spectral density 'zz'",
                 "Units":"meter squared per Hertz",
                 "QC":qf_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'zz' cross-spectral density"}
    cxy_attrs = {"Description":"Coincident cross-spectral density 'xy'",
                 "Units":"meter squared per Hertz",
                 "QC":qf_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'xy' cross-spectral density"}
    qxz_attrs = {"Description":"Quadrature cross-spectral density 'xz'",
                 "Units":"meter squared per Hertz",
                 "QC":qf_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'xz' cross-spectral density"}
    qyz_attrs = {"Description":"Quadrature cross-spectral density 'yz'",
                 "Units":"meter squared per Hertz",
                 "QC":qf_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'yz' cross-spectral density"}
    a1_attrs  = {"Description":"Directional moment 'a1'",
                 "Units":"None",
                 "QC":qf_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'a1' directional moment"}
    b1_attrs  = {"Description":"Directional moment 'b1'",
                 "Units":"None",
                 "QC":qf_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'b1' directional moment"}
    a2_attrs  = {"Description":"Directional moment 'a2'",
                 "Units":"None",
                 "QC":qf_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'a2' directional moment"}
    b2_attrs  = {"Description":"Directional moment 'b2'",
                 "Units":"None",
                 "QC":qf_csd,
                 "QC_Description":"Short-term primary and secondary quality "\
                                 +"code for 'b2' directional moment"}

    # "xarray" outputs
    Dim = ['Time','Frequency']
    Crd = {'Time':lvl0_date,'Frequency':freq}
    for v in lvl1_vars:
        if v.split('_')[0]!='qf':
            exec(f"global {v}_out;"+\
                 f"{v}_out=DataArray({v},dims=Dim,coords=Crd,attrs={v}_attrs)",
                 globals(),locals())

    # Create output dataset for Level 1, type 'Spectral Variables'
    DSout = Dataset({'Sxx':sxx_out,
                     'Syy':syy_out,
                     'Szz':szz_out,
                     'Cxy':cxy_out,
                     'Qxz':qxz_out,
                     'Qyz':qyz_out,
                     'A1':a1_out,
                     'B1':b1_out,
                     'A2':a2_out,
                     'B2':b2_out})

    # Ancillary variables for Level 1, type 'Spectral Variables'
    DSout['Id']                = lvl_d['Info']['Id']
    DSout['Fs']                = fs
    DSout['Water_Depth']       = H
    DSout['Wavenumber']        = wnum
    DSout['Cutoff_Frequency']  = fcut
    DSout['Cutoff_Wavenumber'] = wcut

    # Ancillary attributes for Level 1, type 'Spectral Variables'
    DSout.Id.attrs                = {"Description":"Buoy ID"}
    DSout.Fs.attrs                = {"Description":"Sampling frequency of "\
                                                  +"surface motions",
                                     "Units":"Hertz"}
    DSout.Water_Depth.attrs       = {"Description":"Water column depth",
                                     "Units":"meter"}
    DSout.Cutoff_Frequency.attrs  = {"Description":"Cutoff frequency for "\
                                                  +"%s filtering of "%fpass\
                                                  +"cross-spectral densities",
                                     "Units":"Hertz"}
    DSout.Cutoff_Wavenumber.attrs = {"Description":"Cutoff wavenumber for "\
                                                  +"%s filtering of "%fpass\
                                                  +"cross-spectral densities",
                                     "Units":"cycle per meter"}
    DSout.Time.attrs              = {"Description":"Start timestamp of "\
                                                  +"spectral variable "\
                                                  +"provided every 30 minutes",
                                     "Units":"Timestamp"}
    DSout.Frequency.attrs         = {"Description":"Frequency bins",
                                     "Units":"Hertz"}
    DSout.Wavenumber.attrs        = {"Description":"Wavenumber bins "\
                                                  +"calculated using the "\
                                                  +"dispersion relation for "\
                                                  +"surface gravity waves",
                                     "Units":"cycle per meter"}

    # Write NetCDF for Level 1, type 'Spectral Variables'
    sh('rm %s'%lvl_d['Output']['LVL1_File'])
    DSout.to_netcdf(lvl_d['Output']['LVL1_File'],engine='netcdf4')

# ----------
def writeLvl2(lvl_d,qflt_d):
    '''
    Write level 2 (bulk wave parameters) of Spotter buoy.
    '''

    ## SET
    # Unpack 'lvl_d'
    buoy      = lvl_d['Info']['Id']
    fs        = lvl_d['Info']['Sampling_Frequency']
    lvl2_vars = lvl_d['Input']['LVL2_Vars']
    lvl1_file = lvl_d['Output']['LVL1_File']

    # Initialize outputs
    for v in lvl2_vars: exec(f"global {v}; {v}=[]")

    # Open level 1 wave spectra data
    DS1       = open_dataset(lvl1_file,engine='netcdf4')
    lvl1_date = DS1.Time.values
    freq      = DS1.Frequency.values
    wnum      = DS1.Wavenumber.values
    sxx       = DS1.Sxx.values
    syy       = DS1.Syy.values
    szz       = DS1.Szz.values
    cxy       = DS1.Cxy.values
    qxz       = DS1.Qxz.values
    qyz       = DS1.Qyz.values
    a1        = DS1.A1.values
    b1        = DS1.B1.values
    a2        = DS1.A2.values
    b2        = DS1.B2.values
    qf_csd    = DS1.Szz.QC
    dim       = len(lvl1_date)

    ## PARSE LEVEL 2
    print('\nParsing level 2 for %s: %s to %s ...'\
          %(buoy,lvl1_date[0],lvl1_date[-1]))
    progress = []

    # Iterate over level 1 data
    for i in range(dim):
        # Progress
        iprog    = int(i/(dim-2)*20)
        progress = printProgress(iprog,progress)

        # Retrieve variance spectrum and cross-spectral densities
        Ef = szz[i,:]
        CS = [sxx[i,:],syy[i,:],szz[i,:],cxy[i,:],qxz[i,:],qyz[i,:]]
        A1 = a1[i,:]
        B1 = b1[i,:]
        A2 = a2[i,:]
        B2 = b2[i,:]

        # Compute -1,0,1,2 frequency moments
        if all(~isnan(Ef)):
            mn1 = getFreqMoment(Ef[1:],freq[1:],-1)
            m0  = getFreqMoment(Ef[1:],freq[1:],0)
            m1  = getFreqMoment(Ef[1:],freq[1:],1)
            m2  = getFreqMoment(Ef[1:],freq[1:],2)
        else:
            mn1,m0,m1,m2 = [nan for _ in range(4)]

        # Compute weighted directional moments
        if all(~isnan(CS[0])):
            A1_mean,B1_mean,A2_mean,B2_mean =\
            getDirMoments(CS,weight=True,Ef=Ef,fs=np.diff(freq)[0])

            # Compute mean variables and convert angles to true north degrees
            _tm = (3*pi/2-arctan2(B1_mean,A1_mean))%(2*pi)
            _sm = (2*(1-(A1_mean**2+B1_mean**2)**(1/2)))**(1/2)

            # Compute peak variables and convert angles to true north degrees
            if all(~isnan(abs(Ef))):
                ifmax  = nanargmax(abs(Ef))
                A1_max = A1[ifmax]
                B1_max = B1[ifmax]
                _fp    = freq[ifmax]
                _wp    = wnum[ifmax]
                _tp    = (3*pi/2-np.arctan2(B1_max,A1_max))%(2*pi)
                _sp    = (2*(1-(A1_max**2+B1_max**2)**(1/2)))**(1/2)
            else: _fp,_wp,_tp,_sp     = [nan for _ in range(4)]
        else: _tm,_sm,_fp,_wp,_tp,_sp = [nan for _ in range(6)]

        # Append bulk wave parameters
        hm0.append(4*m0**(1/2))                 # significant wave height
        tmn10.append(mn1/m0)                    # wave energy period
        tm01.append(m0/m1)                      # mean wave period
        tm02.append(sqrt(m0/m2))                # absolute mean wave period
        fp.append(_fp)                          # peak frequency
        wp.append(_wp)                          # peak wavenumber
        tm.append(180/pi*_tm)                   # mean direction
        tp.append(180/pi*_tp)                   # peak direction
        sm.append(180/pi*_sm)                   # mean directional spreading
        sp.append(180/pi*_sp)                   # peak directional spreading

    ## QUALITY CONTROL FOR LONG-TERM BULK WAVE PARAMETERS TIME SERIES
    # Update "qflt_d" for tests 14 and 17
    # 'fv' defines frequencies to validate, here we choose peak frequency
    l2v_order                 = qflt_d['LVL2_Vars_Order']
    qflt_d['Test_14']['wnum'] = wnum
    qflt_d['Test_14']['freq'] = freq
    qflt_d['Test_14']['fv']   = fp
    qflt_d['Test_17']['freq'] = freq

    # Format LT parameters dictionnaries
    # /*
    # The variable order "l2v_order" must be the same for "qflt_d['Tickers']"
    # */
    for i in range(len(l2v_order)):
        # Get variable and test parameters
        bwp     = l2v_order[i]
        eps_16  = qflt_d['Test_16']['eps'][i]
        sf_19   = 4 if bwp=='hm0' else 3
        rmin_19 = 1/qflt_d['Test_19']['rmax'][i] if bwp=='fp'\
                  else qflt_d['Test_19']['rmin'][i]
        rmax_19 = 1/qflt_d['Test_19']['rmin'][i] if bwp=='fp'\
                  else qflt_d['Test_19']['rmax'][i]
        rmin_20 = 1/qflt_d['Test_20']['rmax'][i] if bwp=='fp'\
                  else qflt_d['Test_20']['rmin'][i]
        rmax_20 = 1/qflt_d['Test_20']['rmin'][i] if bwp=='fp'\
                  else qflt_d['Test_20']['rmax'][i]
        eps_20  = qflt_d['Test_20']['eps'][i]

        # Create test variables
        exec(f"global qflt_{bwp};"+\
             f"qflt_{bwp}=deepcopy(qflt_d)",
             globals(),locals())

        # Update test 16
        exec(f"qflt_{bwp}['Test_16']['eps']=eps_16",globals(),locals())

        # Update test 17
        if bwp in ['hm0','tmn10','tm01','tm02','fp']:
            exec(f"qflt_{bwp}['Test_17']['csd_dep']=['szz']",
                 globals(),locals())
        elif bwp in ['tm','tp','sm','sp']:
            exec(f"qflt_{bwp}['Test_17']['csd_dep']=['sxx','syy','szz']",
                 globals(),locals())
        else:
            exec(f"qflt_{bwp}['Test_17']['csd_dep']=['']",
                 globals(),locals())

        # Update test 19
        exec(f"qflt_{bwp}['Test_19']['rmin']=rmin_19",globals(),locals())
        exec(f"qflt_{bwp}['Test_19']['rmax']=rmax_19",globals(),locals())
        exec(f"qflt_{bwp}['Test_19']['set_flag']=sf_19",globals(),locals())

        # Update test 20
        exec(f"qflt_{bwp}['Test_20']['rmin']=rmin_20",globals(),locals())
        exec(f"qflt_{bwp}['Test_20']['rmax']=rmax_20",globals(),locals())
        exec(f"qflt_{bwp}['Test_20']['eps']=eps_20",globals(),locals())
        # omit test 20 for peak and directional parameters
        if bwp in ['fp','tm','tp','sm','sp']:
            exec(f"qflt_{bwp}['Test_20']['Do_Test']=False",globals(),locals())

        # Remove precalculated 'QF' if 'Do_Test' is false
        for k in qflt_d.keys():
            if k.split('_')[-1]!='Order':
                exec(f"if not qflt_{bwp}[k]['Do_Test']:qflt_{bwp}[k]['QF']=[]",
                     globals(),locals())

    # Compute quality flags
    for i in range(len(l2v_order)):
        # Get variable and name
        bwp   = l2v_order[i]
        vname = qflt_d['Tickers_Order'][i]

        # Launch quality control
        exec(f"global qf_{bwp};"+\
             f"{bwp},qf_{bwp}=getLTQF([{bwp},sxx,syy,szz],vname,qflt_{bwp})",
             globals(),locals())

        # Update tests
        for v in l2v_order[i+1:]:
            # Update test 14
            # Once test 14 is done, update test 14 for all wave parameters,
            # since the test remains the same
            if not i and qflt_d['Test_14']['Do_Test']:
                exec(f"qflt_{v}['Test_14']['QF']=qflt_{bwp}['Test_14']['QF']",
                     globals(),locals())
            else:
                exec(f"qflt_{v}['Test_14']['Do_Test']=False",
                     globals(),locals())

            # Update test 19
            exec(f"qflt_{v}['Test_19']['prev_qf']=qflt_{bwp}['Test_19']['QF']",
                 globals(),locals())

    # Recalculate QF secondary including some ST QF and test 19 results
    qf_ord = qflt_d['Test_Order']
    for v in l2v_order:
        exec(f"qflt_{v}['Test_19']['QF']=qflt_{bwp}['Test_19']['QF']",
             globals(),locals())
        exec(f"global qf_{v};"+\
             f"qf_{v}=getQFSecondary(qflt_{v})",globals(),locals())
        for i in range(len(hm0)):
            exec(f"qf_{v}[i]=getQFCombined(qf_{v}[i],qf_csd[i],qf_ord)",
                 globals(),locals())

    # Bulk wave variables attributes
    hm0_attrs   = {"Description":"Significant wave height",
                   "Units":"meter",
                   "QC":qf_hm0,
                   "QC_Description":"Long-term primary and secondary quality "\
                                   +"code for 'Hm0' significant wave height"}
    tmn10_attrs = {"Description":"Wave energy period",
                   "Units":"second",
                   "QC":qf_tmn10,
                   "QC_Description":"Long-term primary and secondary quality "\
                                   +"code for 'Tm-10' wave energy period"}
    tm01_attrs  = {"Description":"Wave mean period",
                   "Units":"second",
                   "QC":qf_tm01,
                   "QC_Description":"Long-term primary and secondary qualily "\
                                   +"code for 'Tm01' wave mean period"}
    tm02_attrs  = {"Description":"Absolute wave mean period",
                   "Units":"second",
                   "QC":qf_tm02,
                   "QC_Description":"Long-term primary and secondary quality "\
                                   +"code for 'Tm02' absolute wave mean "\
                                   +"period"}
    fp_attrs    = {"Description":"Wave peak frequency",
                   "Units":"Hertz",
                   "QC":qf_fp,
                   "QC_Description":"Long-term primary and secondary quality "\
                                   +"code for 'Frequency_Peak' wave peak "\
                                   +"frequency"}
    wp_attrs    = {"Description":"Wave peak wavenumber",
                   "Units":"cycle per meter",
                   "QC":qf_fp,
                   "QC_Description":"Long-term primary and secondary quality "\
                                   +"code for 'Wavenumber_peak' wave peak "\
                                   +"wavenumber"}
    tm_attrs    = {"Description":"Wave mean provenance",
                   "Units":"true north degree",
                   "QC":qf_tm,
                   "QC_Description":"Long-term primary and secondary quality "\
                                   +"code for 'Theta_Mean' wave mean "\
                                   +"provenance"}
    tp_attrs    = {"Description":"Wave peak provenance",
                   "Units":"true north degree",
                   "QC":qf_tp,
                   "QC_Description":"Long-term primary and secondary quality "\
                                   +"code for 'Theta_Peak' wave peak "\
                                   +"provenance"}
    sm_attrs    = {"Description":"Mean directional spreading",
                   "Units":"degree",
                   "QC":qf_sm,
                   "QC_Description":"Long-term primary and secondary quality "\
                                   +"code for 'Sigma_Mean' wave mean "\
                                   +"directional spreading"}
    sp_attrs    = {"Description":"Peak directional spreading",
                   "Units":"degree",
                   "QC":qf_sp,
                   "QC_Description":"Long-term primary and secondary quality "\
                                   +"code for 'Sigma_Peak' wave peak "\
                                   +"directional spreading"}

    # "xarray" outputs
    Dim = ['Time']
    Crd = {'Time':lvl1_date}
    for v in lvl2_vars:
        exec(f"global {v}_out;"+\
             f"{v}_out=DataArray({v},dims=Dim,coords=Crd,attrs={v}_attrs)")

    # Create output dataset for Level 2, type 'Wave Parameters'
    DSout = Dataset({'Hm0':hm0_out,
                     'Tm-10':tmn10_out,
                     'Tm01':tm01_out,
                     'Tm02':tm02_out,
                     'Frequency_Peak':fp_out,
                     'Wavenumber_Peak':wp_out,
                     'Theta_Mean':tm_out,
                     'Theta_Peak':tp_out,
                     'Sigma_Mean':sm_out,
                     'Sigma_Peak':sp_out})

    # Ancillary variables for Level 2, type 'Wave Parameters'
    DSout['Id'] = lvl_d['Info']['Id']

    # Ancillary attributes for Level 2, type 'Wave Parameters'
    DSout.Id.attrs = {"Description":"Buoy ID"}

    # Write NetCDF for Level 2, type 'Wave Parameters'
    sh('rm %s'%lvl_d['Output']['LVL2_File'])
    DSout.to_netcdf(lvl_d['Output']['LVL2_File'],engine='netcdf4')

# END
