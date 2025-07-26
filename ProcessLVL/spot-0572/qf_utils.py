#!/usr/bin/python -u
# Author: Xavier Chartrand
# Email : x.chartrand@protonmail.me
#         xavier.chartrand@ec.gc.ca

'''
Quality flag utilities based on "Manual for Real-Time Quality Control of
In-Situ Surface Wave Data", verson 2.1, February 2019.

Overview of quality flags:

1) Pass             Data have passed critical real-time QC tests and are
                    deemed adequate for use as preliminary data;

2) Not evaluated    Data have not been QC-tested, or the information on
                    quality is not available;

3) Suspect or of    Data are considered to be either suspect or of high
   high interest    interest to operators and users. They are flagged suspect
                    to draw further attention to them by operators;

4) Fail             Data are considered to have failed one or more critical
                    real-time QC checks. If they are disseminated at all, it
                    should be readily apparent that they are not of
                    acceptable quality;

9) Missing data     Data are missing (used as a placeholder).
'''

# Module
import numpy as np
import pandas as pd
import sys
import time
# Functions
from csaps import CubicSmoothingSpline as CSS
from numpy import array,isnan,invert,hstack,transpose,where
from scipy.special import erfcinv
from scipy.stats import median_abs_deviation as mad
# Constants
from scipy.constants import pi

## Short-term test functions
# ---------- #
def test9(st,N):
    '''
    Short-term time series gap.

    "N"         is the number of consecutive points allowed to be missed.

    Status: Strongly Recommended
    '''

    # Quit if all "NaN", return "st" and 9 (missing data)
    if len(where(isnan(st))[0])==len(st): return st,9

    # Get "st" dimension
    dim = len(st)

    # Check for data gap with "NaN" index
    ibad = where(isnan(st))[0]
    if len(ibad)==0: return st,1                # 1 (good) no gap
    else:                                       # check for gaps
        idiff,j = np.diff(ibad),0
        for i in range(len(idiff)):
            i,j = j if j else i,0
            while idiff[i+j]==1 and j<len(idiff):
                print(j,N)
                if j<N-1: j+= 1
                else: return st,4               # 4 (fail) at least "N" gap

    # Spline gaps if any
    x      = array(range(len(st)))
    igood  = where(invert(isnan(st)))[0]
    css_st = CSS(x[igood],st[igood],smooth=1)

    # Return splined short-term time series and 1 (good)
    return css_st.spline(x=x),1

# ---------- #
def test10_quartod(st,N,m,p=2,thrs=0.1):
    '''
    Short-term time series spike (original test).

    "N"         is the multiple of standard deviation that is not allowed to
                be exceeded ;
    "m"         is the number of points of the segment centered on "x[i]"
                that is used to replace spikes with the segment average around
                "x[i]" (x[i] is excluded from the mean calculation) ;
    "thrs"      is the fraction of the number of spikes or outliers over the
                total series length that is not allowed to be exceeded.

    Status: Strongly Recommended
    '''

    # Get "st" dimensions
    dim = len(st)

    # Quit if all "NaN", return "st" and 9 (missing data)
    if len(where(isnan(st))[0])==dim: return st,9

    # Get outliers more than "N" standard deviation
    cnt   = 0
    mst_p = np.copy(st)
    ibad  = where(abs(st-np.nanmean(st))>=N*np.nanstd(st))[0]

    # Check outliers criterion
    for _ in range(p):
        if    len(ibad)>=thrs*dim: return st,4  # 4 (fail) too many outliers
        elif  len(ibad)==0:        return st,1  # 1 (good) no outliers
        else: mst = movMean(mst_p,m,index_list=ibad,remove_i=True)

        # Check if values are "NaN"
        if len(where(isnan(mst))[0])==dim: return st,9

        # Update spike indices
        ibad  = where(abs(mst-np.nanmean(mst))>N*np.nanstd(mst))[0]
        mst_p = np.copy(mst)

    # Return 4 (failed test) if too many outliers after "P" iterations, else 1
    if len(ibad): return st,4
    else:         return mst,1

# ---------- #
def test10(st,N,thrs=0.01,spline=False):
    '''
    Short-term time series spikes (modified). If the short-term time-series
    presents at least 1% of outliers, the series is flagged suspect.

    "N"         is the multiple of standard deviation that is not allowed to
                be exceeded ;
    "thrs"      is the fraction of the number of spikes or outliers over the
                total series length that is not allowed to be exceeded.

    Status: Strongly Recommended
    '''

    # Get "st" dimensions
    dim = len(st)

    # Quit if all "NaN", return "st" and 9 (missing data)
    if len(where(isnan(st))[0])==dim: return st,9

    # Get outliers more than "N" standard deviation
    cnt   = 0
    mst_p = np.copy(st)
    ibad  = where(abs(st-np.nanmean(st))>=N*np.nanstd(st))[0]
    if len(ibad)>=thrs*len(st): return st,3
    else:                       return st,1

# ---------- #
def test11(st,imin,imax,lmin,lmax,nbins):
    '''
    Short-term time series range.

    "imin"      is the instrument minimum ;
    "imax"      is the instrument maximum ;
    "lmin"      is the regional/seasonal/climate/sensor-dependant minimum ;
    "lmax"      is the regional/seasonal/climate/sensor-dependant maximum ;
    "nbins"     is the minimal number of unique values required (no binning).

    Status: Strongly Recommended
    '''

    # Quit if all "NaN" or data are binned, return 9 (missing data)
    if len(where(isnan(st))[0])==len(st) or len(np.unique(st))<nbins: return 9

    # Check if values are in a valid range
    if   any((st<imin)*(st>imax)): return 4     # 4 (fail) out "i" range
    elif any((st<lmin)*(st>lmax)): return 3     # 3 (suspect) out "l" range
    else:                          return 1     # 1 (good) in range

# ---------- #
def test12(st,m,delta):
    '''
    Short-term time series segment shift.

    "m"         is the length of each segment ;
    "delta"     is the mean shift allowed.

    Status: Suggested
    '''

    # Quit if all "NaN", return 9 (missing data)
    if len(where(isnan(st))[0])==len(st): return 9

    # Compute moving mean over the window "m"
    mst = movMean(st,m,index_list=range(len(st)))

    # Check for segment shift
    bexp = np.diff(mst)>delta
    if any(bexp): return 4                      # 4 (fail) segment-shifted
    else:         return 1                      # 1 (good) not segment-shifted

# ---------- #
def test13(st,N,g=9.81):
    '''
    Short-term time series acceleration.

    "N"         is the fraction of the gravitational acceleration that should
                not be exceeded.

    Status: Strongly Recommended
    '''

    # Quit if all "NaN", return 9 (missing data)
    if len(where(isnan(st))[0])==len(st): return 9

    # Count bad acceleration values
    ibad = where(st>N*g)

    # Check for bad values
    if len(ibad): return 3                      # 3 (suspect) large values
    else:         return 1                      # 1 (good) small values

## Long-Terms
# ---------- #
def test14(wnum,freq,H,CS,fval,bwidth):
    '''
    Long-term time series check ratio or check factor.

    The check ratio or check factor "R(f)" is a function of frequency and
    depth and should theoretically be 1 for relatively deep water waves.
    This function is validated for small frequency bands centered on "fval"
    and of width of "bwidth", that should correspond to physically
    intepretable frequency ranges (e.g. peak period, mean period, etc).

    "wnum"      are wavenumber bins resolved ;
    "freq"      are frequency bins resolved ;
    "H"         is the water depth ;
    "CS"        are the "XX","YY" and "ZZ" cross-spectral density ;
    "fval"      is an array of frequencies to validate ;
    "bwidth"    is the half-width of the band centered on "fval".

    Status: Strongly Recommended
    '''

    # Unpack "CS"
    sxx,syy,szz = CS[:,0],CS[:,1],CS[:,2]

    # Compute check factor in frequency bands to validate
    R = []
    for fv in fval:
        i = np.where((freq>fv*(1-bwidth))*(freq<fv*(1+bwidth)))[0]
        R.append(np.mean(np.sqrt(szz[i]/(sxx[i]+syy[i]))/np.tanh(H*wnum[i])))

    # Check for values greater than 1.1 or lesser than 0.9
    R    = np.array(R)
    bexp = len(where(R<0.9)[0])+len(where(R>1.1)[0])

    # Return quality flag
    if bexp: return 3                           # 3 (suspect) check ratio
    else:    return 1                           # 1 (good) check ratio

# ---------- #
def test15(lt,N):
    '''
    Long-term time series mean and standard deviation.

    Check for values in "lt" outside a statistically valid range of "N" times
    the standard deviation around mean.

    "N"         is the number of standard deviation considered for a good
                statistical range around the mean.

    Status: Strongly Recommended
    '''

    # Get "lt" dimensions
    dim = len(lt)

    # Initialize outputs
    qf15 = 3*np.ones(dim)

    # Compute statistical limits
    lower_lim = np.nanmean(lt) - N*np.nanstd(lt)
    upper_lim = np.nanmean(lt) + N*np.nanstd(lt)
    igood     = where((lt>=lower_lim)*(lt<=upper_lim))[0]
    ibad      = where(np.invert((lt>=lower_lim)*(lt<=upper_lim)))[0]

    # Append outputs
    qf15[igood] = 1

    # Return quality flag
    return qf15

# ---------- #
def test16(lt,Ns,Nf,eps):
    '''
    Long-term time series flat line.

    Check invariate observations from sensor and/or data collection failure.

    "Ns"        is the occurence of equal data considered suspsect ;
    "Nf"        is the occurence of equal data considered fail.

    Status: Required
    '''

    # Get "lt" dimensions
    dim = len(lt)

    # Initialize output
    qf16 = np.ones(dim)

    # Iterate over values to count gaps and flat lines
    i,j = 0,1
    while j<dim:
        cnt = 1
        while abs(lt[j]-lt[i])<eps and abs(lt[j])>=eps and abs(lt[i])>=eps:
            j  += 1
            cnt+= 1
            if j==dim: break
        if   cnt>=Nf: qf16[i:j+1] = 4
        elif cnt>=Ns: qf16[i:j+1] = 3
        else:         i,j = j,j+1

    # Return quality flag
    return qf16

# ---------- #
def test17(freq,CS,imin,imax,lmin,lmax,eps=1.E-8):
    '''
    Long-term time series operational frequency range.

    Check if spectral values are measured for a valid frequency range.
    Instead of flagging failed data outside the range, they should be
    already padded with zeros as originally done by Spotter operator. However,
    suspect data are flagged.

    "freq"      are the frequencies ;
    "CS"        are the "XX", "YY" and "ZZ" cross-spectral density ;
    "imin"      is the instrument operator given minimum ;
    "imax"      is the instrument operator given maximum ;
    "lmin"      is the regional/seasonal/climate/sensor-dependant minimum ;
    "lmax"      is the regional/seasonal/climate/sensor-dependant maximum ;

    Status: Required
    '''

    # Unpack "CS"
    sxx,syy,szz = CS[:,0],CS[:,1],CS[:,2]

    # Find failed index
    ifail     = where((freq<imin)*(freq>imax))[0]
    bexp_fail = any(abs(sxx[ifail]>eps)) or\
                any(abs(syy[ifail]>eps)) or\
                any(abs(szz[ifail]>eps))

    # Find any suspect range
    isus     = where((freq<lmin)*(freq>lmax))[0]
    bexp_sus = any(abs(sxx[isus]>eps)) or\
               any(abs(syy[isus]>eps)) or\
               any(abs(szz[isus]>eps))

    # Return quality flag
    if   bexp_fail: return 4                    # 4 (fail) outside range
    elif bexp_sus:  return 3                    # 3 (suspect) suspect range
    else:           return 1                    # 1 (good) range

# ---------- #
def test18(data):
    '''
    Long-term time series low-frequency energy.

    Compare fetch and swell wave direction at low frequencies with minimum
    and maximum energy.

    Status : Required (not performed here for coastal environments)
    '''

    # Get "lt" dimensions
    dim = len(lt)

    # Initialize output
    qf18 = 2*np.ones(dim)

    # Return quality flag
    return qf18

# ---------- #
def test19(lt,rmin,rmax,vname,hm0_flag):
    '''
    Long-term time series acceptable range of bulk wave parameters.

    "rmin"      is the lower bound of acceptable range ;
    "rmax"      is the upper bound of acceptable range ;
    "vname"     is the variable tested.
    "hm0_flag"  indicates if 'Hm0' has been previously tested and failed

    "rmin" and "rmax" are either physically constrained (i.e. from 0 to 360
    for direction and from 0 to 80 for spreading) or corresponds to the
    operator lower and upper values.

    If "vname" is "Hm0" (significant wave height), and the test fails, then 1
    is additionnaly returned to flag other wave bulk parameters. Otherwise, 0
    is additionnaly returned.

    Status: Required
    '''

    # Get "lt" dimensions
    dim = len(lt)

    # Initialize outputs
    qf19 = np.ones(dim)

    # Find indices of values outside range
    ibad = hstack([where(lt<rmin)[0],where(lt>rmax)[0]])

    # Update 'hm0_flag'
    if ~len(hm0_flag): hm0_flag = np.zeros(dim)
    if vname=='Hm0':   hm0_flag[ibad] = 1

    # Update quality flag
    qf19[ibad]               = 3
    qf19[where(hm0_flag)[0]] = 4

    # Return quality flag and "hm0_flag"
    return qf19,hm0_flag

# ---------- #
def test20(lt,ibad,delta,rmin,rmax):
    '''
    Long-term time series acceptable variation of bulk wave parameters.

    "ibad"      are the indices to omit ;
    "delta"     is the acceptable variation ;
    "rmin"      is the minimum value that an instance of "lt" must have ;
    "rmax"      is the maximum value that an instance of "lt" must have ;

    Status: Required
    '''

    # Swap "lt" to array
    lt = np.array(np.copy(lt))

    # Get "lt" dimensions
    dim = len(lt)

    # Initialize output
    qf20 = np.ones(dim)

    # Pad bad values with "NaN"
    lt[ibad] = np.nan

    # Compute variation
    ivar = where(abs(np.diff(lt))>delta)[0] + 1
    imin = where(lt<=rmin)[0]
    imax = where(lt>=rmax)[0]
    ibad = []
    [ibad.append(i) if i not in imin and i not in imax else None\
     for i in ivar]

    # Append outputs
    qf20[ibad] = 4

    # Return quality flag
    return qf20

# ---------- #
def getLTQF(data,qf_spv,varname,qf_d):
    '''
    Carry all LT tests up to 20 for a given data.
    '''

    # Compute dimension
    dim = len(data)

    # Initialize 'qf' outputs
    t_14,t_15,t_16,t_17,t_18,t_19,t_20 = [2*np.ones(dim) for _ in range(7)]

    # Retrieve variable "bwp" ticker index
    i_tck,f_tck = -1,0
    for t in qf_d['Tickers']:
        i_tck+= 1
        if varname==t: f_tck=1; break

    # Retrieve data index to carry on tests
    bexp      = np.array(qf_spv)=='1.0'
    i_to_test = where(bexp)[0]
    i_failed  = where(np.invert(bexp))[0]

    # Do tests
    print('\n\nQuality control for %s'%varname)

    # Test 14
    if qf_d['Test_14']['Do_Test']:
        # Unpack test parameters
        _,wnum14,freq14,H14,CS14,fval14,bw14 = qf_d['Test_14'].values()

        # Print progress
        print('\tPerforming test 14: LT check factor...')
        progress = []
        for i in range(dim):
            # Progress
            iprog    = int(i/(dim-1)*20)
            progress = printProgress(iprog,progress)

            # Perform test
            t_14[i] = test14(wnum14,freq14,H14,CS14[:,i,:],fval14[i,:],bw14)

        print('\tTest 14 done.')
    else: print('\tTest 14 not carried out.')

    # Test 15
    if qf_d['Test_15']['Do_Test'] and varname.split('_')[0]!='Theta':
        # Unpack test parameters
        N15 = qf_d['Test_15']['N']

        # Print progress
        print('\tPerforming test 15: LT mean and standard deviation...')
        _ = printProgress(0,[])

        # Perform test
        t_15 = test15(data,N15)
        _    = printProgress(20,[0])

        print('\tTest 15 done.')
    else: print('\tTest 15 not carried out.')

    # Test 16
    if qf_d['Test_16']['Do_Test'] and f_tck:
        # Unpack test parameters
        Ns16  = qf_d['Test_16']['Ns']
        Nf16  = qf_d['Test_16']['Nf']
        eps16 = qf_d['Test_16']['eps'][i_tck]

        # Print progress
        print('\tPerforming test 16: LT flat line...')
        _ = printProgress(0,[])

        # Perform test
        t_16 = test16(data,Ns16,Nf16,eps16)
        _    = printProgress(20,[0])

        print('\tTest 16 done.')
    else: print('\tTest 16 not carried out.')

    # Test 17
    if qf_d['Test_17']['Do_Test']:
        # Unpack test parameters
        _,freq17,CS17,imin17,imax17,lmin17,lmax17 = qf_d['Test_17'].values()

        # Print progress
        print('\tPerforming test 17: LT operational frequency range...')
        progress = []
        for i in range(dim):
            # Progress
            iprog    = int(i/(dim-1)*20)
            progress = printProgress(iprog,progress)

            # Perform test
            t_17[i] = test17(freq17,CS17[:,i,:],imin17,imax17,lmin17,lmax17)

        print('\tTest 17 done.')
    else: print('\tTest 17 not carried out.')

    # Test 18
    if qf_d['Test_18']['Do_Test']:
        # Unpack test parameters
        _ = qf_d['Test_18'].values()

        # Print progress
        print('\tPerforming test 18: LT low-frequency energy...')
        _ = printProgress(0,[])

        # Perform test
        t_18 = test18(data)
        _    = printProgress(20,[0])

        print('\tTest 18 done.')
    else: print('\tTest 18 not carried out.')

    # Test 19
    if qf_d['Test_19']['Do_Test'] and f_tck:
        # Unpack test parameters
        rmin19   = qf_d['Test_19']['rmin'][i_tck]
        rmax19   = qf_d['Test_19']['rmax'][i_tck]
        hm0_flag = qf_d['Test_19']['hm0_flag']

        # Print progress
        print('\tPerforming test 19: LT acceptable range...')
        _ = printProgress(0,[])

        # Perform test
        t_19,hm0_flag = test19(data,rmin19,rmax19,varname,hm0_flag)
        _             = printProgress(20,[0])

        # Update "hm0_flag"
        qf_d['Test_19']['hm0_flag'] = hm0_flag

        print('\tTest 19 done.')
    else: print('\tTest 19 not carried out.');

    # Test 20
    if qf_d['Test_20']['Do_Test'] and f_tck:
        # Unpack test parameters
        rmin19  = qf_d['Test_19']['rmin'][i_tck]
        rmax19  = qf_d['Test_19']['rmax'][i_tck]
        delta20 = qf_d['Test_20']['delta'][i_tck]

        # Print progress
        print('\tPerforming test 20: LT acceptable variation...')
        _ = printProgress(0,[])

        # Perform test without failed spectral variables
        t_20 = test20(data,i_failed,delta20,rmin19,rmax19)
        _    = printProgress(20,[0])

        print('\tTest 20 done.')
    else: print('\tTest 20 not carried out.')

    # Compute secondary quality flag for all tests
    t_qf = getLTQFSecondary(t_14,t_15,t_16,t_17,t_18,t_19,t_20)

    # Append to indices to test
    qf_data            = np.copy(qf_spv)
    qf_data[i_to_test] = np.array(t_qf)[i_to_test]

    # Return corrected variables and associated quality flags
    return data,qf_data,qf_d

# ---------- #
def getLTQFSecondary(qf14,qf15,qf16,qf17,qf18,qf19,qf20):
    '''
    Establish a secondary quality flag for long-term time series based on
    scores from tests 14 to 20.

    If any "XX" or "YY" flags, a scondary flag is established as follow:
        (X.14)  if quality flag "X" from test 14
        (X.15)  if quality flag "X" from test 15
        (X.16)  if quality flag "X" from test 16
        (X.17)  if quality flag "X" from test 17
        (X.18)  if quality flag "X" from test 18
        (X.19)  if quality flag "X" from test 19
        (X.20)  if quality flag "X" from test 20

    If all 1 or 2, or any 9, the quality flag is passed without any secondary
    one. Else, a "NaN" value is established, but this should raise a warning.
    '''

    # Initialize output
    dim = len(qf14)
    qf  = []

    # Matricize index in priority: 17,16,19,20,15,14,18
    qf_order = [qf17,qf16,qf19,qf20,qf15,qf14,qf18]
    qf_num   = [17,16,19,20,15,14,18]
    m1       = transpose([qf==1 for qf in qf_order])
    m2       = transpose([qf==2 for qf in qf_order])
    m3       = transpose([qf==3 for qf in qf_order])
    m4       = transpose([qf==4 for qf in qf_order])
    m9       = transpose([qf==9 for qf in qf_order])

    # Append code to corresponding first flagged test
    # 4-3-2-1-9            code priority
    # 14-15-16-17-18-19-20 test priority
    isec = 14
    for i in range(dim):
        if   any(m9[i]):       qf.append('9.0')
        elif any(m4[i]):       qf.append('4.%d'%qf_num[where(m4[i])[0][0]])
        elif any(m3[i]):       qf.append('3.%d'%qf_num[where(m3[i])[0][0]])
        elif all(m2[i]):       qf.append('2.0')
        elif all(m1[i]+m2[i]): qf.append('1.0')
        else:                  qf.append(np.nan)

    # Return quality flag with secondary flag
    return qf

# ---------- #
def getQFCombined(qfa,qfb,qf_num=[9,11,10,12,13,17,16,19,20,15,14,18]):
    '''
    Establish the quality flag for a variable computed from "a" and "b".
    The maximum quality flag is kept. If both 'QF' have the same principal
    quality flag, the smaller secondary quality flag is chosen based on
    "qf_num" is chosen.
    '''

    pqa,sqa = float(qfa.split('.')[0]),float(qfa.split('.')[1])
    pqb,sqb = float(qfb.split('.')[0]),float(qfb.split('.')[1])
    if pqa>pqb:
        primary   = pqa
        secondary = sqa
    elif pqa<pqb:
        primary   = pqb
        secondary = sqb
    else:
        primary   = pqa
        i_a       = abs(np.array(qf_num)-sqa).argmin() if sqa else np.nan
        i_b       = abs(np.array(qf_num)-sqb).argmin() if sqb else np.nan
        secondary = np.array(qf_num)[np.nanmin([i_a,i_b])] if sqa or sqb\
                    else 0

    # Return combined quality flag
    return '%d.%d'%(primary,secondary)

# ---------- #
def getSTQF(data,varname,qf_d):
    '''
    Carry all ST tests for a given data.
    '''

    # Compute dimension
    dim = len(data)

    # Initialize 'qf' outputs
    qf9,qf10,qf11,qf12,qf13 = [2*np.ones(dim) for _ in range(5)]

    # Do tests
    print('\n\nQuality control for %s'%varname)

    # Test 9
    if qf_d['Test_9']['Do_Test']:
        # Unpack test parameters
        _,N9 = qf_d['Test_9'].values()

        # Print progress
        print('\tPerforming test 9: ST gap...')
        progress = []
        for i in range(dim):
            # Progress
            iprog    = int(i/(dim-1)*20)
            progress = printProgress(iprog,progress)

            # Perform test
            data[i],qf9[i] = test9(data[i],N9)

        print('\tTest 9 done.')
    else: print('\tTest 9 not carried out.')

    # Test 10
    if qf_d['Test_10']['Do_Test']:
        # Unpack test parameters
        _,N10,m10 = qf_d['Test_10'].values()

        # Print progress
        print('\tPerforming test 10: ST spike...')
        progress = []
        for i in range(dim):
            # Progress
            iprog    = int(i/(dim-1)*20)
            progress = printProgress(iprog,progress)

            # Perform test
            data[i],qf10[i] = test10(data[i],N10,m10)

        print('\tTest 10 done.')
    else: print('\tTest 10 not carried out.')

    # Test 11
    if qf_d['Test_11']['Do_Test']:
        # Unpack test parameters
        _,imin11,imax11,lmin11,lmax11,nbins11 = qf_d['Test_11'].values()

        # Print progress
        print('\tPerforming test 11: ST range...')
        progress = []
        for i in range(dim):
            # Progress
            iprog    = int(i/(dim-1)*20)
            progress = printProgress(iprog,progress)

            # Perform test
            qf11[i] = test11(data[i],imin11,imax11,lmin11,lmax11,nbins11)

        print('\tTest 11 done.')
    else: print('\tTest 11 not carried out.')

    # Test 12
    if qf_d['Test_12']['Do_Test']:
        # Unpack test parameters
        _,m12,delta12 = qf_d['Test_12'].values()

        # Print progress
        print('\tPerforming test 12: ST segment shift...')
        progress = []
        for i in range(dim):
            # Progress
            iprog    = int(i/(dim-1)*20)
            progress = printProgress(iprog,progress)

            # Perform test
            qf12[i] = test12(data[i],m12,delta12)

        print('\tTest 12 done.')
    else: print('\tTest 12 not carried out.')

    # Test 13
    if qf_d['Test_13']['Do_Test']:
        # Unpack test parameters
        _,N13 = qf_d['Test_13'].values()

        # Print progress
        print('\tPerforming test 13: ST acceleration...')
        progress = []
        for i in range(dim):
            # Progress
            iprog    = int(i/(dim-1)*20)
            progress = printProgress(iprog,progress)

            # Perform test
            qf13[i] = test13(data[i],N13)

        print('\tTest 13 done.')
    else: print('\tTest 13 not carried out.')

    # Compute secondary quality flag
    qf_data = getSTQFSecondary(qf9,qf10,qf11,qf12,qf13)

    # Return corrected variables and associated quality flags
    return data,qf_data

# ---------- #
def getSTQFSecondary(qf9,qf10,qf11,qf12,qf13):
    '''
    Establish a secondary quality flag for short-term time series based on
    scores from tests 9 to 13.

    If any 3 or 4 flags, a scondary flag is established as follow:
        (X.9)   if quality flag "X" from test 9
        (X.10)  if quality flag "X" from test 10
        (X.11)  if quality flag "X" from test 11
        (X.12)  if quality flag "X" from test 12
        (X.13)  if quality flag "X" from test 13

    If all 1 or 2, or any 9, the quality flag is passed without any secondary
    one. Else, a "NaN" value is established, but this should raise a warning.
    '''

    # Initialize output
    dim = len(qf9)
    qf  = []

    # Matricize index in priority: 9-11-10-12-13
    qf_order = [qf9,qf11,qf10,qf12,qf13]
    qf_num   = [9,11,12,10,13]
    m1       = transpose([qf==1 for qf in qf_order])
    m2       = transpose([qf==2 for qf in qf_order])
    m3       = transpose([qf==3 for qf in qf_order])
    m4       = transpose([qf==4 for qf in qf_order])
    m9       = transpose([qf==9 for qf in qf_order])

    # Append code to corresponding first flagged test
    # 4-3-2-1-9     code priority
    for i in range(dim):
        if   any(m9[i]):       qf.append('9.0')
        elif any(m4[i]):       qf.append('4.%d'%qf_num[where(m4[i])[0][0]])
        elif any(m3[i]):       qf.append('3.%d'%qf_num[where(m3[i])[0][0]])
        elif all(m2[i]):       qf.append('2.0')
        elif all(m1[i]+m2[i]): qf.append('1.0')
        else:                  qf.append(np.nan)

    # Return quality flags with secondary flags
    return qf

# Aditionnal functions
# ---------- #
def removeOutliers(date,x,n,min_val,method='mad'):
    '''
    Remove zero values and outliers using median absolute deviation.
    '''

    # Find good indices
    i_inval   = where(x>min_val)[0]
    i_not_nan = where(np.invert(isnan(x)))[0]
    i_good    = np.unique(hstack([i_inval,i_not_nan]))

    # Keep good indices
    date = date[i_good]
    x    = x[i_good]

    # Find outliers
    x_dev = x - np.median(x)
    thrs  = -n/(2**(1/2)*erfcinv(3/2))*mad(x)
    i_out = where(abs(x_dev)<thrs)[0]

    # Return date and array without outliers and zeros
    return date[i_out],x[i_out]

# ---------- #
def movMean(x,m,index_list,remove_i=False):
    '''
    Compute moving mean on the interval of "2m+1" points centered on "x[i]".
    If an index list is specified, only those indices will be modified. Else,
    one should pass the whole index range to perform the rolling mean through
    the entire series. If "index_list" is not the whole index range and
    "remove_i" is "True", the rolling mean will discard "x[i]" in the
    computation of each segment, reducing the inverval to "2m".
    '''

    dim   = len(x)
    i0,i1 = -m,m+1
    mx    = np.copy(x)

    for i in index_list:
        xx    = hstack([mx[max(0,i+i0):i],mx[i+1:min(i+i1,dim)]]) if remove_i\
                else mx[max(0,i+i0):min(i+i1,dim)]
        mx[i] = [np.nan if len(where(isnan(xx))[0])==len(xx)\
                 else np.nanmean(xx)][0]

    return mx

# ---------- #
def printProgress(i,progress):
    '''
    Print the progress bar associated to a for loop process.
    '''

    if i not in progress:
        progress.append(i)
        sys.stdout.write('\r')
        sys.stdout.write("\t[%-20s] %d%%" % ('='*i,5*i))
        sys.stdout.flush()

    return progress

# END
