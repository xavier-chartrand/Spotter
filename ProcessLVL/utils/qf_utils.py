#!/usr/bin/python -u
# Author: Xavier Chartrand
# Email : x.chartrand@protonmail.me
#         xavier.chartrand@ec.gc.ca
#         xavier.chartrand@uqar.ca

'''
Quality flag utilities based on "Manual for Real-Time Quality Control of
In-Situ Surface Wave Data", verson 2.1, February 2019 (Bushnell 2019).

Overview of quality flags:
1) Pass             Data have passed critical real-time QC tests and are
                    deemed adequate for use as preliminary data ;

2) Not evaluated    Data have not been tested for quality, or the
                    information on quality is not available ;

3) Suspect or of    Data are considered to be either suspect or of high
   high interest    interest to operators and users.
                    They are flagged suspect to draw further attention to
                    them by operators ;

4) Fail             Data are considered to have failed one or more critical
                    real-time quality control checks. If they are
                    disseminated at all, it should be readily apparent
                    that they are not of acceptable quality ;

9) Missing data     Data are missing ("NaN" used as a placeholder).
'''

# Module
import numpy as np
import sys
import contextlib

# Functions
from csaps import CubicSmoothingSpline as CSS
from numpy import array,copy,diff,isnan,invert,hstack,mean,median,nan,nanmean,\
                  nanmin,nanstd,ones,shape,sqrt,std,tanh,transpose,unique,\
                  where,zeros
from scipy.special import erfcinv
from scipy.stats import median_abs_deviation as mad

## Short-term test functions
# ----------
def test9(st,*args):
    '''
    Short-term time series gaps.

    "args" must comprise:
    "N"         the number of consecutive points allowed to be missed.

    Status: Strongly Recommended
    '''

    # Quit if all "NaN", return "st" and 9 (missing data)
    if len(where(isnan(st))[0])==len(st): return st,9

    # Unpack "args" and get "st" dimensions
    N   = tuple(args)[0]
    dim = len(st)

    # Check for data gap with "NaN" index
    ibad = where(isnan(st))[0]
    if len(ibad)==0: return st,1                # (1) good, no gap
    else:                                       # check for gaps
        idiff,j = diff(ibad),0
        for i in range(len(idiff)):
            i,j = j if j else i,0
            while idiff[i+j]==1 and j<len(idiff):
                if j<N-1: j+= 1
                else: return st,4               # (4) fail, at least one gap
                                                # of "N" points

    # Spline gaps if any
    x      = array(range(len(st)))
    igood  = where(invert(isnan(st)))[0]
    css_st = CSS(x[igood],st[igood],smooth=1)

    # Return splined short-term time series and QF=1
    return css_st.spline(x=x),1

# ----------
def test10(st,*args):
    '''
    Short-term time series spike.

    "args" must comprise:
    "N"         the multiple of standard deviation that is not allowed to be
                exceeded ;
    "m"         the number of points of the segment centered on "x[i]" that is
                used to replace spikes with the segment average around "x[i]"\
                ("x[i]" is excluded from the mean calculation) ;
    "p"         the number of times the test is repeated ;
    "thrs"      the fraction of the number of spikes or outliers over the total
                series length that is not allowed to be exceeded.

    Status: Strongly Recommended
    '''

    # Unpack "args" and get "st" dimensions
    N,m,p,thrs = tuple(args)
    dim        = len(st)

    # Quit if all "NaN", return "st" and 9 (missing data)
    if len(where(isnan(st))[0])==dim: return st,9

    # Get outliers more than "N" standard deviation
    cnt   = 0
    mst_p = copy(st)
    ibad  = where(abs(st-nanmean(st))>=N*nanstd(st))[0]

    # Check outliers criterion
    for _ in range(p):
        if    len(ibad)>=thrs*dim: return st,4  # (4), too many outliers
        elif  len(ibad)==0:        return st,1  # (1), no outliers
        else: mst = movMean(mst_p,m,index_list=ibad,remove_i=True)

        # Check if values are "NaN"
        if len(where(isnan(mst))[0])==dim: return st,9

        # Update spike indices
        ibad  = where(abs(mst-nanmean(mst))>N*nanstd(mst))[0]
        mst_p = copy(mst)

    # Return failed test (4) if too many outliers remains after "p"
    # iterations, else good (1)
    if len(ibad): return st,4
    else:         return mst,1

# ----------
def test11(st,*args):
    '''
    Short-term time series range.

    "args" must comprise:
    "imin"      the instrument minimum ;
    "imax"      the instrument maximum ;
    "lmin"      the regional/seasonal/climate/sensor-dependant minimum ;
    "lmax"      the regional/seasonal/climate/sensor-dependant maximum.

    Status: Strongly Recommended
    '''

    # Unpack "args"
    imin,imax,lmin,lmax = tuple(args)

    # Quit if all "NaN", return 9 (missing data)
    if len(where(isnan(st))[0])==len(st): return 9

    # Check if values are in a valid range
    bexp_i = min(st)<imin or max(st)>imax
    bexp_l = min(st)<lmin or max(st)>lmax
    if   bexp_i: return 4                       # (4), out of "i" range
    elif bexp_l: return 3                       # (3), out of "l" range
    else:        return 1                       # (1), in range

# ----------
def test12(st,*args):
    '''
    Short-term time series segment shift.

    "args" must comprise:
    "m"         the length of each segment ;
    "delta"     the mean shift allowed.

    Status: Suggested
    '''

    # Unpack "args"
    m,delta = tuple(args)

    # Quit if all "NaN", return 9 (missing data)
    if len(where(isnan(st))[0])==len(st): return 9

    # Compute moving mean over the window "m"
    mst = movMean(st,m,index_list=range(len(st)))

    # Check if flattened values by the rolling mean vary more than "delta"
    # around its average
    bexp = std(mst)>=delta
    if bexp: return 4                           # (4), mean shift
    else:    return 1                           # (1), no mean shift

# ----------
def test13(st,*args,g=9.81):
    '''
    Short-term time series acceleration.

    "args" must comprise:
    "N"         the fraction of the gravitational acceleration that should not
                be exceeded.

    Status: Strongly Recommended
    '''

    # Unpack "args"
    N = tuple(args)

    # Quit if all "NaN", return 9 (missing data)
    if len(where(isnan(st))[0])==len(st): return 9

    # Count bad acceleration values
    ibad = where(st>N*g)

    # Check for bad values
    if len(ibad): return 3                      # (3), too large values
    else:         return 1                      # (1), not too large values

# ----------
def testNH(st,*args):
    '''
    Addition for: Short-term time series.

    "args" must comprise:
    "hcheck"    a flag indicating if heading have been available for correcting
                horizontal acceleration.
    '''

    # Unpack "args"
    hcheck = tuple(args)

    # Return quality flag
    return [4 if h else 1 for h in hcheck]

## Long-Terms
# ----------
# For each LT tests, the variable "lt" is defined as "lt=[bwp,sxx,syy,szz]"
# where "bwp" is a bulk wave parameter to be tested, and "sxx,syy,szz" are
# cross-spectral densities. For some tests, either all or some of these
# variables are necessary, but this choice of nomenclature uniformize
# the Python's processing routine.
#
# ----------
def test14(lt,*args):
    '''
    Long-term time series check ratio or check factor.

    The check ratio or check factor "R(f)" is a function of frequency and
    depth and should theoretically be 1 for relatively deep water waves.
    This function is validated for small frequency bands centered on "fv" and
    of width of "bw", that should correspond to physically intepretable
    frequency ranges (e.g. peak period, mean period, etc).

    "args" must comprise:
    "wnum"      wavenumber bins resolved ;
    "freq"      frequency bins resolved ;
    "H"         the water depth ;
    "fv"        a frequency to validate ;
    "bw"        the half-width of the band centered on "fval".

    Status: Strongly Recommended
    '''

    # Unpack "lt"
    _,sxx,syy,szz = [l for l in lt]

    # Unpack "args"
    wnum,freq,H,fv,bw = tuple(args)

    # Initialize outputs
    dim  = len(fv)
    qf14 = 3*ones(dim)

    # Get frequency index to
    iwn = [where((freq>=fv[i]*(1-bw))*(freq<=fv[i]*(1+bw)))[0]\
           for i in range(dim)]

    # Compute check factor in frequency bands to validate
    imiss = []
    R     = zeros(len(fv))
    j     = 0
    for i in iwn:
        if len(i):
            j+= 1
            R+= mean(sqrt(szz[:,i]/(sxx[:,i]+syy[:,i]))/tanh(H*wnum[i]),axis=1)
        else:
            imiss.append(i)

    # get "R" mean
    R = copy(R/j)

    # Update test results for good (1) ratios
    qf14[where((R>=0.9)*(R<=1.1))[0]] = 1

    # Flag missing "NaN" values (9)
    qf14[imiss] = nan

    # Return quality flag
    return qf14

# ---------- #
def test15(lt,*args):
    '''
    Long-term time series mean and standard deviation.

    Check for values of "bwp" outside a statistically valid range of "N" times
    the standard deviation around the mean.

    "args" must comprise:
    "N"         the number of standard deviation considered for a good
                statistical range around the mean.

    Status: Strongly Recommended
    '''

    # Unpack "lt"
    bwp,_,_,_ = [l for l in lt]

    # Unpack "args"
    N = tuple(args)

    # Initialize outputs
    qf15 = 3*ones(len(bwp))

    # Compute statistical limits
    lower_lim = nanmean(bwp) - N*array(nanstd(bwp))
    upper_lim = nanmean(bwp) + N*array(nanstd(bwp))
    igood     = where((bwp>=lower_lim)*(bwp<=upper_lim))[0]
    inan      = where(isnan(bwp))[0]

    # Append outputs
    qf15[igood] = 1
    qf15[inan]  = 9

    # Return quality flag
    return qf15

# ---------- #
def test16(lt,*args):
    '''
    Long-term time series flat line.

    Check for invariate observations from sensor or data collection failure.

    "args" must comprise:
    "Ns"        the suspicious consecutive number of same data ;
    "Nf"        the faulty consecutive number of same data ;
    "eps"       the difference not allowed to be exceeded.

    Status: Required
    '''

    # Unpack "lt"
    bwp,_,_,_ = [l for l in lt]

    # Unpack "args"
    Ns,Nf,eps = tuple(args)

    # Initialize output
    dim  = len(bwp)
    qf16 = ones(dim)

    # Iterate over values to measure the length of flat lines
    i,j = 0,1
    while j<dim:
        cnt = 1
        while abs(bwp[j]-bwp[i])<=eps\
        and abs(array(bwp[j]))>=eps and abs(array(bwp[i]))>=eps:
            j  += 1
            cnt+= 1
            if j==dim: break
        if   cnt>=Nf: qf16[i:j] = 4           # (4), long flat lines
        elif cnt>=Ns: qf16[i:j] = 3           # (3), suspicious flat lines
        else:         i,j       = j,j+1       # (1), no flat line

    # Flag missing "NaN" values (9)
    qf16[where(isnan(bwp))[0]] = 9

    # Return quality flag
    return qf16

# ---------- #
def test17(lt,*args):
    '''
    Long-term time series operational frequency range.

    Check if spectral values are measured inside a valid frequency range.
    Note that instead of flagging failed data outside the range, they should
    have already been filtered, or padded with zeros or "NaN" by the operator.

    "args" must comprise:
    "freq"      frequencies ;
    "csd_dep"   bulk wave parameters CSD dependancies ;
    "imin"      the instrument minimum range operator-defined ;
    "imax"      the instrument maximum range operator-defined ;
    "lmin"      the regional/seasonal/climate/sensor-dependant minimum ;
    "lmax"      the regional/seasonal/climate/sensor-dependant maximum ;
    "eps"       a noisefloor not to be exceeded outside the valid range.

    Status: Required
    '''

    # Unpack "lt"
    _,sxx,syy,szz = [l for l in lt]

    # Unpack "args"
    freq,csd_dep,imin,imax,lmin,lmax,eps = tuple(args)

    # Initialize outputs
    dim  = shape(sxx)[0]
    qf17 = ones(dim)

    # Find indices where data lie outside the faulty or suspicious range
    ifail    = where((freq<imin)*(freq>imax))[0]
    isus     = where((freq<lmin)*(freq>lmax))[0]
    csd_vars = []
    for cs in csd_dep: exec(f"csd_vars.append({cs})",globals(),locals())

    # Search for any spectra exceeding noisefloor at selected frequencies
    ifail,isus,inan = [[] for _ in range(3)]
    for i in range(dim):
        stdout_fail,stdout_sus,stdout_nan = [[] for _ in range(3)]
        for cs in csd_vars:
            exec(f"stdout_fail.append(any(abs(cs[i,ifail])))",
                 globals(),locals())
            exec(f"stdout_sus.append(any(abs(cs[i,isus])))",
                 globals(),locals())
            exec(f"stdout_nan.append(all(isnan(cs[i,:])))",
                 globals(),locals())
        if any(stdout_fail):ifail.append(i)
        if any(stdout_sus):isus.append(i)
        if any(stdout_nan):inan.append(i)

    # Append results
    qf17[isus]  = 3
    qf17[ifail] = 4
    qf17[inan]  = 9

    # Return quality flag
    return qf17

# ---------- #
def test18(lt,*args):
    '''
    Long-term time series low-frequency energy.

    Compare fetch and swell wave direction at low frequencies with minimum
    and maximum energy.

    Status: Required
    '''

    # /*
    # Test not performed here, since we are filtering high-energy infragravity
    # waves for AZMP buoys. Otherwise, one should perform the testing procedure
    # accordingly with specified deep-water conditions.
    # */

    # Return quality flag
    return 2*ones(len(lt[0]))

# ---------- #
def test19(lt,*args):
    '''
    Long-term time series acceptable range of bulk wave parameters.
    "args" must comprise:
    "rmin"      the lower bound of acceptable range ;
    "rmax"      the upper bound of acceptable range ;
    "set_flag"  should be 2 if variable is significant wave height, else 1
                if variable is wave period or wave directional parameters,
                else 0 ;
    "prev_qf"   quality codes previously calculated for other wave parameters.

    "rmin" and "rmax" are either physically constrained (i.e. from 0 to 360
    for direction and, arbitrarly, from 0 to 80 for spreading) or corresponds
    to the operator-defined lower and upper values.

    Significant wave height should be tested first. Then, results should be
    taken into consideration for remaining tests. If the significant wave
    height fails this test for some given times "t[i]", then all other wave
    parameter quality codes should also being set to (4), for same "t[i]".
    Otherwise, (3) should be indicated for remaining wave parameters. Hence,
    "prev_qf" is a variable dedicated for updating and passed on every
    "test_19" procedures. "set_flag" is a variable for indicating the type of
    wave parameter:
     4 for significant wave height ;
     3 for wave period and other directional parameters.

    Status: Required
    '''

    # Unpack "lt"
    bwp,_,_,_ = [l for l in lt]

    # Unpack "args"
    rmin,rmax,set_flag,prev_qf = tuple(args)

    # Initialize outputs
    qf19 = prev_qf if len(prev_qf) else ones(len(bwp))

    # Find indices of values outside range
    ibad = hstack([where(bwp<rmin)[0],where(bwp>rmax)[0]])

    # Update quality flag
    for i in range(len(bwp)):
        if i in ibad: qf19[i] = max(qf19[i],set_flag)

    # Return quality flag
    return qf19

# ---------- #
def test20(lt,*args):
    '''
    Long-term time series acceptable variation of bulk wave parameters.

    "args" must comprise:
    "rmin"      the lower bound of acceptable range ;
    "rmax"      the upper bound of acceptable range ;
    "eps"       the acceptable variation.

    Status: Required
    '''

    # Unpack "lt"
    bwp,_,_,_ = [l for l in lt]

    # Unpack "args"
    rmin,rmax,eps = tuple(args)

    # Initialize output
    qf20 = ones(len(bwp))

    # Compute variation
    ivar = where(abs(diff(bwp))>eps)[0] + 1
    imin = where(bwp<=rmin)[0]
    imax = where(bwp>=rmax)[0]
    ibad = []
    [ibad.append(i) if i not in hstack([imin,imax]) else None for i in ivar]

    # Append outputs
    if len(ibad): qf20[ibad] = 4

    # Flag missing "NaN" values (9)
    qf20[where(isnan(bwp))] = 9

    # Return quality flag
    return qf20

# ---------- #
def getLTQF(data,varname,qf_d,update_hm0=False):
    '''
    Carry all LT tests for a given data.
    '''

    # Do tests
    print('\n\nQuality control for %s'%varname)
    dim = len(data[0])

    # Loop for each test
    for o in [k.split('_')[-1] for k in qf_d.keys()\
              if k.split('_')[-1]!='Order']:
        qft = []
        if qf_d['Test_%s'%o]['Do_Test'] and not len(qf_d['Test_%s'%o]['QF']):
            # Unpack test parameters
            args = [v for k,v in qf_d['Test_%s'%o].items()\
                    if k not in ['Do_Test','QF']]

            # Perform test if not already performed
            exec("global stdout; stdout=test%s(data,*args)"\
                 %o,globals(),locals())
            qft = copy(stdout)
            print('\tTest %s done.'%o)
        elif len(qf_d['Test_%s'%o]['QF']):
            print('\tTest %s already carried out.'%o)
            qft = qf_d['Test_%s'%o]['QF']
        else:
            print('\tTest %s not carried out.'%o)
            qft = 2*ones(dim)

        # Append test results
        qf_d['Test_%s'%o]['QF'] = qft

    # Compute secondary quality flag
    qf_data = getQFSecondary(qf_d)

    # Return corrected variables and associated quality flags
    return data[0],qf_data

# ---------- #
def getQFCombined(qfa,qfb,qf_ord):
    '''
    Establish the quality flag for a variable computed from "a" and "b".
    The maximum quality flag is kept. If both 'QF' have the same principal
    quality flag, the smaller secondary quality flag is chosen based on
    "qf_ord" is chosen.
    '''

    pqa,sqa = [q for q in qfa.split('.')]
    pqb,sqb = [q for q in qfb.split('.')]

    # Check if "sqa","sqb" in "qf_ord"
    bexp_a = sqa in qf_ord
    bexp_b = sqb in qf_ord

    if not bexp_a and not bexp_b:
        prm = '9' if (pqa=='9' or pqb=='9') else\
              '2' if (pqa=='2' and pqb=='2') else\
              '1'
        sec = '0'
    elif bexp_a and not bexp_b:
        prm,sec = pqa,sqa
    elif not bexp_a and bexp_b:
        prm,sec = pqb,sqb
    else:
        if pqa>pqb:
            prm = pqa
            sec = sqa
        elif pqa<pqb and sqb in qf_ord:
            prm = pqb
            sec = sqb
        else:
            prm = pqa
            i_a = where([q==sqa for q in qf_ord])[0][0]\
                  if sqa in qf_ord and sqa!='0' else nan
            i_b = where([q==sqb for q in qf_ord])[0][0]\
                  if sqb in qf_ord and sqb!='0' else nan
            sec = array(qf_ord)[int(nanmin([i_a,i_b]))]\
                  if ~isnan(i_a) or ~isnan(i_a) else 0

    # Return combined quality flag
    return '%s.%s'%(prm,sec)

# ---------- #
def getSTQF(data,varname,qf_d):
    '''
    Carry all ST tests for a given data.
    '''

    # Do tests
    print('\n\nQuality control for %s'%varname)
    dim = shape(data)[0]

    # Loop for each test
    for o in [k.split('_')[-1] for k in qf_d.keys()\
              if k.split('_')[-1]!='Order']:
        qft = []
        if qf_d['Test_%s'%o]['Do_Test'] and not len(qf_d['Test_%s'%o]['QF']):
            # Unpack test parameters
            args = [v for v in qf_d['Test_%s'%o].values()][1:-3]

            # Print progress
            print('\tPerforming test %s...'%o)
            progress = []
            for i in range(dim):
                # Progress
                iprog    = int(i/(dim-1)*20)
                progress = printProgress(iprog,progress)

                # Perform test if not already performed
                exec("global stdout; stdout=test%s(data[i],*args)"\
                     %o,globals(),locals())
                if qf_d['Test_%s'%o]['Update_Data']:
                    data[i] = stdout[0]
                    qft.append(stdout[1])
                else:
                    qft.append(stdout)
            print('\tTest %s done.'%o)
        elif len(qf_d['Test_%s'%o]['QF']):
            print('\tTest %s already carried out.'%o)
            qft = qf_d['Test_%s'%o]['QF']
        else:
            print('\tTest %s not carried out.'%o)
            qft = 2*ones(dim)

        # Append test results
        qf_d['Test_%s'%o]['QF'] = qft

    # Compute secondary quality flag
    qf_data = getQFSecondary(qf_d)

    # Return corrected variables and associated quality flags
    return data,qf_data

# ---------- #
def getQFSecondary(qf_dct):

    '''
    Establish a secondary quality flag for short- or long-term time series.

    "qf" is a dictionnary containing primary quality code for each test.
    "qf.keys()" must corresponds to the test number, which is used here
    as secondary quality code.

    As multiple tests may flag data, priority number is indicated in
    "qf_dct.Test_Order" entries, from the first being in less priority, and the
    last being the most prioritary one.

    If any 3 or 4 flags, the quality code is established as follow:
        (X.Y) if quality flag "X" from test "Y"

    If all 1 or 2, or any 9, the quality flag is passed without any secondary
    one. Else, a "NaN" value is established, but this should raise a warning.
    '''

    # Initialize output
    qf_val,qf_sec,qf_out = [[] for _ in range(3)]

    # Retrieve test results by specified order
    qf_ord = qf_dct['Test_Order']
    [qf_val.append(qf_dct['Test_%s'%o]['QF'])\
     if o in [k.split('_')[-1] for k in qf_dct.keys()]\
     and len(qf_dct['Test_%s'%o]['QF']) else None\
     for o in qf_ord]
    [qf_sec.append(o) if o in [k.split('_')[-1] for k in qf_dct.keys()]\
     else None\
     for o in qf_ord]

    # Matricize primary indice
    m1 = np.transpose([[q==1 for q in qf] for qf in qf_val])
    m2 = np.transpose([[q==2 for q in qf] for qf in qf_val])
    m3 = np.transpose([[q==3 for q in qf] for qf in qf_val])
    m4 = np.transpose([[q==4 for q in qf] for qf in qf_val])
    m9 = np.transpose([[q==9 for q in qf] for qf in qf_val])

    # Append code to corresponding first flagged test
    # 9-4-3-2-1 code priority
    for i in range(len(m1)):
        if any(m9[i]):         qf = '9.0'
        elif any(m4[i]):       qf = '4.%s'%qf_sec[where(m4[i])[0][-1]]
        elif any(m3[i]):       qf = '3.%s'%qf_sec[where(m3[i])[0][-1]]
        elif all(m2[i]):       qf = '2.0'
        elif all(m1[i]+m2[i]): qf = '1.0'
        else:                  qf = 'nan'
        qf_out.append(qf)

    # Return quality flags with secondary flags
    return qf_out

# ----------#
def normalizeAclSTQF(qf_data,qf_dct):
    '''
    Normalize quality codes between 3D surface motion ST time series, following
    the same priority order as defined in the "getSTQFSecondary" function.

    If acceleration data of the same type ("h" for horizontal or "v" for
    vertical) shares dissimilar quality codes for a given time, then their
    quality code are normalized.

    "data" must be formatted as follows: "[x_acl,y_acl,z_acl]"
    "qf_data" must be formatted as follows: "[qf_x_acl,qf_y_acl,qf_z_acl]"
    '''

    # Initialize outputs
    n_qf_h,n_qf_v = [[] for _ in range(2)]

    # "st" dimensions
    for qf in qf_data:
        if len(qf):
            dim=len(qf)
            break

    # Normalize quality codes
    for i in range(dim):
        qf_h,qf_v = [nan for _ in range(2)]
        # Retrieve primary and secondary quality codes
        qf_prm = array([qf[i].split('.')[0] for qf in qf_data])
        qf_sec = array([qf[i].split('.')[1] for qf in qf_data])

        # Normalize quality flag per type and test order
        if '4' in qf_prm:
            iqf = where(qf_prm=='4')[0]
            for o in qf_dct['Test_Order']:
                if o in qf_sec[iqf]:
                    qf_v = '4.%s'%o if 'v' in qf_dct['Test_%s'%o]['Type'] else\
                           '%s.%s'%(qf_prm[2],qf_sec[2])
                    qf_h = '4.%s'%o if 'h' in qf_dct['Test_%s'%o]['Type'] else\
                           getQFCombined('%s.%s'%(qf_prm[0],qf_sec[0]),
                                         '%s.%s'%(qf_prm[1],qf_sec[1]))
                    break
        elif '3' in qf_prm:
            iqf = where(qf_prm=='3')[0]
            for o in qf_dct['Test_Order']:
                if o in qf_sec[iqf]:
                    qf_v = '3.%s'%o if 'v' in qf_dct['Test_%s'%o]['Type'] else\
                           '%s.%s'%(qf_prm[2],qf_sec[2])
                    qf_h = '3.%s'%o if 'h' in qf_ctd['Test_%s'%o]['Type'] else\
                            getQFCombined('%s.%s'%(qf_prm[0],qf_sec[0]),
                                          '%s.%s'%(qf_prm[1],qf_sec[1]))
                    break
        else:
            qf_h = '1.0' if all([qf=='1' for qf in qf_prm[:2]]) else\
                   '9.0' if any([qf=='9' for qf in qf_prm[:2]]) else\
                   '2.0'
            qf_v = '1.0' if qf_prm[2]=='1' else\
                   '9.0' if qf_prm[2]=='9' else\
                   '2.0'

        # Append normalized quality code
        n_qf_h.append(qf_h)
        n_qf_v.append(qf_v)

    return array(n_qf_h),array(n_qf_v)

# Aditionnal functions
# ---------- #
def removeOutliers(date,x,n,min_val,method='mad'):
    '''
    Remove zero values and outliers using median absolute deviation criterion.
    '''

    # Find good indices
    i_inval   = where(x>min_val)[0]
    i_not_nan = where(invert(isnan(x)))[0]
    i_good    = unique(hstack([i_inval,i_not_nan]))

    # Keep good indices
    date = date[i_good]
    x    = x[i_good]

    # Find outliers
    x_dev = x - median(x)
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
    mx    = copy(x)

    for i in index_list:
        xx = hstack([mx[max(0,i+i0):i],mx[i+1:min(i+i1,dim)]]) if remove_i\
             else mx[max(0,i+i0):min(i+i1,dim)]
        if not len(where(isnan(xx))[0])==len(xx):
            mx[i] = nanmean(xx)

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
