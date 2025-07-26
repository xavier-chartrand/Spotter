#!/usr/bin/python -u
# Author: Xavier Chartrand
# Email : x.chartrand@protonmail.me
#         xavier.chartrand@uqar.ca
#         xavier.chartrand@ec.gc.ca

'''
OGSL statistics utilities.
'''

# Module
import numpy as np
import os
import pandas as pd
# Functions
from numpy import hstack,isnan,mean,median,nan,std,unique,where
from scipy.special import erfcinv
from scipy.stats import median_abs_deviation as mad
# Constants
from scipy.constants import pi
# Shell commands in python
# ----------
def sh(s): os.system("bash -c '%s'"%s)

# OGSL statistics utilities
# ----------
def removeOutliers(date,x,n,min_val,method='mad'):
    '''
    Remove zero values and outliers using median absolute deviation.
    '''

    # Find good indices
    i_inval   = where(x>min_val)[0]
    i_not_nan = where(~isnan(x))[0]
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

# END
