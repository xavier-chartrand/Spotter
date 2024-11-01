#!/usr/bin/python -u
# Author: Xavier Chartrand
# Email : x.chartrand@protonmail.me
#         xavier.chartrand@ec.gc.ca

'''
Compute statistics for SPOT buoy, from AZMP OGSL data.

The station "iml-4" is chosen as it it the closest to Spotter deployments.
'''

# Custom utilities
from ogsl_statistics_utils import *

# ---------- #
## MAIN
# Stations: iml-[4]
station_list = ['iml-4']

# Dictionnaries
header_k = {'Date':'Date',
            'AirTemperature':'Air Temperature',
            'AirHumidity':'Humidity',
            'Density':'Density',
            'Pressure':'Pressure',
            'Salinity':'Salinity',
            'Temperature':'Water Temperature',
            'WindSpeed':'Wind Speed',
            'WindGusts':'Wind Gust',
            'WindDirection':'Wind Direction',
            'Hm0':'Wave Significant Height',
            'Tm01':'Wave Period',
            'ThetaMean':'Wave Mean Direction',
            'ThetaPeak':'Wave Peak Direction',
            'SigmaMean':'Wave Mean Spreading',
            'SigmaPeak':'Wave Peak Spreading'}

# Output header and labels
header = '%s %s %s %s %s %s'%('Parameter,'.ljust(25),
                              'min,'.ljust(10),
                              'max,'.ljust(10),
                              'mean,'.ljust(10),
                              'std,'.ljust(10),
                              'set size')

# Iterate over all stations
for station in station_list:
    # Load file list for actual station
    f_dir  = '%s/'%station
    f_list = os.popen('ls %s%s*'%(f_dir,station)).read()\
                     .rstrip('\n').split('\n')
    years  = [f.rstrip('.csv').split('_')[-1] for f in f_list]

    # Initialize outputs
    date,atemp,sig,pres,sal,temp,wspd,wgst,hm0,tm01 = [[] for _ in range(10)]

    # Append data
    for file in f_list:
        DS = pd.read_csv(file,delimiter=',',skipinitialspace=True)
        date.append(DS[header_k['Date']])
        atemp.append(DS[header_k['AirTemperature']])
        sig.append(DS[header_k['Density']])
        pres.append(DS[header_k['Pressure']])
        sal.append(DS[header_k['Salinity']])
        temp.append(DS[header_k['Temperature']])
        wspd.append(DS[header_k['WindSpeed']])
        wgst.append(DS[header_k['WindGusts']])
        hm0.append(DS[header_k['Hm0']])
        tm01.append(DS[header_k['Tm01']])

    # "hstack" data
    date  = hstack(date)
    atemp = hstack(atemp)
    sig   = hstack(sig)
    pres  = hstack(pres)
    sal   = hstack(sal)
    temp  = hstack(temp)
    wspd  = hstack(wspd)
    wgst  = hstack(wgst)
    hm0   = hstack(hm0)
    tm01  = hstack(tm01)

    # Minimal acceptable values (based on Spotter specifications)
    # "hm0" min value is given by the buoy error on displacement which is 2cm
    # "tm01" values is the minimal period detected, which is 1.25s
    min_vals = [-10,1000,95,0,0,0,0,0.02,1.25]

    # Define std treshold for every variables
    n_std    = 5
    std_vals = hstack([8*[n_std],1*[2*n_std]])

    # Remove outliers
    _,atemp = removeOutliers(date,atemp,std_vals[0],min_vals[0])
    _,sig   = removeOutliers(date,sig,std_vals[1],min_vals[1])
    _,pres  = removeOutliers(date,pres,std_vals[2],min_vals[2])
    _,sal   = removeOutliers(date,sal,std_vals[3],min_vals[3])
    _,temp  = removeOutliers(date,temp,std_vals[4],min_vals[4])
    _,wspd  = removeOutliers(date,wspd,std_vals[5],min_vals[5])
    _,wgst  = removeOutliers(date,wgst,std_vals[6],min_vals[6])
    _,hm0   = removeOutliers(date,hm0,std_vals[7],min_vals[7])
    _,tm01  = removeOutliers(date,tm01,std_vals[8],min_vals[8])

    # Calculate statistical limits
    atemp_l = [np.mean(atemp)-std_vals[0]*np.std(atemp),
               np.mean(atemp)+std_vals[0]*np.std(atemp),
               np.mean(atemp),
               np.std(atemp),
               len(atemp)]
    sig_l   = [np.mean(sig)-std_vals[1]*np.std(sig),
               np.mean(sig)+std_vals[1]*np.std(sig),
               np.mean(sig),
               np.std(sig),
               len(sig)]
    pres_l  = [np.mean(pres)-std_vals[2]*np.std(pres),
               np.mean(pres)+std_vals[2]*np.std(pres),
               np.mean(pres),
               np.std(pres),
               len(pres)]
    sal_l   = [np.mean(sal)-std_vals[3]*np.std(sal),
               np.mean(sal)+std_vals[3]*np.std(sal),
               np.mean(sal),
               np.std(sal),
               len(sal)]
    temp_l  = [min_vals[4],
               np.mean(temp)+std_vals[4]*np.std(temp),
               np.mean(temp),
               np.std(temp),
               len(temp)]
    wspd_l  = [min_vals[5],
               np.mean(wspd)+std_vals[5]*np.std(wspd),
               np.mean(wspd),
               np.std(wspd),
               len(wspd)]
    wgst_l  = [min_vals[6],
               np.mean(wgst)+std_vals[6]*np.std(wgst),
               np.mean(wgst),
               np.std(wgst),
               len(wgst)]
    hm0_l   = [min_vals[7],
               np.mean(hm0)+std_vals[7]*np.std(hm0),
               np.mean(hm0),
               np.std(hm0),
               len(hm0)]
    tm01_l  = [min_vals[8],
               np.mean(tm01)+std_vals[8]*np.std(tm01),
               np.mean(tm01),
               np.std(tm01),
               len(tm01)]

    # Set variables not calculated with statistical limits
    ahumi_l  = [np.nan,np.nan,np.nan,np.nan,0]
    wdir_l   = [0,360,np.nan,np.nan,0]
    thetam_l = [0,360,np.nan,np.nan,0]
    thetap_l = [0,360,np.nan,np.nan,0]
    sigmam_l = [0,80,np.nan,np.nan,0]
    sigmap_l = [0,80,np.nan,np.nan,0]

    # Output keys and values
    tuples = [atemp_l,
              ahumi_l,
              sig_l,
              pres_l,
              sal_l,
              temp_l,
              wspd_l,
              wgst_l,
              wdir_l,
              hm0_l,
              tm01_l,
              thetam_l,
              thetap_l,
              sigmam_l,
              sigmap_l]

    # Pipe outputs
    dir_out = 'statistics/'
    f_out   = '%s_statistics.csv'%station
    sh('''echo "# STATION %s">%s'''%(station,dir_out+f_out))
    sh('''echo "# YEARS   %s\n">>%s'''%(', '.join(years),dir_out+f_out))
    sh('''echo "%s">>%s'''%(header,dir_out+f_out))

    cnt    = -1
    values = [v for v in header_k.values()]
    for v in values[1:]:
        cnt   +=  1
        bwp    = v + ','
        val0   = ('%.4f'%tuples[cnt][0]).rstrip('0').rstrip('.') + ','
        val1   = ('%.4f'%tuples[cnt][1]).rstrip('0').rstrip('.') + ','
        val2   = ('%.4f'%tuples[cnt][2]).rstrip('0').rstrip('.') + ','
        val3   = ('%.4f'%tuples[cnt][3]).rstrip('0').rstrip('.') + ','
        val4   = '%d'%tuples[cnt][4]
        string = '%s %s %s %s %s %s'%(bwp.ljust(25),
                                      val0.ljust(10),
                                      val1.ljust(10),
                                      val2.ljust(10),
                                      val3.ljust(10),
                                      val4)
        sh('''echo "%s">>%s'''%(string,dir_out+f_out))

# END
