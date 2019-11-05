#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division

import sys

import pandas as pd
import pickle
import numpy as np
import glob
from scipy import stats
from statistics import mean
import scipy.interpolate as interp

from metpy.units import pandas_dataframe_to_unit_arrays
import metpy.calc as mpcalc
import metpy.constants as mpconst


import matplotlib.pyplot as plt
import matplotlib as mpl


def str2int(x):
    if x[0] == '0':
        return int(x[-1])
    else:
        return int(x)

def int2str(x):
    if x < 10:
        return '0'+str(x)
    else:
        return str(x)

def datetime2doy(ydt):
    #print(ydt[:4])
    year = int(ydt[:4])
    month = str2int(ydt[4:6])
    date = str2int(ydt[6:8])
    hour = str2int(ydt[8:10])
    minute = str2int(ydt[10:12])

    leap = np.array([31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366])
    nonleap = np.array([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])

    if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
        summonth = leap
    else:
        summonth = nonleap

    doy = (summonth[month-2] + date) * 1.0
    minofday = np.round(((hour * 60 + minute) / 1440) ,2)
    doy += minofday

    return doy

def doy2datetime(doy, year):

    doydate, doytime = divmod(doy, 1)
    doydate = int(doydate)

    leap = np.array([31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366])
    nonleap = np.array([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])

    if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
        summonth = leap
    else:
        summonth = nonleap
    month = np.where(summonth-doydate > 0)[0][0] + 1
    dom = doydate - summonth[month-2]

    doytime = doytime * 86400
    hr, doymin = divmod(doytime, 3600)
    hr = int(hr)
    mins, sec = divmod(doymin, 60)
    mins = int(mins)
    sec = int(sec)

    return month, dom, hr, mins, sec

def rel_loc1(df,pp):
    """
    returns:
    relative distance (deg)
    relative latitude (deg)
    relative longitude (deg)
    """
    filename = pp.split('/')[-1].strip('.pkl')
    #txt_path = '/Users/norac/STI/notebook/composite/dennis/txt/' + filename + '.txt'
    txt_path = '/Volumes/Seagate_Exp/STI/10k/txt/' + filename + '.txt'

    with open(txt_path,'r') as f:
        line = f.readline()

    cen_lon = float(line.split(' ')[-2])
    cen_lat = float(line.split(' ')[-1])

    sonde_lon = float(filename.split('_')[-2])
    sonde_lat = float(filename.split('_')[-1])

    rel_lon = sonde_lon - cen_lon
    rel_lat = sonde_lat - cen_lat

    rel_dist = np.sqrt(rel_lat**2 + rel_lon**2)
    return rel_dist, rel_lat, rel_lon


def tr_winds(df, rel_lat, rel_lon):
    """
    returns tangential wind speed (m/s) and radial wind speed (m/s)
    """
    wnd_spd = df['WSPD'].values
    wnd_spd = wnd_spd.astype(np.float)

    wnd_dir_deg = df['WDIR'].values
    wnd_dir_deg = wnd_dir_deg.astype(np.float)
    wnd_dir = np.deg2rad(wnd_dir_deg)    # in rad

    if rel_lon >= 0 and rel_lat > 0:     # [0,pi/2)
        pos_dir = np.arctan(rel_lon/rel_lat)

    elif rel_lon > 0 and rel_lat <= 0:   # [pi/2,pi)
        pos_dir = np.arctan(-rel_lat/rel_lon) + np.pi/2

    elif rel_lon <= 0 and rel_lat < 0:   # [pi,3pi/2)
        pos_dir = np.arctan(rel_lon/rel_lat) + np.pi

    elif rel_lon < 0 and rel_lat >= 0:   # [3pi/2,2pi)
        pos_dir = np.arctan(-rel_lat/rel_lon) + np.pi*3/2

    elif rel_lat == 0 and rel_lon == 0:
        pos_dir = 0

    wnd_tan = wnd_spd * np.sin((wnd_dir - pos_dir))
    wnd_rad = - wnd_spd * np.cos((wnd_dir - pos_dir))
    return wnd_tan, wnd_rad

def spddir2uvwind(wnd_spd, wnd_dir):
    """
    Convert full wind speed into u and v conponents.
    Parameters
    ---------
    wnd_spd: array-like
        Full wind speed (m/s for dropsonde data).
    wnd_dir: array-like
        Wind direction (deg). Northernly: 0 deg Westernly: 270 deg.
    Returns
    ------
    u_wind: array-like
        U (lateral) conponent of the full wind.
    v_wind: array-like
        V (meridional) conponent of the full wind.
    """
    wnd_dir_rad = np.deg2rad(wnd_dir)
    u_wind = np.sin(wnd_dir_rad - np.pi) * wnd_spd
    v_wind = np.cos(wnd_dir_rad - np.pi) * wnd_spd

    return u_wind, v_wind

def Helicity_2d(u, v, z):
    """
    Calculate single level helicity from u, v winds.
    Parameters
    ---------
    u: array-like
        U (lateral) conponent of the full wind.
    v_wind: array-like
        V (meridional) conponent of the full wind.
    z: array-like
        Level height (m).
    Returns
    ------
    Helicity: array-like
        H = v*dudz - u*dvdz. (m-2/s-2)
    """
    dudz = np.round(np.gradient(u, z, edge_order=1),3)
    dvdz = np.round(np.gradient(v, z, edge_order=1),3)
    H = np.round((v*dudz - u * dvdz),2)

    return H

def Helicity_sfc3k(df):
    spd = df['WSPD'].values.astype(np.float)
    dirc = df['WDIR'].values.astype(np.float)
    height = df['ALTD'].values.astype(np.float)
    u_wind, v_wind = spddir2uvwind(spd, dirc)

    # INTERPOLATE
    if height[-1] > 3000:
        return np.nan
    else:
        st_idx = np.where(height<3000)[0][0]
        st = np.round(height[st_idx],0)     # Discard the height[0]
        ed = np.round(height[-1],0)
        if (st-ed)>10:
            if len(height)>100:
                resamp_nb = len(height)//100*100
            elif len(height)>10:
                resamp_nb = len(height)//10*10
            else:
                resamp_nb = len(height)
            height_itp = np.linspace(st-1, ed+1, resamp_nb)
            delta_h = height_itp[0]-height_itp[1]


            if delta_h >= 10:
                height_itp = np.arange(st-1, ed+1, -10)

            fu = interp.interp1d(height[st_idx:], u_wind[st_idx:])
            #print(height_itp[0])
            #print(height[st_idx])
            u_wind_itp = fu(height_itp)

            fv  = interp.interp1d(height[st_idx:], v_wind[st_idx:])
            v_wind_itp = fv(height_itp)
            Heli = Helicity_2d(u_wind_itp, v_wind_itp, height_itp)
            Heli_sfc3k = sum(Heli)
            return Heli_sfc3k
        else:
            return np.nan



def SRH_sfc3k(df, u_storm=0, v_storm=0, column_top=3000):
    spd = df['WSPD'].values.astype(np.float)
    dirc = df['WDIR'].values.astype(np.float)
    height = df['ALTD'].values.astype(np.float)
    u_wind, v_wind = spddir2uvwind(spd, dirc)

    # INTERPOLATE
    if height[-1] > column_top:
        return np.nan
    else:
        st_idx = np.where(height<column_top)[0][0]
        st = np.round(height[st_idx],0)     # Discard the height[0]
        ed = np.round(height[-1],0)
        if (st-ed)>10:
            if len(height)>100:
                resamp_nb = len(height)//100*100
            elif len(height)>10:
                resamp_nb = len(height)//10*10
            else:
                resamp_nb = len(height)
            height_itp = np.linspace(st-1, ed+1, resamp_nb)

            delta_h = height_itp[0]-height_itp[1]


            if delta_h >= 10:
                height_itp = np.arange(st-1, ed+1, -10)
            hght_diff = height_itp[:-1] - height_itp[1:]

            fu = interp.interp1d(height[st_idx:], u_wind[st_idx:])
            #print(height_itp[0])
            #print(height[st_idx])
            u_wind_itp = fu(height_itp)

            fv  = interp.interp1d(height[st_idx:], v_wind[st_idx:])
            v_wind_itp = fv(height_itp)

            u_storm_relative = u_wind_itp - u_storm
            v_storm_relative = v_wind_itp - v_storm

            Heli = Helicity_2d(u_storm_relative, v_storm_relative, height_itp)
            Heli_sfc3k = np.dot(Heli, hght_diff)
            #Heli_sfc3k = sum(Heli)
            return Heli_sfc3k
        else:
            return np.nan


def SRH_metpy(df, u_storm=0, v_storm=0, column_top=3000):
 #https://unidata.github.io/MetPy/latest/_modules/metpy/calc/kinematics.html#storm_relative_helicity
    spd = df['WSPD'].values.astype(np.float)
    dirc = df['WDIR'].values.astype(np.float)
    height = df['ALTD'].values.astype(np.float)
    u_wind, v_wind = spddir2uvwind(spd, dirc)

    # INTERPOLATE
    if height[-1] > column_top:
        return np.nan
    else:
        st_idx = np.where(height<column_top)[0][0]
        st = np.round(height[st_idx],0)     # Discard the height[0]
        ed = np.round(height[-1],0)
        if (st-ed)>10:
            if len(height)>100:
                resamp_nb = len(height)//100*100
            elif len(height)>10:
                resamp_nb = len(height)//10*10
            else:
                resamp_nb = len(height)
            height_itp = np.linspace(st-1, ed+1, resamp_nb)
            delta_h = height_itp[0]-height_itp[1]


            if delta_h >= 10:
                height_itp = np.arange(st-1, ed+1, -10)

            fu = interp.interp1d(height[st_idx:], u_wind[st_idx:])
            #print(height_itp[0])
            #print(height[st_idx])
            u_wind_itp = fu(height_itp)

            fv  = interp.interp1d(height[st_idx:], v_wind[st_idx:])
            v_wind_itp = fv(height_itp)

            u_storm_relative = u_wind_itp - u_storm
            v_storm_relative = v_wind_itp - v_storm

            int_layers = (u_storm_relative[1:] * v_storm_relative[:-1]
                          - u_storm_relative[:-1] * v_storm_relative[1:])

            positive_srh = int_layers[int_layers > 0.].sum()
            #if np.ma.is_masked(positive_srh):
            #    positive_srh = 0.0
            negative_srh = int_layers[int_layers < 0.].sum()
            #if np.ma.is_masked(negative_srh):
            #    negative_srh = 0.0
            srh = int_layers.sum()

            return srh, positive_srh, negative_srh
        else:
            return np.nan, np.nan, np.nan



def attach_units(df):
    """
    returns:
    a dataframe with unit attached to pres, temp, and hum
    for metpy calculation
    """
    pres_l = df['PRES'].values
    temp_1 = df['TEMP'].values
    hum_l = df['HUM'].values

    dic = {'PRES':pres_l,
           'TEMP':temp_1,
           'HUM': hum_l}
    df_new = pd.DataFrame(data=dic)
    #print(df_new['HUM'].values)
    my_units = {
            'PRES':'hPa',
            'TEMP':'degC',
            'HUM': 'percent'}

    my_united_data = pandas_dataframe_to_unit_arrays(df_new, column_units=my_units)

    # type(my_united_data) == dict
    return my_united_data


def calc_thta_vir(united_data):
    """
    returns virtual potential temperature (K)
    and equvalent potential temperaure (K)
    """

    pres = united_data['PRES']
    temp = united_data['TEMP']
    rh = united_data['HUM']
    mixing = mpcalc.mixing_ratio_from_relative_humidity(rh, temp, pres)
    theta_vir = mpcalc.virtual_potential_temperature(pres, temp, mixing)

    td = mpcalc.dewpoint_rh(temp, rh)
    theta_e = mpcalc.equivalent_potential_temperature(pres, temp, td)

    return theta_vir, theta_e


def grab_recon_datetime(recon_name):
    """
        Find out the launch time from the txt file.
    """

   # path = '/Users/norac/STI/HRD-DROPSONDE_PROCESSED/txt/'+recon_name+'.txt'
    path = '/Volumes/Seagate_Exp/STI/10k/txt/'+recon_name+'.txt'
    with open(path,'r') as f:
        lines = f.readlines()
    try:
        for line in lines[-30:]:
            if line[10:40] == 'COM Launch Time (y,m,d,h,m,s):':
                recon = line
                break

        date_slash = recon.split()[-2][:-1]
        time_colon = recon.split()[-1][:8]
        if date_slash[4] == '/':
            date = date_slash.split('/')[0] + date_slash.split('/')[1] + date_slash.split('/')[2]
        elif date_slash[4] == '-':
            date = date_slash.split('-')[0] + date_slash.split('-')[1] + date_slash.split('-')[2]
        time = time_colon.split(':')[0] + time_colon.split(':')[1]
        #print('datetime',date, time)
        datetime = int(date+time)
        return datetime
    except UnboundLocalError:
        with open(path,'r') as f:
            line1 = f.readline()
        yymmdd = line1.split(' ')[5]
        hhmmss = line1.split(' ')[6]
        date = yymmdd.split('-')[0] + yymmdd.split('-')[1] + yymmdd.split('-')[2]
        time = hhmmss.split(':')[0] + hhmmss.split(':')[1]
        #print('datetime',date, time)
        datetime = int(date+time)
        print('File is not complete')

        return datetime

def wind_color(spd):
    if spd >= 40:
        r = 255/255
        g = 94/255
        b = 99/255
    elif spd>=30 and spd<40:
        r = 255/255
        g = 141/255
        b = 60/255
    elif spd>=20 and spd<30:
        r = 255/255
        g = 191/255
        b = 89/255
    elif spd>=15 and spd<20:
        r = 0/255   #r = 255/255
        g = 251/255 #g = 230/255
        b = 244/255 #b = 134/255
    elif spd>=5 and spd<10:
        r = 84/255  #r = 255/255
        g = 184/255 #g = 254/255
        b = 244/255 #b = 209/255
    else:
        r = 255/255
        g = 254/255
        b = 209/255
    color = [r,g,b]
    return color


def cat_knots(wnd_cat):
    """
        Assign category according to when the dropsonde was launched.
        于后访提供的插值信息
    """

    #path = '/Volumes/Seagate_Exp/STI/10k/txt/'+reconname+'.txt'
    #path = '/Users/norac/STI/joaquin/txt/g114615114_201510021136_JOAQUIN_-74.83_21.29.txt'
    #with open(path,'r') as f:
    #    first_line = f.readline()
    #wnd_cat = float(first_line.split(' ')[-4])
    ########### wnd_cat = df['TYITY'].values[-1]  ##########
    if wnd_cat < 64:
        cat = -1
        r = 255/255
        g = 254/255
        b = 209/255
    elif wnd_cat>=64 and wnd_cat < 83:
        cat = 1
        r = 84/255  #r = 255/255
        g = 184/255 #g = 254/255
        b = 244/255 #b = 209/255
    elif wnd_cat>=83 and wnd_cat<96:
        cat = 2
        r = 0/255   #r = 255/255
        g = 251/255 #g = 230/255
        b = 244/255 #b = 134/255
    elif wnd_cat>=96 and wnd_cat<113:
        cat = 3
        r = 255/255
        g = 191/255
        b = 89/255
    elif wnd_cat>=113 and wnd_cat<137:
        cat = 4
        r = 255/255
        g = 141/255
        b = 60/255
    elif wnd_cat>=137:
        cat = 5
        r = 255/255
        g = 94/255
        b = 99/255
    color = [r,g,b]

    return cat,color

def cat_mps(wnd_cat):
    """
        Assign category according to when the dropsonde was launched.
        于后访提供的插值信息
    """

    #path = '/Volumes/Seagate_Exp/STI/10k/txt/'+reconname+'.txt'
    #path = '/Users/norac/STI/joaquin/txt/g114615114_201510021136_JOAQUIN_-74.83_21.29.txt'
    #with open(path,'r') as f:
    #    first_line = f.readline()
    #wnd_cat = float(first_line.split(' ')[-4])
    ########### wnd_cat = df['TYITY'].values[-1]  ##########
    if wnd_cat < 33:
        cat = -1
        r = 255/255
        g = 254/255
        b = 209/255
    elif wnd_cat>=33 and wnd_cat < 43:
        cat = 1
        r = 84/255  #r = 255/255
        g = 184/255 #g = 254/255
        b = 244/255 #b = 209/255
    elif wnd_cat>=43 and wnd_cat<50:
        cat = 2
        r = 0/255   #r = 255/255
        g = 251/255 #g = 230/255
        b = 244/255 #b = 134/255
    elif wnd_cat>=50 and wnd_cat<58:
        cat = 3
        r = 255/255
        g = 191/255
        b = 89/255
    elif wnd_cat>=58 and wnd_cat<70:
        cat = 4
        r = 255/255
        g = 141/255
        b = 60/255
    elif wnd_cat>=70:
        cat = 5
        r = 255/255
        g = 94/255
        b = 99/255
    color = [r,g,b]

    return cat,color


def identify_stage(path,df_BT):
    """
        Generate the linear interpolated latitude and longitude
        for the current file (path(pkl)) from 6hr best track file (df_BT).
    """

    #recon_info = path.strip('/Users/norac/STI/HRD-DROPSONDE_PROCESSED/pkl/')
    recon_info = path.split('/')[-1]
    recon_info = recon_info.strip('.pkl')

    datetime_recon = grab_recon_datetime(recon_info)
    # Wrong!
    datetime_BT_6hr = df_BT['DATETIME']
    for i in range(len(df_BT)-1):
       # print('recon-match',datetime_recon - datetime_BT_6hr[i])
       # print('match+1 - match',datetime_BT_6hr[i+1] - datetime_BT_6hr[i])
        if (datetime_recon - datetime_BT_6hr[i]) >= 0 and (datetime_recon - datetime_BT_6hr[i]) < (datetime_BT_6hr[i+1] - datetime_BT_6hr[i]):
            match_time_idx = i
   #         match_time = datetime_BT_6hr[i]
   #         time_BT = str(datetime_BT_6hr[i])[-4:]
   #
   # time_recon = str(datetime_recon)[-4:]
   #
   # hour_diff = int(time_recon[:2]) - int(time_BT[:2])
   # min_diff = int(time_recon[2:]) - int(time_BT[2:])
   # time_diff = hour_diff*60 + min_diff


##########
    match_monthdatetime = str(datetime_BT_6hr[match_time_idx])[4:]
    match_month = int(match_monthdatetime[:2])
    match_date = int(match_monthdatetime[2:4])
    match_hour = int(match_monthdatetime[4:6])
    match_total = match_date*24 + match_hour
    #print('match datetime:',match_month,'/', match_date, match_hour,'z')
    #print('match_total',match_total)

    lon_match = df_BT['LON'][match_time_idx]
    lat_match = df_BT['LAT'][match_time_idx]
    max_wnd_match = int(df_BT['WIND'][match_time_idx])
    pres_min_match = df_BT['PRES'][match_time_idx]


    hr12_monthdatetime = str(datetime_BT_6hr[len(df_BT)-1])[4:]
    hr12_idx = len(df_BT)-1

    ###########

    for i in range(match_time_idx,len(df_BT)):
        #print(datetime_BT_6hr[i])

        hr12_monthdatetime = str(datetime_BT_6hr[i])[4:]
        hr12_month = int(hr12_monthdatetime[:2])
        hr12_date = int(hr12_monthdatetime[2:4])
        hr12_hour = int(hr12_monthdatetime[4:6])
        hr12_total = hr12_date*24 + match_hour
        #print('hr12_total', hr12_total)
        if hr12_month == match_month:
            if hr12_total - match_total >= 12:
        #        print('how')
                hr12_idx = i
                break
        if hr12_month - match_month == 1:
        #    print('what')
            if 24 + match_hour - hr12_hour >= 12:
                hr12_idx = i
                break


    lon_hr12 = df_BT['LON'][hr12_idx]
    lat_hr12 = df_BT['LAT'][hr12_idx]
    max_wnd_hr12 = int(df_BT['WIND'][hr12_idx])
    pres_min_hr12 = df_BT['PRES'][hr12_idx]

    print('match time info:   ',match_monthdatetime, lon_match, lat_match, max_wnd_match, pres_min_match)
    print('12hours later:     ',hr12_monthdatetime, lon_hr12, lat_hr12, max_wnd_hr12, pres_min_hr12)

    #print(type(max_wnd_hr12))
    delta_I = max_wnd_hr12- max_wnd_match
    if delta_I > 10:
        cat_delta_i = 'Rapid Intensification'
        color = 'tab:red'
    elif delta_I > 5 and delta_I <= 10:
        cat_delta_i = 'Intensification'
        color = 'tab:orange'
    elif delta_I >= -5 and delta_I <= 5:
        cat_delta_i = 'Steady State'
        color = 'tab:cyan'
    elif delta_I <= -10:
        cat_delta_i = 'Decreasing'
        color = 'tab:green'

    #print('Hurricane Stage:',cat_delta_i)
    return cat_delta_i, color

def xy2deg(x,y):
    """
    Calculate the rotation (degree) of (x,y) to (0,0)
    Due North is 0 deg and increase by anti-cyclonic rotation (clockwise)

    Parameters
    ---------
    x: float
      First argument in Cartesian coordinates.
    y: float
      Second argument in Cartesian coordinates.

    Returns
    ------
    pos_dir: float
      Direction of (x,y) to (0,0) in Cartesian coord (unit: degree)


    """
    if x >= 0 and y > 0:     # [0,pi/2)
        pos_dir = np.arctan(x/y)

    elif x > 0 and y <= 0:   # [pi/2,pi)
        pos_dir = np.arctan(-y/x) + np.pi/2

    elif x <= 0 and y < 0:   # [pi,3pi/2)
        pos_dir = np.arctan(x/y) + np.pi

    elif x < 0 and y >= 0:   # [3pi/2,2pi)
        pos_dir = np.arctan(-y/x) + np.pi*3/2

    elif y == 0 and x == 0:
        pos_dir = 0

    return np.rad2deg(pos_dir)


def id_quadrant_wrong(moving_direction, pos_to_center_ds):
    """
    Identify relative quadrants of dropsonde to storm moving direction.

    Parameters
    ---------
    moving_direction: float
      Storm moving direction (degree). (0: East, 90: North, etc.)
    pos_to_center_ds: float
      Dropsonde position (direction) to the storm center. (0: East, 90: North, etc.)

    Returns
    ------
    qua: int
      Relative quadrant of dropsonde to storm moving direction.
      qaudrant 1 : northeast, Right-front
      qaudrant 2 : southeast, Right-rear
      qaudrant 3 : southwest, Left-rear
      qaudrant 4 : northwest, Left-front

    """

    east_mv = moving_direction - 90
    north_mv = moving_direction
    west_mv = moving_direction + 90
    south_mv = moving_direction + 180
    enws = [east_mv, north_mv, west_mv, south_mv]
    for i in enws:
        if i > 360:
            i -= 360
        if i < 0:
            i += 360
    try:
        quad = np.where(pos_to_center_ds-enws > 0)[0][-1] + 1
    except IndexError:
        quad = 4

    return quad

def id_quadrant(moving_direction, pos_to_center_ds):
    pos_to_center_adjusted = pos_to_center_ds - moving_direction
    nesw = [0, 90, 180, 270]

    if pos_to_center_adjusted > 360:
        pos_to_center_adjusted -= 360
    if pos_to_center_adjusted < 0:
        pos_to_center_adjusted += 360

    nesw = np.asarray(nesw)
    quad = np.where(pos_to_center_adjusted-nesw >= 0)[0][-1] + 1
    return quad


def matchtime_and_h6_in_BT(recon_info, df_BT, interv=6):
    """
    Find the coresponding dropsonde release time (datetime_recon) in Best Track file. t-6 < datetime_recon < t.
    Find next timestamp in Best Track file, 6 hours later or more.
    The information can be used to calculate the storm moving direction, and identify storm stage (e.g. RI), given 12 hours interval

    Parameters
    ---------
    recon_info: string
      dropsonde information (e.g. g991835112_199908282135_DENNIS_-76.14_30.1)
    df_BT: pandas DataFrame
      Best Track dataframe
    interv: int, optional
      6 hrs or 12 hrs.
      By default, interv is 6hrs

    Returns
    ------
    match_time_idx: int
      index of matchtime in Best Track file
    hr6_idx: int
      index that 6 (or 12) hours after the matchtime in Best Track file
    """

    datetime_recon = grab_recon_datetime(recon_info)
    datetime_BT_6hr = df_BT['DATETIME']

    ############  datetime_BT_6hr[i] is to find where the hurricane is from t-6 to t  ##############
    for i in range(len(df_BT)-1):
        if (datetime_recon - datetime_BT_6hr[i]) >= 0 and ((datetime_recon - datetime_BT_6hr[i]) < (datetime_BT_6hr[i+1] - datetime_BT_6hr[i])):
            match_time_idx = i

    match_monthdatetime = str(datetime_BT_6hr[match_time_idx])[4:]
    match_month = int(match_monthdatetime[:2])
    match_date = int(match_monthdatetime[2:4])
    match_hour = int(match_monthdatetime[4:6])
    match_total = match_date*24 + match_hour
    #print('match datetime:',match_month,'/', match_date, match_hour,'z')
    #print('match_total',match_total)



                ############  hr6_  where the hurricane is from t to t+6    ##############
    hr6_monthdatetime = str(datetime_BT_6hr[len(df_BT)-1])[4:]
    hr6_idx = len(df_BT)-1

    for i in range(match_time_idx,len(df_BT)):
        #print(datetime_BT_6hr[i])

        hr6_monthdatetime = str(datetime_BT_6hr[i])[4:]
        hr6_month = int(hr6_monthdatetime[:2])
        hr6_date = int(hr6_monthdatetime[2:4])
        hr6_hour = int(hr6_monthdatetime[4:6])
        hr6_total = hr6_date*24 + hr6_hour
        #print('hr6_total', hr6_total)
        if hr6_month == match_month:            # sonde is dropped before 18z of the last day of the month
            if hr6_total - match_total >= interv:
                hr6_idx = i
                break
        if hr6_month - match_month == 1:        # sonde is dropped after 18z of the last day of the month
            if 24 + match_hour - hr6_hour >= interv:
                hr6_idx = i
                break

    return match_time_idx, hr6_idx

def utc2lst(time_utc,lon):
    timezone_lon = np.array([-172.5, -157.5, -142.5, -127.5, -112.5,  -97.5,  -82.5,  -67.5,
                    -52.5,  -37.5,  -22.5,   -7.5,    7.5,   22.5,   37.5,   52.5,
                    67.5,   82.5,   97.5,  112.5,  127.5,  142.5,  157.5,  172.5, 187.5])

    tz_plus = np.array([-11, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,
                1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12, -12])

    if lon < -172.5:
        lon += 360

    idx = np.where(lon>timezone_lon)[0][-1]
    time_lst = time_utc + tz_plus[idx]
    if time_lst >= 24:
        time_lst -= 24
    if time_lst < 0:
        time_lst += 24

    return time_lst
