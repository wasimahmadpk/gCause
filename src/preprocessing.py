import math
import h5py
import pickle
import random
import pathlib
import parameters
import numpy as np
from os import path
import pandas as pd
from math import sqrt
from datetime import datetime
from scipy.special import stdtr
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression, mutual_info_regression

# El Nino imports
import matplotlib
import netCDF4
from netCDF4 import Dataset,num2date
import datetime
from matplotlib import pyplot as plt
import xarray as xr
from tigramite import data_processing as pp


np.random.seed(1)
pars = parameters.get_geo_params()

win_size = pars.get("win_size")
training_length = pars.get("train_len")
prediction_length = pars.get("pred_len")


def get_shuffled_ts(SAMPLE_RATE, DURATION, root):
    # Number of samples in normalized_tone
    N = SAMPLE_RATE * DURATION
    yf = rfft(root)
    xf = rfftfreq(N, 1 / SAMPLE_RATE)
    # plt.plot(xf, np.abs(yf))
    # plt.show()
    new_ts = irfft(shuffle(yf))
    return new_ts


def deseasonalize(var, interval):
    deseasonalize_data = []
    for i in range(interval, len(var)):
        value = var[i] - var[i - interval]
        deseasonalize_data.append(value)
    return deseasonalize_data


def running_avg_effect(y, yint):

#  Break temporal dependency and generate a new time series
    pars = parameters.get_sig_params()
    SAMPLE_RATE = pars.get("sample_rate")  # Hertz
    DURATION = pars.get("duration")  # Seconds
    rae = 0
    for i in range(len(y)):
        ace = 1/((training_length + 1 + i) - training_length) * (rae + (y[i] - yint[i]))
    return rae


# Normalization (MixMax/ Standard)
def normalize(data, type='minmax'):

    if type == 'std':
        return (np.array(data) - np.mean(data))/np.std(data)

    elif type == 'minmax':
        return (np.array(data) - np.min(data))/(np.max(data) - np.min(data))


def down_sample(data, win_size, partition=None):
    agg_data = []
    daily_data = []
    for i in range(len(data)):
        daily_data.append(data[i])

        if (i % win_size) == 0:

            if partition == None:
                agg_data.append(sum(daily_data) / win_size)
                daily_data = []
            elif partition == 'gpp':
                agg_data.append(sum(daily_data[24: 30]) / 6)
                daily_data = []
            elif partition == 'reco':
                agg_data.append(sum(daily_data[40: 48]) / 8)
                daily_data = []
    return agg_data


def SNR(s, n):
    Ps = np.sqrt(np.mean(np.array(s) ** 2))
    Pn = np.sqrt(np.mean(np.array(n) ** 2))
    SNR = Ps / Pn
    return 10 * math.log(SNR, 10)


def mean_absolute_percentage_error(y_true, y_pred):

    return np.mean(np.abs((y_true - y_pred) / y_true))


def mutual_information(x, y):
    mi = mutual_info_regression(x, y)
    mi /= np.max(mi)
    return mi


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def load_river_data():
    # Load river discharges data
    stations = ["dillingen", "kempten", "lenggries"]
        # Read the average daily discharges at each of these stations and combine them into a single pandas dataframe
    average_discharges = None

    for station in stations:

        filename = pathlib.Path("../datasets/river_discharge_data/data_" + station + ".csv")
        new_frame = pd.read_csv(filename, sep=";", skiprows=range(10))
        new_frame = new_frame[["Datum", "Mittelwert"]]

        new_frame = new_frame.rename(columns={"Mittelwert": station.capitalize(), "Datum": "Date"})
        new_frame.replace({",": "."}, regex=True, inplace=True)

        new_frame[station.capitalize()] = new_frame[station.capitalize()].astype(float)

        if average_discharges is None:
            average_discharges = new_frame
        else:
            average_discharges = average_discharges.merge(new_frame, on="Date")
    
    dillingen = average_discharges.iloc[:, 1].tolist()
    kempton = average_discharges.iloc[:, 2].tolist()
    lenggries = average_discharges.iloc[:, 3].tolist()

    data = {'Kt': kempton, 'Dt': dillingen, 'Lt': lenggries}
    df = pd.DataFrame(data, columns=['Kt', 'Dt', 'Lt'])
    df = df.apply(normalize)

    return df


def load_climate_data():
    # Load river discharges data
    
    df = pd.read_csv('/home/ahmad/Projects/gCause/datasets/environment_dataset/light.txt', sep=" ", header=None)
    df.columns = ["NEP", "PPFD"]
    df = df.apply(normalize)

    return df


def load_geo_data():
    # Load river discharges data
    path = '/home/ahmad/Projects/gCause/datasets/geo_dataset/moxa_data_D.csv'
    # vars = ['DateTime', 'rain', 'temperature_outside', 'pressure_outside', 'gw_mb',
    #    'gw_sr', 'gw_sg', 'gw_west', 'gw_knee', 'gw_south', 'wind_x', 'winx_y',
    #    'snow_load', 'humidity', 'glob_radiaton', 'strain_ew_uncorrected',
    #    'strain_ns_uncorrected', 'strain_ew_corrected', 'strain_ns_corrected',
    #    'tides_ew', 'tides_ns']
    vars = ['DateTime', 'gw_mb', 'gw_sg', 'gw_knee', 'gw_south', 'strain_ew_corrected', 'strain_ns_corrected']
    # vars = ['DateTime', 'temperature_outside', 'pressure_outside', 'wind_x', 'snow_load', 'strain_ew_corrected', 'strain_ns_corrected']
    data = pd.read_csv(path, usecols=vars)
    
    # Read spring and summer season geo-climatic data
    start_date = '2016-05-15'
    end_date = '2017-05-15'
    # mask = (data['DateTime'] > '2014-11-01') & (data['DateTime'] <= '2015-05-28')  # '2015-06-30') Regime 1
    # mask = (data['DateTime'] > '2015-05-01') & (data['DateTime'] <= '2015-10-30')  # Regime 2
    # data = data.loc[mask]
    data = data.fillna(method='pad')
    data = data.set_index('DateTime')
    # data = data[start_date: ]
    data = data.apply(normalize)

    return data


def load_hackathon_data():
    # Load river discharges data
    bot, bov = simple_load_csv("/home/ahmad/Projects/gCause/datasets/hackathon_data/blood-oxygenation_interpolated_3600_pt_avg_14.csv")
    wt, wv = simple_load_csv("/home/ahmad/Projects/gCause/datasets/hackathon_data/weight_interpolated_3600_pt_avg_6.csv")
    hrt, hrv = simple_load_csv("/home/ahmad/Projects/gCause/datasets/hackathon_data/resting-heart-rate_interpolated_3600_iv_avg_4.csv")
    st, sv = simple_load_csv("/home/ahmad/Projects/gCause/datasets/hackathon_data/step-amount_interpolated_3600_iv_ct_15.csv")
    it, iv = simple_load_csv("/home/ahmad/Projects/gCause/datasets/hackathon_data/in-bed_interpolated_3600_iv_sp_19.csv")
    at, av = simple_load_csv("/home/ahmad/Projects/gCause/datasets/hackathon_data/awake_interpolated_3600_iv_sp_18.csv")

        # plt.plot(bov)
        # plt.plot(wv)
        # plt.plot(hrv)
        # plt.show()

        # v15 = np.nan_to_num(aggregate_avg(ts_15, v_15, 60 * 60))
        # v3 = np.nan_to_num(aggregate_avg(ts_3, v_3, 60 * 60))
        # v2 = np.nan_to_num(aggregate_avg(ts_2, v_2, 60 * 60))
        # v1 = np.nan_to_num(aggregate_avg(ts_1, v_1, 60 * 60))

    data = {'BO': bov[7500:10000], 'WV': wv[7500:10000], 'HR': hrv[7500:10000], 'Step': sv[7500:10000], 'IB': iv[7500:10000], 'Awake': av[7500:10000]}
    df = pd.DataFrame(data, columns=['BO', 'WV', 'HR', 'Step', 'IB', 'Awake'])

    return df

def load_nino_data():

    xdata = xr.open_dataset('/home/ahmad/Projects/gCause/datasets/nino/AirTempData.nc')
    crit_list = []


    for i in range(2,5): # grid coarsening parameter for NINO longitude
        for k in range(1,4): # grid coarsening parameter NINO latitude, smaller range because NINo 3.4 has limited latitudinal grid-boxes 
            for j in range(2,5): # grid coarsening parameter for BCT latitude
                for l in range(2,5): # grid coarsening parameter for BCT longitude
                    # print(k,i,j,l)
                    # if k==1 and i==3 and j==3 and l==2:
                      if k==3 and i==2 and j==3 and l==2:
                        #ENSO LAT 6,-6, LON 190, 240
                        #BCT LAT 65,50 LON 200, 240
                        #TATL LAT 25, 5, LON 305, 325

                        Xregion=xdata.sel(lat=slice(6.,-6.,k), lon = slice(190.,240.,i))
                        Yregion=xdata.sel(lat=slice(65.,50.,j), lon = slice(200.,240.,l))
                    
                        # de-seasonlize
                        #----------------
                        monthlymean = Xregion.groupby("time.month").mean("time")
                        anomalies_Xregion = Xregion.groupby("time.month") - monthlymean
                        Yregion_monthlymean = Yregion.groupby("time.month").mean("time")
                        anomalies_Yregion = Yregion.groupby("time.month") - Yregion_monthlymean

                        # functions to consider triples on months
                        #-----------------------------------------

                        def is_ond(month):
                            return (month >= 9) & (month <= 12)

                        def is_son(month):
                            return (month >= 9) & (month <= 11)

                        def is_ndj(month):
                            return ((month >= 11) & (month <= 12)) or (month==1)

                        def is_jfm(month):
                            return (month >= 1) & (month <= 3)

                        # NINO for oct-nov-dec
                        #--------------------

                        ond_Xregion = anomalies_Xregion.sel(time=is_ond(xdata['time.month']))
                        ond_Xregion_by_year = ond_Xregion.groupby("time.year").mean()
                        num_ond_Xregion = np.array(ond_Xregion_by_year.to_array())[0]
                        print(f'Here is the shape: {num_ond_Xregion.shape}')
                        reshaped_Xregion = np.reshape(num_ond_Xregion, newshape = (num_ond_Xregion.shape[0],num_ond_Xregion.shape[1]*num_ond_Xregion.shape[2]))

                        # BCT for jan-feb-mar
                        #-------------------

                        jfm_Yregion = anomalies_Yregion.sel(time=is_jfm(xdata['time.month']))
                        jfm_Yregion_by_year = jfm_Yregion.groupby("time.year").mean()
                        num_jfm_Yregion = np.array(jfm_Yregion_by_year.to_array())[0]
                        reshaped_Yregion = np.reshape(num_jfm_Yregion, newshape = (num_jfm_Yregion.shape[0],num_jfm_Yregion.shape[1]*num_jfm_Yregion.shape[2]))

                        #Consider cases where group sizes are not further apart than 10 grid boxes
                        #------------------------------------------------------------------------
                        if abs(reshaped_Xregion.shape[1]-reshaped_Yregion.shape[1])<12:

                            #GAUSSIAN KERNEL SMOOTHING
                            #-------------------------
                            for var in range(reshaped_Xregion.shape[1]):
                                reshaped_Xregion[:, var] = pp.smooth(reshaped_Xregion[:, var], smooth_width=12*10, kernel='gaussian', mask=None,
                                                            residuals=True)
                            for var in range(reshaped_Yregion.shape[1]):
                                reshaped_Yregion[:, var] = pp.smooth(reshaped_Yregion[:, var], smooth_width=12*10, kernel='gaussian', mask=None,
                                                            residuals=True)
                            ##################################
                            def shift_by_one(array1, array2, t):
                                if t == 0:
                                    return array1, array2
                                elif t < 0:
                                    s = -t
                                    newarray1 = array1[:-s, :]
                                    newarray2 = array2[s:, :]
                                    return newarray1, newarray2

                                else:
                                    newarray1 = array1[t:, :]
                                    newarray2 = array2
                                    return newarray1, newarray2

                            shifted_Yregion, shifted_Xregion = shift_by_one(reshaped_Yregion,reshaped_Xregion, 1)
                            print(f'X : {shifted_Xregion.shape}, Y: {shifted_Yregion.shape}')
                            shifted_XregionT = np.transpose(shifted_Xregion)
                            shifted_YregionT = np.transpose(shifted_Yregion)
                            cols = ['ENSO$_1$', 'ENSO$_2$', 'BCT$_1$', 'BCT$_2$']
                            XYregion = np.concatenate((shifted_Xregion[0:72, 0:2], shifted_Yregion[0:72, 0:2]), axis=1)
                            data = pd.DataFrame(data=XYregion, columns=[str(i) for i in range(XYregion.shape[1])]) #[str(i) for i in range(XYregion.shape[1])]
                            # df = pd.concat([shifted_Xregion, shifted_Yregion], axis=1)

                            tigra_Xregion = pp.DataFrame(shifted_Xregion)
                            tigra_Yregion = pp.DataFrame(shifted_Yregion)
                            print(reshaped_Xregion.shape, reshaped_Yregion.shape)
                            print(shifted_Xregion.shape, shifted_Yregion.shape)
                            
                            # print(f'Number of Nans: {data.isnull().sum()}')
                            df = data.apply(normalize, type='minmax')
                            return df


def load_flux_data():

    # "Load fluxnet 2015 data for various sites"
    # US-Ton : FLX_US-Ton_FLUXNET2015_SUBSET_2001-2014_1-4/FLX_US-Ton_FLUXNET2015_SUBSET_HH_2001-2014_1-4.csv
    # FR-Pue : FLX_FR-Pue_FLUXNET2015_SUBSET_2000-2014_2-4/FLX_FR-Pue_FLUXNET2015_SUBSET_HH_2000-2014_2-4.csv
    # DE-Hai : FLX_DE-Hai_FLUXNET2015_SUBSET_2000-2012_1-4/FLX_DE-Hai_FLUXNET2015_SUBSET_HH_2000-2012_1-4.csv
    # IT-MBo : FLX_IT-MBo_FLUXNET2015_SUBSET_2003-2013_1-4/FLX_IT-MBo_FLUXNET2015_SUBSET_HH_2003-2013_1-4.csv
    fluxnet = pd.read_csv("/home/ahmad/Projects/gCause/datasets/fluxnet2015/FLX_DE-Hai_FLUXNET2015_SUBSET_2000-2012_1-4/FLX_DE-Hai_FLUXNET2015_SUBSET_HH_2000-2012_1-4.csv") 
    org = fluxnet['SW_IN_F']
    otemp = fluxnet['TA_F']
    ovpd = fluxnet['VPD_F']
    # oppt = fluxnet['P_F']
    # nee = fluxnet['NEE_VUT_50']
    ogpp = fluxnet['GPP_NT_VUT_50']
    oreco = fluxnet['RECO_NT_VUT_50']
    
    # ************* LOad FLUXNET2015 data ***************
    rg = normalize(down_sample(org, win_size))
    temp = normalize(down_sample(otemp, win_size))
    # gpp = normalize(down_sample(nee, win_size, partition='gpp'))
    # reco = normalize(down_sample(nee, win_size, partition='reco'))
    gpp = normalize(down_sample(ogpp, win_size))
    reco = normalize(down_sample(oreco, win_size))
    # ppt = normalize(down_sample(oppt, win_size))
    vpd = normalize(down_sample(ovpd, win_size))
    # swc = normalize(down_sample(oswc, win_size))
    # heat = normalize(down_sample(oheat, win_size))

    data = {'Rg': rg[7000:12000], 'T': temp[7000:12000], 'GPP': gpp[7000:12000], 'Reco': reco[7000:12000]}
    df = pd.DataFrame(data, columns=['Rg', 'T', 'GPP', 'Reco'])
    # df = df.apply(normalize)
    return df

# Load synthetically generated time series
def load_syn_data():
    #****************** Load synthetic data *************************
    data = pd.read_csv("../datasets/synthetic_datasets/synthetic_gts.csv")
    df = data.apply(normalize)
    return df

# Load synthetically generated multi-regime time series
def load_multiregime_data():
    # *******************Load synthetic data *************************
    df = pd.read_csv("../datasets/synthetic_datasets/synthetic_data_regimes.csv")
    # df = df.apply(normalize)
    return df