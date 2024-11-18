import math
import os
import h5py
import pickle
import random
import pathlib
import parameters
import numpy as np
import scipy as sci
from os import path
import pandas as pd
from math import sqrt
from datetime import datetime
from scipy.special import stdtr
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
# El Nino imports
import matplotlib
import netCDF4
from netCDF4 import Dataset,num2date
from matplotlib import pyplot as plt
import xarray as xr
from tigramite import data_processing as pp


np.random.seed(1)
pars = parameters.get_flux_params()

win_size = pars.get("win_size")
training_length = pars.get("train_len")
prediction_length = pars.get("pred_len")


def plot_boxplots(methods_metrics_dict, plot_path, csv_filename="method_metrics.csv"):

    """
    This function takes a dictionary of methods with their metrics from multiple experiments, 
    creates boxplots for each metric, saves them as PDFs, and saves the dictionary as a CSV.

    Parameters:
        methods_metrics_dict (dict): Dictionary where each key is a method name, and value is 
                                        another dictionary of experiment metrics.
        plot_path (str): Directory where the PDF files will be saved.
        csv_filename (str): Name of the CSV file to save the data.
    """
    # Ensure the plot path exists
    os.makedirs(plot_path, exist_ok=True)

    # Flatten the data for conversion to a DataFrame
    flattened_data = []

    for method, experiments in methods_metrics_dict.items():
        for experiment, metrics in experiments.items():
            for metric, value in metrics.items():
                flattened_data.append({"Method": method, "Experiment": experiment, "Metric": metric, "Value": value})

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(flattened_data)

    # Step 2: Save DataFrame to CSV in the original form
    csv_full_path = os.path.join(plot_path, csv_filename)
    df.to_csv(csv_full_path, index=False)
    print(f"CSV file saved as '{csv_full_path}'")

    # Step 3: Plot and save boxplots as PDF files for each metric
    metrics = df["Metric"].unique()
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        # Filter DataFrame for current metric
        metric_data = df[df["Metric"] == metric]
        # Create a boxplot with Method on the x-axis and Value on the y-axis
        metric_data.boxplot(column="Value", by="Method", grid=False)
        plt.title(f"Boxplot for {metric}")
        plt.suptitle("")  # Remove the automatic 'by' title
        plt.xlabel("Method")
        plt.ylabel(metric)

        # Construct full PDF path and save the plot
        pdf_filename = f"boxplot_{metric}.pdf"
        pdf_full_path = os.path.join(plot_path, pdf_filename)
        plt.savefig(pdf_full_path, format="pdf")
        print(f"Boxplot for {metric} saved as '{pdf_full_path}'")
        plt.close()


def generate_causal_graph(n):
    # Create an n x n array with random 0s and 1s
    array = np.random.randint(0, 2, size=(n, n))
    
    # Set the diagonal elements to 1
    np.fill_diagonal(array, 1)

    # Remove upper triangle (set to 0)
    for i in range(n):
        for j in range(i + 1, n):
            array[i][j] = 0
            
    return array

def calculate_shd(ground_truth, predicted):
    """
    Calculates the Structural Hamming Distance (SHD) between two DAG adjacency matrices.
    SHD is the number of differing edges between the ground truth and predicted matrices.
    
    :param ground_truth: np.ndarray - Ground truth adjacency matrix (n x n).
    :param predicted: np.ndarray - Predicted adjacency matrix (n x n).
    :return: int - The SHD between the two matrices.
    """
    if ground_truth.shape != predicted.shape:
        raise ValueError("Both matrices must have the same shape")
    
    # Element-wise comparison (1 where they differ, 0 where they are the same)
    differences = ground_truth != predicted
    
    # Sum the number of differing edges (the number of 1s in the differences matrix)
    shd = np.sum(differences)
    
    # Calculate the maximum possible number of edges (n * (n - 1) for an n x n adjacency matrix without self-loops)
    n = ground_truth.shape[0]
    max_edges = n * (n - 1)
    
    # Calculate normalized SHD
    normalized_shd = shd / max_edges
    
    return shd

# Function to compute metrics for each predicted graph and find the best one
def evaluate_best_predicted_graph(actual, predicted_list):
    best_metrics = None
    best_f1_score = -1  # Initialize with a low F1 score
    best_tpr = -1       # Initialize TPR for tiebreakers
    best_fpr = float('inf')  # Initialize FPR with high value for tiebreakers
    
    # Flatten actual graph once, since it's common for all predictions
    y_true_flat = [item for sublist in actual for item in sublist]
    
    for predicted in predicted_list:
        # Flatten predicted graph
        y_pred_flat = [item for sublist in predicted for item in sublist]

        # Calculate confusion matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Calculate metrics
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0  # True Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) != 0 else 0  # True Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) != 0 else 0  # False Negative Rate
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        shd = calculate_shd(np.array(actual), np.array(predicted))

        # Check if this prediction has the highest F1 score, or if tied, check TPR and FPR
        if (f1_score > best_f1_score or
            (f1_score == best_f1_score and tpr > best_tpr) or
            (f1_score == best_f1_score and tpr == best_tpr and fpr < best_fpr)):
            
            # Update the best metrics
            best_f1_score = f1_score
            best_tpr = tpr
            best_fpr = fpr
            
            best_metrics = {
                'TPR': tpr,
                'TNR': tnr,
                'FPR': fpr,
                'FNR': fnr,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'Fscore': f1_score,
                'SHD': shd
            }
    
    return best_metrics


def get_ground_truth(matrix, group_sizes):
    # Convert the input matrix to a numpy array
    matrix = np.array(matrix)
    print(f'Test matrix: {matrix}')
    # Determine the number of groups
    num_groups = len(group_sizes)
    
    # Initialize the reduced matrix with zeros
    reduced_matrix = np.zeros((num_groups, num_groups), dtype=int)
    
    # Determine the indices that belong to each group
    groups = []
    start_idx = 0
    print(f'Group sizes: {group_sizes}')
    for size in group_sizes:
        groups.append(list(range(start_idx, start_idx + size)))
        start_idx += size
    print(f'Test groups: {groups}')
    # Fill in the reduced matrix
    for i, group_i in enumerate(groups):
        for j, group_j in enumerate(groups):
            for var_i in group_i:
                for var_j in group_j:
                    if matrix[var_j, var_i] != 0:  # Check if var_j causes var_i
                        reduced_matrix[j, i] = 1  # Since rows are causes, update [j, i]
                        break  # No need to check further if one causal relationship is found
    
    # Ensure all groups have self-connection
    np.fill_diagonal(reduced_matrix, 1)
    return reduced_matrix


# Function to convert timestamp to formatted date
def convert_timestamp(timestamp):
    # Assuming the timestamp format is 'YYMMDD'
    timestamp = str(timestamp)
    year = int(timestamp[:4])
    month = int(timestamp[4:6])
    day = int(timestamp[6:8])
    hour = int(timestamp[8:10])
    min = int(timestamp[10:12])

    # Convert to datetime object
    date_obj = datetime(year, month, day, hour, min)

    # Format the datetime object as 'DD-Mon-YYYY'
    formatted_date = date_obj.strftime('%d-%b-%Y %H:%M')

    return formatted_date

def get_shuffled_ts(SAMPLE_RATE, DURATION, root):
    # Number of samples in normalized_tone
    N = SAMPLE_RATE * DURATION
    yf = rfft(root)
    xf = rfftfreq(N, 1 / SAMPLE_RATE)
    # plt.plot(xf, np.abs(yf))
    # plt.show()
    new_ts = irfft(shuffle(yf))
    return new_ts


def get_ground_truth(matrix, group_sizes):
    # Convert the input matrix to a numpy array
    matrix = np.array(matrix)
    
    # Determine the number of groups
    num_groups = len(group_sizes)
    
    # Initialize the reduced matrix with zeros
    reduced_matrix = np.zeros((num_groups, num_groups), dtype=int)
    
    # Determine the indices that belong to each group
    groups = []
    start_idx = 0
    for size in group_sizes:
        groups.append(list(range(start_idx, start_idx + size)))
        start_idx += size
    
    # Fill in the reduced matrix
    for i, group_i in enumerate(groups):
        for j, group_j in enumerate(groups):
            for var_i in group_i:
                for var_j in group_j:
                    if matrix[var_j, var_i] != 0:  # Check if var_j causes var_i
                        reduced_matrix[j, i] = 1  # Since rows are causes, update [j, i]
                        break  # No need to check further if one causal relationship is found
    
    # Ensure all groups have self-connection
    np.fill_diagonal(reduced_matrix, 1)
    return reduced_matrix


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
    
    average_discharges['Date'] = pd.to_datetime(average_discharges['Date'])
    average_discharges.set_index('Date', inplace=True)
    df = average_discharges.apply(normalize)

    return df


def load_climate_data():
    # Load climate discharges data
    
    df = pd.read_csv('/home/ahmad/Projects/gCause/datasets/environment_dataset/light.txt', sep=" ", header=None)
    df.columns = ["NEP", "PPFD"]
    df = df.apply(normalize)

    return df


def read_ground_truth(file_path):
    """
    Reads a CSV file containing binary data with extra headers and index columns,
    discarding the first two rows and the first two columns, and returns a numpy array
    of the binary data.
    
    Parameters:
    - file_path (str): The path to the CSV file.
    
    Returns:
    - np.ndarray: A 2D numpy array containing the binary data.
    """
    # Read the CSV file, starting from the third row, and use the first column as the index
    df = pd.read_csv(file_path, header=1, index_col=0)
    
    # Drop the first column to get only the binary data
    binary_data_df = df.iloc[:, 1:]

    # Convert the DataFrame to a numpy array of integers
    binary_data = binary_data_df.to_numpy().astype(int)
    
    return binary_data.T



def load_rivernet(river):
    
    # Load river discharges data
    path_data = f'/home/ahmad/Projects/gCause/datasets/rivernet/{river}_data.csv'
    path_ground_truth = f'/home/ahmad/Projects/gCause/datasets/rivernet/{river}_label.csv'

    data = pd.read_csv(path_data)

    data['datetime'] = pd.to_datetime(data['datetime'])
    # Set datetime as the index
    data.set_index('datetime', inplace=True)
    # Resample the data to daily (24-hour) resolution, using the mean to aggregate values
    # data = data.resample('D').mean()
    data = data.resample('16H').mean()
    
    ground_truth = read_ground_truth(path_ground_truth)
    # np.fill_diagonal(ground_truth, 1)
    print(f'Ground truth: \n {ground_truth}')
    # data = data.set_index('datetime')

    check_trailing_nans = np.where(data.isnull().values.any(axis=1) == 0)[0]
    data = data[check_trailing_nans.min() : check_trailing_nans.max()+1]
    # assert data.isnull().sum().max() == 0, "Check nans!"
    data.interpolate(inplace=True)

    # Apply seasonal differencing (lag = 365) to all columns
    df_diff = data.copy()  # Copy original DataFrame to preserve it

    for column in data.columns:
        # Apply seasonal differencing to each column
        df_diff[column] = data[column] - data[column].shift(548)

    # Drop NaN values caused by shifting (from the first 365 days)
    df_diff.dropna(inplace=True)

    # # Plot the original and differenced data for each column
    # plt.figure(figsize=(12, 8))

    # for i, column in enumerate(data.columns, 1):
    #     plt.subplot(len(data.columns), 1, i)
    #     plt.plot(data[column], label=f'Original {column}', color='blue', alpha=0.7)
    #     plt.plot(df_diff[column], label=f'Differenced {column}', color='orange', linestyle='--')
    #     plt.legend()
    #     plt.title(f"Seasonal Differencing for Column {column}")

    # plt.tight_layout()
    # plt.show()

    # Display the differenced data (to check results)
    df = df_diff.apply(normalize)
    print(df)

    return df, ground_truth, ground_truth # get_ground_truth(generate_causal_graph(len(vars)-1), [4, 2])


def load_geo_data(start, end):
    # Load goeclimate data
    path = '/home/ahmad/Projects/gCause/datasets/geo_dataset/moxa_data_H.csv'
    # vars = ['DateTime', 'rain', 'temperature_outside', 'pressure_outside', 'gw_mb',
    #    'gw_sr', 'gw_sg', 'gw_west', 'gw_knee', 'gw_south', 'wind_x', 'winx_y',
    #    'snow_load', 'humidity', 'glob_radiaton', 'strain_ew_uncorrected',
    #    'strain_ns_uncorrected', 'strain_ew_corrected', 'strain_ns_corrected',
    #    'tides_ew', 'tides_ns']

    # groundwater group: ['gw_mb', 'gw_sg', , 'gw_sr', 'gw_west', 'gw_knee', 'gw_south']
    # climate group: ['temperature_outside', 'pressure_outside', 'wind_x', 'winx_y', 'humidity', 'glob_radiaton']
    # strain group: ['strain_ew_corrected', 'strain_ns_corrected'] 
    vars = ['DateTime', 'temperature_outside', 'pressure_outside', 'wind_x', 'glob_radiaton', 'gw_mb', 'gw_west', 'strain_ew_corrected', 'strain_ns_corrected']
    # vars = ['DateTime', 'temperature_outside', 'pressure_outside', 'wind_x', 'glob_radiaton', 'strain_ew_corrected', 'strain_ns_corrected']
    # vars = ['DateTime', 'temperature_outside', 'pressure_outside', 'wind_x', 'snow_load', 'strain_ew_corrected', 'strain_ns_corrected']
    data = pd.read_csv(path, usecols=vars)
    
    # # Read spring and summer season geo-climatic data
    # start_date = '2014-11-01'
    # end_date = '2015-05-28'
    # mask = (data['DateTime'] > start_date) & (data['DateTime'] <= end_date)  # '2015-06-30') Regime 1
    # # mask = (data['DateTime'] > '2015-05-01') & (data['DateTime'] <= '2015-10-30')  # Regime 2
    # data = data.loc[mask]
    data = data.fillna(method='pad')
    data = data.set_index('DateTime')
    data = data.iloc[start: end]
    data = data.apply(normalize)
    print(data.describe())

    return data, np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]]), generate_causal_graph(len(vars)-1) # get_ground_truth(generate_causal_graph(len(vars)-1), [4, 2])


def load_hackathon_data():
    # Load river discharges data
    bot, bov = simple_load_csv("/home/ahmad/Projects/gCause/datasets/hackathon_data/blood-oxygenation_interpolated_3600_pt_avg_14.csv")
    wt, wv = simple_load_csv("/home/ahmad/Projects/gCause/datasets/hackathon_data/weight_interpolated_3600_pt_avg_6.csv")
    hrt, hrv = simple_load_csv("/home/ahmad/Projects/gCause/datasets/hackathon_data/resting-heart-rate_interpolated_3600_iv_avg_4.csv")
    st, sv = simple_load_csv("/home/ahmad/Projects/gCause/datasets/hackathon_data/step-amount_interpolated_3600_iv_ct_15.csv")
    it, iv = simple_load_csv("/home/ahmad/Projects/gCause/datasets/hackathon_data/in-bed_interpolated_3600_iv_sp_19.csv")
    at, av = simple_load_csv("/home/ahmad/Projects/gCause/datasets/hackathon_data/awake_interpolated_3600_iv_sp_18.csv")

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
                        #------------------------------------------------------------------------

                        jfm_Yregion = anomalies_Yregion.sel(time=is_jfm(xdata['time.month']))
                        jfm_Yregion_by_year = jfm_Yregion.groupby("time.year").mean()
                        num_jfm_Yregion = np.array(jfm_Yregion_by_year.to_array())[0]
                        reshaped_Yregion = np.reshape(num_jfm_Yregion, newshape = (num_jfm_Yregion.shape[0],num_jfm_Yregion.shape[1]*num_jfm_Yregion.shape[2]))

                        #Consider cases where group sizes are not further apart than 10 grid boxes
                        #------------------------------------------------------------------------
                        if abs(reshaped_Xregion.shape[1]-reshaped_Yregion.shape[1])<12:

                            #GAUSSIAN KERNEL SMOOTHING
                            #-----------------------------------------------
                            for var in range(reshaped_Xregion.shape[1]):
                                reshaped_Xregion[:, var] = pp.smooth(reshaped_Xregion[:, var], smooth_width=12*10, kernel='gaussian', mask=None,
                                                            residuals=True)
                            for var in range(reshaped_Yregion.shape[1]):
                                reshaped_Yregion[:, var] = pp.smooth(reshaped_Yregion[:, var], smooth_width=12*10, kernel='gaussian', mask=None,
                                                            residuals=True)
                            # ----------------------------------------------
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


def load_flux_data(start, end):


    # ------------ Climate Ecosystem Variables -------------------------
    # Index(['TIMESTAMP_START', 'TIMESTAMP_END', 'TA_F', 'TA_F_QC', 'SW_IN_POT',
    #    'SW_IN_F', 'SW_IN_F_QC', 'LW_IN_F', 'LW_IN_F_QC', 'VPD_F', 'VPD_F_QC',
    #    'PA_F', 'PA_F_QC', 'P_F', 'P_F_QC', 'WS_F', 'WS_F_QC', 'WD', 'USTAR',
    #    'RH', 'NETRAD', 'PPFD_IN', 'PPFD_DIF', 'SW_OUT', 'LW_OUT', 'CO2_F_MDS',
    #    'CO2_F_MDS_QC', 'TS_F_MDS_1', 'TS_F_MDS_1_QC', 'G_F_MDS', 'G_F_MDS_QC',
    #    'LE_F_MDS', 'LE_F_MDS_QC', 'LE_CORR', 'LE_CORR_25', 'LE_CORR_75',
    #    'LE_RANDUNC', 'H_F_MDS', 'H_F_MDS_QC', 'H_CORR', 'H_CORR_25',
    #    'H_CORR_75', 'H_RANDUNC', 'NIGHT', 'NEE_VUT_REF', 'NEE_VUT_REF_QC',
    #    'NEE_VUT_REF_RANDUNC', 'NEE_VUT_25', 'NEE_VUT_50', 'NEE_VUT_75',
    #    'NEE_VUT_25_QC', 'NEE_VUT_50_QC', 'NEE_VUT_75_QC', 'RECO_NT_VUT_REF',
    #    'RECO_NT_VUT_25', 'RECO_NT_VUT_50', 'RECO_NT_VUT_75', 'GPP_NT_VUT_REF',
    #    'GPP_NT_VUT_25', 'GPP_NT_VUT_50', 'GPP_NT_VUT_75', 'RECO_DT_VUT_REF',
    #    'RECO_DT_VUT_25', 'RECO_DT_VUT_50', 'RECO_DT_VUT_75', 'GPP_DT_VUT_REF',
    #    'GPP_DT_VUT_25', 'GPP_DT_VUT_50', 'GPP_DT_VUT_75', 'RECO_SR',
    #    'RECO_SR_N'],
    #   dtype='object')
    # ------------------------------------------------------------------

    # "Load fluxnet 2015 data for various sites"
    USTon = 'FLX_US-Ton_FLUXNET2015_SUBSET_2001-2014_1-4/FLX_US-Ton_FLUXNET2015_SUBSET_HH_2001-2014_1-4.csv'
    FRPue = 'FLX_FR-Pue_FLUXNET2015_SUBSET_2000-2014_2-4/FLX_FR-Pue_FLUXNET2015_SUBSET_HH_2000-2014_2-4.csv'
    DEHai = 'FLX_DE-Hai_FLUXNET2015_SUBSET_2000-2012_1-4/FLX_DE-Hai_FLUXNET2015_SUBSET_HH_2000-2012_1-4.csv'
    ITMBo = 'FLX_IT-MBo_FLUXNET2015_SUBSET_2003-2013_1-4/FLX_IT-MBo_FLUXNET2015_SUBSET_HH_2003-2013_1-4.csv'

    # Calculate the number of rows to read
    num_rows = end - start + 1
    # Define the rows to skip, excluding the header row
    rows_to_skip = list(range(1, start)) if start != 0 else start
    
    start_date = '15-Jun-2003 00:00'
    end_date ='15-Aug-2003 23:30'
    # col_list = ['TIMESTAMP_START', 'SW_IN_POT', 'SW_IN_F', 'TA_F', 'TA_F_QC']
    col_list = ['TIMESTAMP_START', 'SW_IN_F', 'TA_F', 'GPP_NT_VUT_50', 'RECO_NT_VUT_50', 'NEE_VUT_50']
    # Convert the 'date' column to datetime objects
    
    fluxnet = pd.read_csv("/home/ahmad/Projects/gCause/datasets/fluxnet2015/" + FRPue, usecols=col_list, skiprows=rows_to_skip, nrows=num_rows)
    # ----------------------------------------------
   
    fluxnet['TIMESTAMP_START'] = fluxnet['TIMESTAMP_START'].apply(convert_timestamp)

    fluxnet['TIMESTAMP_START'] = pd.to_datetime(fluxnet['TIMESTAMP_START'])

    # fluxnet = fluxnet[(fluxnet['TIMESTAMP_START'] >= start_date) & (fluxnet['TIMESTAMP_START'] <= end_date)][col_list]
    fluxnet.set_index('TIMESTAMP_START', inplace=True)
    # fluxnet = fluxnet.iloc[start: end]
    # ----------------------------------------------
    # data = {'Rg': rg[start: end], 'T': temp[start: end], 'GPP': gpp[start: end], 'Reco': reco[start: end]}
    # df = pd.DataFrame(data, columns=['Rg', 'T', 'GPP', 'Reco'])
    fluxnet = fluxnet.apply(normalize)
    return fluxnet

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

def generate_variable_list(N):
    """
    Generates a list of variable names in the format Var$_i$ where i ranges from 1 to N.
    
    Parameters:
    N (int): The number of variables to generate.
    
    Returns:
    list: A list of variable names formatted as Var$_i$.
    """
    return [f'Var$_{i}$' for i in range(1, N + 1)]


def load_netsim_data():

    # Load data from a .npz file
    file_path = r'../datasets/netsim/sim3_subject_4.npz'
    loaded_data = np.load(file_path)

    n = loaded_data['n.npy']
    T = loaded_data['T.npy']
    Gref = loaded_data['Gref.npy']
    # Access individual arrays within the .npz file
    nvars = 15
    cols = generate_variable_list(nvars)
    data = loaded_data['X_np.npy']
    data = data.transpose()
    df = pd.DataFrame(data[:, 0:nvars], columns=cols)
    df = df.apply(normalize)
    return df

def load_sims_data(groups):
    # Load .mat file
    mat_file_path = r'../datasets/sims/sim4.mat'
    mat_data = sci.io.loadmat(mat_file_path)
    dim = groups*5
    cols = [f'Z{i}' for i in range(1, dim+1)]
    df = pd.DataFrame(data= mat_data['ts'][: 200, :dim], columns=cols)
    df = df.apply(normalize)

    cgraphs = mat_data['net'][0, :dim, :dim].T

    return df, cgraphs