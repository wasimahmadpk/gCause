import time
import pickle
import pathlib
import parameters
import vgc
import baseline.pcmci as pcmci
import numpy as np
import mxnet as mx
import pandas as pd
from os import path
from math import sqrt
from itertools import islice
import preprocessing as prep
from datetime import datetime
import matplotlib.pyplot as plt
from knockoffs import Knockoffs
from scipy.special import stdtr
from inference import modelTest
from gluonts.trainer import Trainer
from gctest import groupCause
from scms import StructuralCausalModel
from gluonts.evaluation import Evaluator
from sklearn.metrics import mean_squared_error
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepar._network import DeepARTrainingNetwork
from gluonts.evaluation.backtest import make_evaluation_predictions
from scipy.stats import ttest_ind, ttest_ind_from_stats, ttest_1samp
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput

np.random.seed(1)
mx.random.seed(2)

start_time = time.time()
# ------------------ Parameters -------------------------------
pars = parameters.get_syn_params()
dim = pars['dim']
freq = pars["freq"]
epochs = pars["epochs"]
win_size = pars["win_size"]
num_sliding_win = pars["num_sliding_win"]
training_length = pars["train_len"]
prediction_length = pars["pred_len"]
num_samples = pars["num_samples"]
num_layers = pars["num_layers"]
num_cells = pars["num_cells"]
dropout_rate = pars["dropout_rate"]
batch_size = pars["batch_size"]
plot_path = pars["plot_path"]
groups = pars["groups"]
path = pars.get('model_path')

# Calculate average metrics
def calculate_average_metrics(performance_dicts):
    sums = {}
    count = len(performance_dicts)
    
    for metrics in performance_dicts.values():
        for key, value in metrics.items():
            if key not in sums:
                sums[key] = 0
            sums[key] += value
    
    avg_metrics = {key: value / count for key, value in sums.items()}
    return avg_metrics

# Plot metrics
def plot_metrics(performance_dicts, metric_name):
    x = sorted(performance_dicts.keys())
    y = [performance_dicts[param][metric_name] for param in x]

    fig, ax = plt.subplots()
    ax.plot(x, y, label=f'{metric_name}')
    ax.set_xticks(x)
    plt.xlabel('Groups')
    plt.ylabel(metric_name)
    # plt.title(f'{metric_name.capitalize()} vs Parameter Value')
    plt.grid(True)
    plt.legend()
    filename = pathlib.Path(plot_path + f'{metric_name}_groups.pdf')
    plt.savefig(filename)  # Save the figure
    # plt.show()

def generate_group_dicts(num_nodes, num_groups):
    
    groups = {}
    group_sizes = {}
    
    base_group_size = num_nodes // num_groups
    remainder = num_nodes % num_groups
    
    start_idx = 0
    
    for i in range(num_groups):
        group_key = f"g{i+1}"
        
        if i < remainder:
            group_size = base_group_size + 1
        else:
            group_size = base_group_size
        
        end_idx = start_idx + group_size
        groups[group_key] = [start_idx, end_idx]
        group_sizes[group_key] = [group_size]
        
        start_idx = end_idx
    
    result = {
        'group_num': num_groups,
        'groups': groups,
        'groups_size': group_sizes
    }
    
    return result
# ------------- Load river discharges data -------------------
# df = prep.load_river_data()
# df = prep.load_climate_data()
# df = prep.load_geo_data()

# df = prep.load_flux_data()
# print(df.corr())

# # Calculate the cross-correlation for lags ranging from -10 to 10 months difference
# cross_corr = [df['0'].corr(df['5'].shift(i)) for i in range(-10, 10)]
# # Print the cross-correlation values
# print(cross_corr)

# df = data.loc[:1000].copy()
# print(df.head())
# print(df.describe())
# print(df.shape)
# ------------------------------------------------------------
metrics_dict = {}
groups_variation = True
for idense in np.arange(0.5, 0.5 + 0.1, 0.1):
    model_name = pars.get('model_name') + '_idense_full_' + f'{idense}' + '.sav'
    # Run CDMI n times
    # ------------ SCMs----------------------------------------
    # Load synthetic data
    model = StructuralCausalModel()
    num_nodes = 9
    nonlinearity = 0.20
    interaction_density = idense
    df, links, causal_graph = model.generate_multi_var_ts(
        num_nodes, nonlinearity, interaction_density, num_samples=2000)
    df.head()
    # ---------------------------------------------------------

    original_data = []
    # dim = len(df.columns)
    columns = df.columns

    for col in df:
        original_data.append(df[col])

    original_data = np.array(original_data)
    # training set
    train_ds = ListDataset(
        [
            {'start': "01/03/2015 00:00:00",
            'target': original_data[:, 0: training_length].tolist()
            }
        ],
        freq=freq,
        one_dim_target=False
    )

    # ------------------- create estimator -------------------
    estimator = DeepAREstimator(
        prediction_length=prediction_length,
        context_length=prediction_length,
        freq=freq,
        num_layers=num_layers,
        # num_cells=num_cells,
        dropout_rate=dropout_rate,
        trainer=Trainer(
            ctx="cpu",
            epochs=epochs,
            hybridize=False,
            batch_size=32
        ),
        distr_output=MultivariateGaussianOutput(dim=dim)
    )
    # --------------------------------------------------------
    # load model if not already trained
    # model_path = "../models/flux_model_jan-mar_0.sav"
    # model_path = "../models/FR_Pue_2002_232.sav"
    model_path = pathlib.Path(path + model_name)

    filename = pathlib.Path(model_path)
    if not filename.exists():
        print("Training model....")
        predictor = estimator.train(train_ds)
        # save the model to disk
        pickle.dump(predictor, open(filename, 'wb'))

    # ----------- Generate Knockoffs -----------------------
    data_actual = np.array(original_data[:, :]).transpose()
    n = len(original_data[:, 0])
    obj = Knockoffs()
    pars.update({'length': n, 'dim': dim, 'col': columns})
    knockoffs = obj.Generate_Knockoffs(data_actual, pars)
    # ------------------------------------------------------

    if groups_variation:

        for numgroup in range(2, 6):

            group_dict = generate_group_dicts(num_nodes, numgroup)

            pars['group_num'] = group_dict['group_num']
            pars['groups'] = group_dict['groups']
            pars['group_size'] = group_dict['groups_size']
            # Function for estimating causal impact among variables
            metrics, predicted_graph, end_time = groupCause(original_data, knockoffs, model_path, pars)
            # Calculate difference
            # elapsed_time = end_time - start_time
            # Print elapsed time
            # print("Computation time: ", round(elapsed_time/60), "mins")
            metrics_dict[numgroup] = metrics
    else:
            metrics, predicted_graph, end_time = groupCause(original_data, knockoffs, model_path, pars)
            metrics_dict[idense] = metrics

# Calculate the average for each metric
avg_metrics = calculate_average_metrics(metrics_dict)
print("Average Metrics:", avg_metrics)

# Plot and save accuracy
metics = ['Accuracy', 'Fscore', 'FPR', 'TPR']
for metric_name in metrics:
    plot_metrics(metrics_dict, metric_name)

 # Calculate difference
end_time = time.time()
elapsed_time = end_time - start_time
# Print elapsed time
print("Computation time: ", round(elapsed_time/60), "mins")