import time
import pickle
import pathlib
import parameters
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
# ---------------------------------------------------------
metrics_dict = {}

for nonlin in np.arange(0.2, 1 + 0.2, 0.2):
    model_name = pars.get('model_name') + 'nonlinearity_' + '{nonlin}'
    # Run CDMI n times
    # ------------ SCMs----------------------------------------
    # Load synthetic data
    model = StructuralCausalModel()
    num_nodes = 8
    nonlinearity = nonlin
    interaction_density = 0.25
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

    # Function for estimating causal impact among variables
    metrics, predicted_graph, end_time = groupCause(original_data, knockoffs, model_path, pars)

    # Calculate difference
    # elapsed_time = end_time - start_time
    # Print elapsed time
    # print("Computation time: ", round(elapsed_time/60), "mins")
    metrics_dict[nonlin] = metrics

# Calculate the average for each metric
avg_metrics = {}


# Set to store inner keys
performance_metrics_keys = set()

# Iterate over the outer dictionary
for params_keys, inner_dict in metrics_dict.items():
    # Update the set with keys from the inner dictionary
    performance_metrics_keys.update(inner_dict.keys())

# Convert set to list if needed
performance_metrics_keys = list(performance_metrics_keys)

keys = metrics_dict.keys()
for key in performance_metrics_keys:
    avg_metrics[key] = np.mean([metrics[key] for metrics in metrics_dict.values])

print(avg_metrics)


# Extract and plot metrics
def plot_metrics(metrics_dict, metric_name):
    x = list(metrics_dict.keys())
    y = [metrics[metric_name] for metrics in metrics_dict.values()]

    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel('Nonlinearity')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} vs Nonlinearity')
    plt.grid(True)
    filename = pathlib.Path(plot_path + f'{metric_name}_nonlinearity.pdf')
    plt.savefig(filename)  # Save the figure
    plt.show()

# Plot and save accuracy
metics = ['Accuracy', 'Fscore', 'FPR', 'TPR']
for metric_name in metrics:
    plot_metrics(metrics_dict, metric_name)

 # Calculate difference
end_time = time.time()
elapsed_time = end_time - start_time
# Print elapsed time
print("Computation time: ", round(elapsed_time/60), "mins")