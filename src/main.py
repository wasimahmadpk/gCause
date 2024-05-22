import math
import pickle
import random
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
from forecast import modelTest
from gluonts.trainer import Trainer
from groupcause import groupCause
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

# Parameters

pars = parameters.get_flux_params()
dim = pars['dim']
freq = pars["freq"]
epochs = pars["epochs"]
win_size = pars["win_size"]
slidingwin_size = pars["slidingwin_size"]
training_length = pars["train_len"]
prediction_length = pars["pred_len"]
num_samples = pars["num_samples"]
num_layers = pars["num_layers"]
num_cells = pars["num_cells"]
dropout_rate = pars["dropout_rate"]
batch_size = pars["batch_size"]
plot_path = pars["plot_path"]
groups = pars["groups"]

# Load river discharges data
# df = prep.load_river_data()
# df = prep.load_climate_data()
# df = prep.load_geo_data()

df = prep.load_flux_data()
print(df.corr())

# # Calculate the cross-correlation for lags ranging from -10 to 10 months difference
# cross_corr = [df['0'].corr(df['5'].shift(i)) for i in range(-10, 10)]
# # Print the cross-correlation values
# print(cross_corr)

# df = data.loc[:1000].copy()
# print(df.head())
print(df.describe())
print(df.shape)

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

# create estimator
estimator = DeepAREstimator(
    prediction_length=prediction_length,
    context_length=prediction_length,
    freq=freq,
    num_layers=num_layers,
    num_cells=num_cells,
    dropout_rate=dropout_rate,
    trainer=Trainer(
        ctx="cpu",
        epochs=epochs,
        hybridize=False,
        batch_size=24
    ),
    distr_output=MultivariateGaussianOutput(dim=dim)
)

# load model if not already trained
# model_path = "../models/flux_model_jan-mar_0.sav"
# model_path = "../models/FR_Pue_2002_232.sav"
model_name = pars.get('model_name')
path = pars.get('model_path')
model_path = pathlib.Path(path + model_name)

filename = pathlib.Path(model_path)
if not filename.exists():
    print("Training model....")
    predictor = estimator.train(train_ds)
    # save the model to disk
    pickle.dump(predictor, open(filename, 'wb'))

# Generate Knockoffs
data_actual = np.array(original_data[:, :]).transpose()
n = len(original_data[:, 0])
obj = Knockoffs()
pars.update({'length': n, 'dim': dim, 'col': columns})
knockoffs = obj.Generate_Knockoffs(data_actual, pars)

# Function for estimating causal impact among variables
groupCause(original_data, knockoffs, model_path, pars)
