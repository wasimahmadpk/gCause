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
from deepcause import deepCause
import matplotlib.pyplot as plt
from knockoffs import Knockoffs
from scipy.special import stdtr
from forecast import modelTest
from gluonts.trainer import Trainer
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
# pars = parameters.get_geo_params()
pars = parameters.get_syn_params()
freq = pars.get("freq")
epochs = pars.get("epochs")
win_size = pars.get("win_size")
slidingwin_size = pars.get("slidingwin_size")
training_length = pars.get("train_len")
prediction_length = pars.get("pred_len")
num_samples = pars.get("num_samples")
num_layers = pars.get("num_layers")
num_cells = pars.get("num_cells")
dropout_rate = pars.get("dropout_rate")
batch_size = pars.get("batch_size")
plot_path = pars.get("plot_path")
groups = pars.get("groups")

# Load river discharges data
# df = prep.load_river_data()
# df = prep.load_climate_data()
# df = prep.load_geo_data()

df = prep.load_syn_data()
df = df.iloc[:, :]

# df = data.loc[:1000].copy()
print(df.describe())
print(df.shape)
print(df.head(5))

# df.plot.scatter(x='BO', y='Awake', c='blue')
# plt.xlabel("PPFD ($\mu$ mol photons $m^{2}s^{-1}$)")
# plt.ylabel("NEP ($\mu$ mol $CO_2$ $m^{2}s^{-1}$)")
# filename = pathlib.Path(plot_path + "PPFD->NEP_Scatter.pdf")
# plt.savefig(filename)
# plt.show()

original_data = []
dim = len(df.columns)
columns = df.columns
print(f"Dimension {dim} and Columns: {df.columns}")

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
        batch_size=32
    ),
    distr_output=MultivariateGaussianOutput(dim=dim)
)

# load model if not already trained
model_path = "../models/trained_model_syn0.sav"

filename = pathlib.Path(model_path)
if not filename.exists():
    print("Training forecasting model....")
    predictor = estimator.train(train_ds)
    # save the model to disk
    pickle.dump(predictor, open(filename, 'wb'))

# Generate Knockoffs
group_start = groups['g1'][0]
group_end = groups['g1'][1]

data_actual = np.array(original_data[:,: ]).transpose()
n = len(original_data[:, 0])
obj = Knockoffs()
knockoffs = obj.Generate_Knockoffs(n, dim, data_actual)

params = {"dim": dim, "col": columns}
# Function for estimating causal impact among variables
deepCause(original_data, knockoffs, groups, model_path, params)