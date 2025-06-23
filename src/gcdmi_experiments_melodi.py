import time
import parameters
import baseline.vgc as vgc
import baseline.pcmci as pcmci
import gcdmi
from data_loader import *
import numpy as np
import mxnet as mx
import pandas as pd
from preprocessing import *
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA

np.random.seed(1)
start_time = time.time()
# ----------------------------------------------------- 
#                     Parameters 
# -----------------------------------------------------
pars = parameters.get_melodi_params()
dim = pars['dim']
freq = pars["freq"]
epochs = pars["epochs"]
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
num_of_groups = pars.get('group_num') 
print(num_of_groups)

# ------------------------------------------------------------
metrics_dict_gcdmi, metrics_dict_ccdmi, metrics_dict_pcmci, metrics_dict_vgc = {}, {}, {}, {}
groups_variation = False
method = {'Full': 'Full', 'Group': 'Group', 'Canonical': 'Canonical'}

# regime_size = 4500 # hourly sampled data
# start = 0
start, end, step = 4, 33, 3
for combi in np.arange(start, end, step):
    
    print("----------*****-----------------------*****------------")
    print(f'Experiment: {combi} ')
    model_name = 'melodi_'+f'{combi}'+'.sav'
    pars['model_name'] = model_name

    df, full_graph = load_melodi_data(start, step)
    print(f'Shape: {df.shape}')
    print(df.head())

    group_size_list = [value[0] for value in pars['groups_size'].values()]
    ground_truth = full_graph
    print(f'Ground Truth: {ground_truth}')
    
    metrics_gcdmi, predicted_graph_gcdmi, end_time = gcdmi.causal_graph(df, pars, ground_truth, method['Group'])
    metrics_ccdmi, predicted_graph_ccdmi, end_time = gcdmi.causal_graph(calculate_multi_group_cc(df, group_size_list), pars, ground_truth, method['Canonical'])
    metrics_pcmci, predicted_graph_pcmci = pcmci.causal_graph(calculate_multi_group_cc(df, group_size_list), ground_truth) #pcmci.causal_graph(df, ground_truth) #
    metrics_vgc, predicted_graph_vgc = vgc.causal_graph(calculate_multi_group_cc(df, group_size_list), ground_truth) #vgc.causal_graph(df, ground_truth) #

    metrics_dict_gcdmi[combi] = metrics_gcdmi
    metrics_dict_ccdmi[combi] = metrics_ccdmi
    metrics_dict_pcmci[combi] = metrics_pcmci
    metrics_dict_vgc[combi] = metrics_vgc
   

methods_metrics_dict = {'gCDMI': metrics_dict_gcdmi, 'MC-CDMI': metrics_dict_ccdmi, 'MC-PCMCI': metrics_dict_pcmci, 'MC-VGC': metrics_dict_vgc}
# Plot and save accuracy
metrics = ['Accuracy', 'SHD', 'Precision', 'Recall', 'Fscore', 'TPR', 'FPR', 'FNR']
for metric_name in metrics:
    plot_real_metrics(methods_metrics_dict, plot_path, metric_name)

plot_boxplots(methods_metrics_dict, plot_path)
 # Calculate difference
end_time = time.time()
elapsed_time = end_time - start_time
# Print elapsed time
print("Computation time: ", round(elapsed_time/60), "mins")