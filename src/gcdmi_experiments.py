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
mx.random.seed(2)

start_time = time.time()
# ----------------------------------------------------- 
#                     Parameters 
# -----------------------------------------------------
pars = parameters.get_sims_params()
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

# ------------------------------------------------------------
metrics_dict_cdmi, metrics_dict_gcdmi, metrics_dict_ccdmi, metrics_dict_pcmci, metrics_dict_vgc = {}, {}, {}, {}, {}
groups_variation = False
method = {'Full': 'Full', 'Group': 'Group', 'Canonical': 'Canonical'}

# regime_size = 4500 # hourly sampled data
# start = 0
for numgroups in np.arange(2, 12, 2):
    
    print("----------*****-----------------------*****------------")
    print(f'Experiment: {numgroups} ')
    model_name = 'netsim_'+f'{numgroups}'+'.sav'
    pars['model_name'] = model_name


    np.random.seed(1)
    nodes = 2*numgroups
    num_groups = numgroups

    # df, reduced_graph, full_graph = prep.load_geo_data(start, start+regime_size)
    # start = start + regime_size

    df, full_graph = load_sims_data(numgroups)
    print(f'Shape: {df.shape}')
    print(df.head())

    group_dict = generate_group_dicts(num_groups*5, numgroups)
    print(f'Group dict: {group_dict}')
    pars['group_num'] = group_dict['group_num']
    pars['group_num_fs'] = group_dict['group_num_fs']
    pars['groups'] = group_dict['groups']
    pars['groups_cc'] = group_dict['groups_cc']
    pars['groups_fs'] = group_dict['groups_fs']
    pars['groups_size'] = group_dict['groups_size']
    pars['groups_size_cc'] = group_dict['groups_size_cc']
    pars['groups_size_fs'] = group_dict['groups_size_fs']

    group_size_list = [value[0] for value in pars['groups_size'].values()]
    ground_truth = get_ground_truth(full_graph, group_size_list)
    print(f'Ground Truth: {ground_truth}')
    
        # --------------- Test CDMI on full-set ------------------
    metrics_gcdmi, predicted_graph_gcdmi, end_time = gcdmi.causal_graph(df, pars, ground_truth, method['Group'])
    metrics_ccdmi, predicted_graph_ccdmi, end_time = gcdmi.causal_graph(calculate_multi_group_cc(df, group_size_list), pars, ground_truth, method['Canonical'])
    metrics_pcmci, predicted_graph_pcmci = pcmci.causal_graph(calculate_multi_group_cc(df, group_size_list), ground_truth) #pcmci.causal_graph(df, ground_truth) #
    metrics_vgc, predicted_graph_vgc = vgc.causal_graph(calculate_multi_group_cc(df, group_size_list), ground_truth) #vgc.causal_graph(df, ground_truth) #

    metrics_dict_gcdmi[numgroups] = metrics_gcdmi
    metrics_dict_ccdmi[numgroups] = metrics_ccdmi
    metrics_dict_pcmci[numgroups] = metrics_pcmci
    metrics_dict_vgc[numgroups] = metrics_vgc

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