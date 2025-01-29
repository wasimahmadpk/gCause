import io
import re
import sys
import time
import math
import pickle
import pathlib
import random
import parameters
import numpy as np
import mxnet as mx
import pandas as pd
import seaborn as sns
from scipy import stats
import preprocessing as prep
from inference import modelTest
import matplotlib.pyplot as plt
from knockoffs import Knockoffs
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar._network import DeepARTrainingNetwork
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput
from scipy.stats import ttest_ind, ttest_ind_from_stats, ttest_1samp, ks_2samp, anderson_ksamp, kstest, spearmanr
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

np.random.seed(1)
mx.random.seed(2)

data_type = 'synthetic'
params = parameters.get_syn_params()
num_samples = params["num_samples"]
step = params["step_size"]
training_length = params["train_len"]
prediction_length = params["pred_len"]
frequency = params["freq"]
plot_path = params["plot_path"]
# Get prior skeleton
prior_graph = params['prior_graph']
true_conf_mat = params["true_graph"]
group_num = params['group_num']
groups = params['groups']
num_sliding_win = params['num_sliding_win']


def kld(P, Q, epsilon=1e-10):
    """
    Calculate the Kullback-Leibler divergence between two probability distributions.

    Parameters:
    - P: np.ndarray, the first distribution
    - Q: np.ndarray, the second distribution
    - epsilon: float, small value added for numerical stability

    Returns:
    - kl_div: float, the Kullback-Leibler divergence from P to Q
    """
    # Step 1: Normalize the distributions
    P = P / np.sum(P) if np.sum(P) > 0 else P
    Q = Q / np.sum(Q) if np.sum(Q) > 0 else Q

    # Step 2: Add a small value to avoid log(0)
    P = P + epsilon
    Q = Q + epsilon

    # Step 3: Calculate KL Divergence
    kl_div = np.sum(P * np.log(P / Q))
    return round(kl_div, 3)

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

def convert_variable_name(variable_name):
    # Use regular expression to find and replace digits with subscript format
    return re.sub(r'(\d+)', lambda match: f'$_{match.group(1)}$', variable_name)


def causal_criteria(list1, list2):

    n1, n2 = np.count_nonzero(list1), np.count_nonzero(list2)
    c1, c2 = n1/len(list1), n2/len(list2)
    return [c1, c2]

def evaluate(actual, predicted):

    y_true_flat = [item for sublist in actual for item in sublist]
    y_pred_flat = [item for sublist in predicted for item in sublist]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    # Extract TP, TN, FP, FN from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate rates
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0  # True Positive Rate (Sensitivity)
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0  # True Negative Rate (Specificity)
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0  # False Negative Rate
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    # Create a dictionary to store metrics
    shd = calculate_shd(np.array(actual), np.array(predicted))
    metrics = {
        # 'TP': tp,
        # 'TN': tn,
        # 'FP': fp,
        # 'FN': fn,
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
    
    return metrics

# Diebold-Mariano test function
def dm_test(err1, err2, h=1, crit="MSE"):
    
    # Compute the loss differentials (MSE differences)
    d_mse = err1 - err2

    # Compute the mean and variance of the loss differentials
    mean_d = np.mean(d_mse)
    var_d = np.var(d_mse, ddof=1)

    # Number of forecasts
    T = len(d_mse)

    # Compute the DM test statistic
    dm_stat = mean_d / np.sqrt((var_d / T) * (1 + 2 * np.sum([1 - i / T for i in range(1, T)])))

    # Compute the p-value
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

def groupCause(df, odata, model, params, ground_truth, method='Group'):

    p_val_list = [0.05] #np.arange(0.03, 0.1, 0.03)
    num_samples = params['num_samples']
    step = params['step_size']
    training_length = params['train_len']
    prediction_length = params["pred_len"]
    frequency = params["freq"]
    plot_path = params["plot_path"]
    # Get prior skeleton
    prior_graph = params['prior_graph']
    true_conf_mat = params["true_graph"]
    group_num = params['group_num']
    columns = params.get('col')
    step = params.get('step_size')
    
    if method=='Group':
        groups = params['groups']
    elif method=='Full':
        groups = params['groups_fs']
        group_num = params['group_num_fs']
        print(f'Select number of groups: {group_num}')
    else:
        groups = params['groups_cc']

     # Extract variable name and number using regular expressions
    # formatted_names = [re.match(r'([A-Za-z]+)(\d+)', name).groups() for name in columns]
    # Create formatted variable names
    # formatted_columns = [f'${name}_{{{number}}}$' for name, number in formatted_names]

    formatted_columns = columns

    conf_mat, conf_mat_indist, conf_mat_uniform, causal_direction = [], [], [], []
    pvalues, pval_indist, pval_uniform = [], [], []
    causal_decision, causal_decision_1tier, indist_cause, uni_cause, group_cause = [], [], [], [], []

    # Generate Knockoffs
      # Generate Knockoffs
    data_actual = np.array(odata[: , 0: training_length + prediction_length]).transpose()
    n = len(data_actual[:, 0])
    params.update({'length': n})
    obj = Knockoffs()
    knockoffs = obj.Generate_Knockoffs(data_actual, params)
    
    # ------------------------------------------------------------------------------
    #         Inference for joint distribution of the multivarite system 
    # ------------------------------------------------------------------------------
    mse_realization, mape_realization = [], []
    causal_matrix, causal_matrix_1tier = [], []
    ci_matrix, kld_matrix = [] , []
    data_range = list(range(len(odata)))
    for r in range(1):  # number of time series realizations
                        
        start_batch = 10
        mse_batches, mape_batches = [], []
        
        for iter in range(num_sliding_win): # batches

            test_data= df.iloc[start_batch: start_batch + training_length + prediction_length]
            test_ds = ListDataset(
                [
                     {'start': test_data.index[0],
                     'target': test_data.values.T.tolist()
                    }
                ],
                freq=frequency,
                one_dim_target=False
            )
            multi_var_point_mse, muti_var_point_mape = modelTest(model, test_ds, num_samples, test_data, data_range,
                                        prediction_length, iter, False, 0)

            mse_batches.append(multi_var_point_mse)
            mape_batches.append(multi_var_point_mse)
            start_batch = start_batch + step
        
        mse_realization.append(np.array(mse_batches))
        mape_realization.append(np.array(mape_batches))
    
    mse_realization = np.array(mse_realization)
    mape_realization = np.array(mape_realization)
    # print('Forecast Multivariate_Point')
    # print(np.array(multi_var_point_mse).shape)

    # print('Forecast Multivariate Realization')
    # print(np.array(mape_realization).shape)
    # print(np.stack(mape_realization))
    # print(np.array(mape_realization))
    # ------------------------------------------------------------------------------------
    for g in range(group_num):
        ci_links = []
        causal_links = []
        causal_links_1tier = []
        for h in range(group_num):
            cause_list = []
            ci_list = []
            cause_list_1tier = []
         
            # if g==h:
                # causal_links.append(1)
            if g > -1: #g!=h:
                
                start_effect, end_effect = groups.get(f'g{h+1}')[0], groups.get(f'g{h+1}')[1]
                start_cause, end_cause = groups.get(f'g{g+1}')[0], groups.get(f'g{g+1}')[1] 
                cause_group, effect_group = f'Group: {g+1}', f'Group: {h+1}'
                
                # p-values
                pvi, pvu = [], []
                
                knockoff_samples = np.array(knockoffs[:, start_cause: end_cause]).transpose() 
                knockoff_samples = knockoff_samples + np.random.normal(0, 0.01, knockoff_samples.shape)
                knockoff_samples = np.random.uniform(np.min(odata), np.max(odata), knockoff_samples.shape)
                interventionlist = [knockoff_samples]
                heuristic_itn_types = ['In-dist']

                mapeslol, mapeslolint = [], []
                range_effect_group = list(range(start_effect, end_effect))
                mse_interventions, imse_interventions, mape_interventions, imape_interventions = [], [], [], []
                    
                for m in range(len(interventionlist)):  # apply all types of intervention methods

                    intervene = interventionlist[m]
                    # np.random.shuffle(intervene)
                    imse_realization, imape_realization = [], []
                    
                    for r in range(2):  # realizations
                        
                        start_batch = 10
                        imse_batches, imape_batches = [], []
                        
                        for iter in range(num_sliding_win): # batches

                            test_data = df.iloc[start_batch: start_batch + training_length + prediction_length] #.values.T.tolist()
                            test_ds = ListDataset(
                                [
                                     {'start': test_data.index[0],
                                     'target': test_data.values.T.tolist()
                                      }
                                ],
                                freq=frequency,
                                one_dim_target=False
                            )
                        
                            int_data = df.iloc[start_batch: start_batch + training_length + prediction_length].copy()
                            int_data.iloc[:, start_cause:end_cause] = intervene.T
                        
                            test_dsint = ListDataset(
                                [
                                   {'start': test_data.index[0],
                                    'target': int_data.values.T.tolist()
                                    }
                                ],
                                freq=frequency,
                                one_dim_target=False
                            )

                            # Get the required inference here
                            multi_var_point_imse, muti_var_point_imape = modelTest(model, test_dsint, num_samples, test_data,
                                                        range_effect_group, prediction_length, iter, True, m)

        
                            
                            data_actual = np.array(odata[:, start_batch: start_batch + training_length + prediction_length]).transpose()
                            obj = Knockoffs()
                            knockoffs = obj.Generate_Knockoffs(data_actual, params)
                            knockoff_samples = np.array(knockoffs[:, start_cause: end_cause]).transpose()
                            knockoff_samples = knockoff_samples + np.random.normal(0, 0.01, knockoff_samples.shape)
                            knockoff_samples = np.random.uniform(np.min(odata), np.max(odata), knockoff_samples.shape)
                            intervene = knockoff_samples
                          
                            imse_batches.append(multi_var_point_imse)
                            imape_batches.append(multi_var_point_imse)
                            # np.random.shuffle(intervene)
                            start_batch = start_batch + step
                        
                        imse_realization.append(np.array(imse_batches))
                        imape_realization.append(np.array(imape_batches))
                    
                   
                    mse_mean = np.mean(np.stack(mse_realization[:, :, start_effect: end_effect]), axis=0)
                    imse_mean = np.mean(np.stack(imse_realization), axis=0)
                    mape_mean = np.mean(np.stack(mape_realization[:, :, start_effect: end_effect]), axis=0)
                    imape_mean = np.mean(np.stack(imape_realization), axis=0)
                
                intervention_type = 'In-distribution'
                for j in range(start_effect, end_effect):

                    print("----------*****-----------------------*****------------")
                    # print(f"MSE(Mean): {list(np.mean(mselol, axis=0))}")
                    if len(columns) > 0:
                        print(f"Causal Link: {cause_group} --------------> {effect_group} ({columns[j]})")
                        print("----------*****-----------------------*****------------")
                        fnamehist = plot_path + "{columns[i]}_{columns[j]}:hist"
                    else:
                        print(f"Causal Link: {cause_group} --------------> Z_{j + 1}")
                        print("----------*****-----------------------*****------------")
                        fnamehist = plot_path + "{Z_[i + 1]}_{Z_[j + 1]}:hist"
                    
                    pvals = []
                    
                    # ----------------------------------------------------------- 
                    #      Invariance testing (distribution and correlation) 
                    # -----------------------------------------------------------


                    # -----------------------------------------------------------
                    # Perform the DM test
                    # t, pv_corr = dm_test(np.array(mape_interventions[m][:, j-start_effect]), np.array(imape_interventions[m][:, j-start_effect]))
                    # print(f"DM Statistic: {t}, p-value: {p}")
                    # -----------------------------------------------------------

                    # -----------------------------------------------------------
                    #         Calculate CausalImpatc in terms of KLD
                    # -----------------------------------------------------------
                    kld_val = kld(mape_mean[:, j-start_effect], imape_mean[:, j-start_effect])
                    
                    # Calculate Spearman correlation coefficient and its p-value
                    corr, pv_corr = spearmanr(mape_mean[:, j-start_effect], imape_mean[:, j-start_effect])
                    print("Intervention: " + intervention_type)
                    # t, p = ttest_ind(np.array(mape_interventions[m][:, j-start_effect]), np.array(imape_interventions[m][:, j-start_effect]))
                    t, p = ks_2samp(np.array(mape_mean[:, j-start_effect]), np.array(imape_mean[:, j-start_effect]))
                    # ad_test = anderson_ksamp([np.array(mape_interventions[m][:, j-start_effect]), np.array(imape_interventions[m][:, j-start_effect])])  # Anderson-Darling Test
                    # p = ad_test[2]
                    pvals.append(p)
                    
                    print(f'Test statistic: {round(t, 2)}, pv-dist: {round(p, 2)}, pv-corr: {round(pv_corr, 2)}, kld: {kld_val}')
                    if p < 0.05:
                        print("\033[92mNull hypothesis is rejected\033[0m")
                        causal_decision.append(1)
                        causal_decision_1tier.append(1)
                        print("-------------------------------------------------------")
                    else:
                        causal_decision_1tier.append(0)
                        if pv_corr < 0.50:
                            print("\033[94mFail to reject null hypothesis\033[0m")
                            causal_decision.append(0)
                            print("-------------------------------------------------------")
                        else:
                            print("\033[92mNull hypothesis is rejected\033[0m")
                            causal_decision.append(1)
                            print("-------------------------------------------------------")
                    
                    pvi.append(pvals[0])

                    # --------------------------- plot residuals ----------------------------------
                    fig = plt.figure()
                    ax = fig.add_subplot(111)

                    # Calculate Spearman correlation coefficient and its p-value
                    corr, p_val = spearmanr(mape_mean[:, j-start_effect], imape_mean[:, j-start_effect])

                    plt.plot(mape_mean[:, j-start_effect], color='g', alpha=0.7, label='Actual $\epsilon$')
                    plt.plot(imape_mean[:, j-start_effect], color='r', alpha=0.7, label='Counterfactual $\epsilon$')
                    plt.title(f'corr: {round(corr, 2)}, p-val: {round(p_val, 2)}')
                    if len(columns) > 0:
                        # effect_var = re.sub(r'(\d+)', lambda match: f'$_{match.group(1)}$', columns[j])
                        effect_var = formatted_columns[j]
                        ax.set_ylabel(f'{cause_group} ---> {effect_var}')
                    else:
                        # plt.ylabel(f"CSS: Z_{i + 1} ---> Z_{j + 1}")
                        ax.set_ylabel(f'{cause_group} ---> Z_{j + 1}')
                    
                    plt.gcf()
                    ax.legend()
                    filename = pathlib.Path(plot_path + f'res_{cause_group} ---> {columns[j]}.pdf')
                    plt.savefig(filename)
                    plt.show()
                
                    # -------------------------- plot residuals distribution ---------------------------
                    fig = plt.figure()
                    ax1 = fig.add_subplot(111)
                    
                    # Create the KDE plot
                    sns.kdeplot(mape_mean[:, j-start_effect], shade=True, color="g", label="Actual")
                    # Calculate the mean of the data
                    mean_value = np.mean(mape_mean[:, j-start_effect])
                    # Add a vertical line at the mean
                    plt.axvline(mean_value, color="g", linestyle="--", label=f"Mean: {mean_value:.2f}")
                    sns.kdeplot(imape_mean[:, j-start_effect], shade=True, color="y", label="Counterfactual")
                    # Calculate the mean of the data
                    mean_value = np.mean(imape_mean[:, j-start_effect])
                    # Add a vertical line at the mean
                    plt.axvline(mean_value, color="y", linestyle="--", label=f"Mean: {mean_value:.2f}")
                    
                    # sns.distplot(mapelol[0], color='red', label='Actual')
                    # sns.distplot(mapelolint[0], color='green', label='Counterfactual')
                    
                    if len(columns) > 0:
                        # effect_var = re.sub(r'(\d+)', lambda match: f'$_{match.group(1)}$', columns[j])
                        effect_var = formatted_columns[j]
                        ax1.set_ylabel(f"{cause_group} ---> {effect_var}")
                    else:
                        # plt.ylabel(f"CSS: Z_{i + 1} ---> Z_{j + 1}")
                        ax1.set_ylabel(f"{cause_group} ---> Z_{j + 1}")

                    rnd = random.randint(1, 9999)
                    plt.gcf()
                    ax1.legend()
                    filename = pathlib.Path(plot_path + f"{cause_group} ---> {columns[j]}_{method}_{rnd}.pdf")
                    plt.savefig(filename)
                    plt.show()
                    # plt.close()
                    #-------------------------------------------------------------------------------------
                    cause_list.append(causal_decision[0])
                    cause_list_1tier.append(causal_decision_1tier[0])
                    indist_cause.append(causal_decision[0])

                    ci_list.append(p)
                    causal_decision = []
            
                if h == group_num-1 or g==group_num-1:
                    
                    for q in range(1, end_effect-start_effect):
                    
                        mape_df = pd.DataFrame(data=mse_mean, columns=columns[start_effect: end_effect])
                        mape_int_df = pd.DataFrame(data=imse_mean, columns=columns[start_effect: end_effect])

                        # Create a single plot
                        fig = plt.figure()
                        ax2 = fig.add_subplot(111)

                        # Plot the first bivariate distribution with transparency
                        sns.kdeplot(data=mape_df, x=columns[start_effect], y=columns[start_effect+q], cmap='Blues',
                                        alpha=0.75, fill=True, levels=4, color='blue', label='Actual') #fill= True, cmap="Blues", alpha=0.5

                        # Plot the second bivariate distribution on top with transparency
                        sns.kdeplot(data=mape_int_df, x=columns[start_effect], y=columns[start_effect+q], cmap='Reds',
                                        alpha=0.60, fill=True, levels=4, color='red', label='Counterfactual') # fill=True, cmap="Reds", fill=True, cmap="Reds",

                        if len(columns) > 0:
                            # plt.ylabel(f"CSS: {columns[i]} ---> {columns[j]}")
                            # effect_var = re.sub(r'(\d+)', lambda match: f'$_{match.group(1)}$', columns[q])
                            effect_var = formatted_columns[q]
                            ax1.set_ylabel(f"{cause_group} ---> {effect_var}")
                        else:
                            # plt.ylabel(f"CSS: Z_{i + 1} ---> Z_{j + 1}")
                            ax1.set_ylabel(f"{cause_group} ---> Z_{q + 1}")

                        # Show the plot
                        plt.gcf()

                        # Add a custom legend
                        legend_elements = [
                        Patch(facecolor=plt.cm.Greens(100), alpha=0.70, edgecolor='b', label='Actual'),
                        Patch(facecolor=plt.cm.Oranges(100), alpha=0.85, edgecolor='r', label='Counterfactual')
                        ]
                        ax2.legend(handles=legend_elements)
                        filename = pathlib.Path(plot_path + f"{cause_group} ---> {columns[q+start_effect]}_2d_{method}_{rnd}.pdf")
                        plt.savefig(filename)
                        # plt.show()
                
                ci_links.append(ci_list[0])
                if method=='Full':
                    causal_links.append(cause_list[0])
                    causal_links_1tier.append(cause_list_1tier[0])
                else: 
                    causal_links.append(1 if 1 in cause_list else 0)
                    causal_links_1tier.append(1 if 1 in cause_list_1tier else 0)

        ci_matrix.append(ci_links)
        causal_matrix.append(causal_links)
        causal_matrix_1tier.append(causal_links_1tier)
    
    pval_indist.append(pvi)

    conf_mat_indist = conf_mat_indist + indist_cause
    indist_cause = []

    pvalues.append(pval_indist)
    # print("p-values: ", pvalues)
    # ground_truth = get_ground_truth(group_num)
    conf_mat.append(conf_mat_indist)
    print("--------------------------------------------------------")
    # print("Pair-wise Graph: ", conf_mat)
    print(f'Actual Causal Graph: \n {ground_truth}')
    print(f'Discovered Causal Graph: \n {np.array(causal_matrix)}')
    print(f'Causal Impact Graph: \n {np.array(ci_matrix)}')
    print("----------*****-----------------------*****-------------")

  
    print("----------*****-----------------------*****-------------")
    pred = np.array(kld_matrix)   #1 - np.array(kld_matrix)
    actual_lab = prep.remove_diagonal_and_flatten(ground_truth)
    pred_score = prep.remove_diagonal_and_flatten(pred)
    fmax = prep.f1_max(actual_lab, pred_score)
     
    # Calculate metrics
    metrics = prep.evaluate_best_predicted_graph(ground_truth, np.array([causal_matrix]))
    metrics_1tier = prep.evaluate_best_predicted_graph(ground_truth, np.array([causal_matrix_1tier]))
    metrics['Fscore'] = fmax[1]
    # Print metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

    # Print metrics
    print("----------*****--------Tier1-INtest--------*****------------")
    for metric, value in metrics_1tier.items():
       print(f"{metric}: {value:.2f}")

    return metrics, conf_mat, metrics_1tier






