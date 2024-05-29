import io
import re
import sys
import time
import math
import pickle
import pathlib
import parameters
import numpy as np
import mxnet as mx
import pandas as pd
import seaborn as sns
import preprocessing as prep
from inference import modelTest
import matplotlib.pyplot as plt
from knockoffs import Knockoffs
from matplotlib.patches import Patch
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar._network import DeepARTrainingNetwork
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput
from scipy.stats import ttest_ind, ttest_ind_from_stats, ttest_1samp, ks_2samp, kstest, spearmanr
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

np.random.seed(1)
mx.random.seed(2)

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


def convert_variable_name(variable_name):
    # Use regular expression to find and replace digits with subscript format
    return re.sub(r'(\d+)', lambda match: f'$_{match.group(1)}$', variable_name)


def causal_criteria(list1, list2):

    n1, n2 = np.count_nonzero(list1), np.count_nonzero(list2)
    c1, c2 = n1/len(list1), n2/len(list2)
    return [c1, c2]


def groupCause(odata, knockoffs, model, params):

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

    columns = params.get('col')

     # Extract variable name and number using regular expressions
    formatted_names = [re.match(r'([A-Za-z]+)(\d+)', name).groups() for name in columns]
    # Create formatted variable names
    formatted_columns = [f'${name}_{{{number}}}$' for name, number in formatted_names]
    
    filename = pathlib.Path(model)
    if not filename.exists():
        print("Training model....")
        predictor = estimator.train(train_ds)
        # save the model to disk
        pickle.dump(predictor, open(filename, 'wb'))

    conf_mat, conf_mat_indist, conf_mat_uniform, causal_direction = [], [], [], []
    pvalues, pval_indist, pval_uniform = [], [], []
    causal_decision, indist_cause, uni_cause, group_cause = [], [], [], []

    # # create a text trap and redirect stdout
    # text_trap = io.StringIO()
    # sys.stdout = text_trap

    # Generate Knockoffs
    data_actual = np.array(odata[: , 0: training_length + prediction_length]).transpose()
    obj = Knockoffs()
    n = len(odata[:, 0])
    knockoffs = obj.Generate_Knockoffs(data_actual, params)

    # now restore stdout function
    # sys.stdout = sys.__stdout__
    
    # ------------------------------------------------------------------------------
    #         Inference for joint distribution of the multivarite system 
    # ------------------------------------------------------------------------------
    mse_realization, mape_realization = [], []
    data_range = list(range(len(odata)))
    for r in range(2):  # realizations
                        
        start_batch = 10
        mse_batches, mape_batches = [], []
        
        for iter in range(25): # batches

            test_data = odata[: , start_batch: start_batch + training_length + prediction_length].copy()
            test_ds = ListDataset(
                [
                    {'start': "01/04/2001 00:00:00",
                        'target': test_data
                    }
                ],
                freq=frequency,
                one_dim_target=False
            )

            multi_var_point_mse, muti_var_point_mape = modelTest(model, test_ds, num_samples, test_data, data_range,
                                        prediction_length, iter, False, 0)
                # # now restore stdout function
                # sys.stdout = sys.__stdout__

            mse_batches.append(multi_var_point_mse)
            mape_batches.append(multi_var_point_mse)
            start_batch = start_batch + 3
        
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
        cause_list = []
        for h in range(group_num):

            if g!=h:
                
                start_effect = groups.get(f'g{h+1}')[0]
                end_effect = groups.get(f'g{h+1}')[1]
                start_cause = groups.get(f'g{g+1}')[0]
                end_cause = groups.get(f'g{g+1}')[1]
                cause_group = f'Group: {g+1}'
                effect_group = f'Group: {h+1}' 
                
                # p-values
                pvi, pvu = [], []
                
                knockoff_samples = np.array(knockoffs[:, start_cause: end_cause]).transpose()
                interventionlist = [knockoff_samples]
                heuristic_itn_types = ['In-dist']

                mapeslol, mapeslolint = [], []
                range_effect_group = list(range(start_effect, end_effect))

                mse_interventions, imse_interventions, mape_interventions, imape_interventions = [], [], [], []
                    
                for m in range(len(interventionlist)):  # apply all types of intervention methods

                    intervene = interventionlist[m]
                    imse_realization, imape_realization = [], []
                    
                    for r in range(2):  # realizations
                        
                        start_batch = 10
                        imse_batches, imape_batches = [], []
                        
                        for iter in range(25): # batches

                            test_data = odata[: , start_batch: start_batch + training_length + prediction_length].copy()
                            test_ds = ListDataset(
                                [
                                    {'start': "01/04/2001 00:00:00",
                                        'target': test_data
                                    }
                                ],
                                freq=frequency,
                                one_dim_target=False
                            )
                        
                            int_data = odata[: , start_batch: start_batch + training_length + prediction_length].copy()
                            int_data[start_cause: end_cause, :] = intervene
                        
                            test_dsint = ListDataset(
                                [
                                    {'start': "01/04/2001 00:00:00",
                                    'target': int_data
                                    }
                                ],
                                freq=frequency,
                                one_dim_target=False
                            )

                            # Get the required inference here
                            multi_var_point_imse, muti_var_point_imape = modelTest(model, test_dsint, num_samples, test_data,
                                                        range_effect_group, prediction_length, iter, True, m)

                            if m == 0:
                                #  # create a text trap and redirect stdout
                                # text_trap = io.StringIO()
                                # sys.stdout = text_trap
                                # Generate multiple version Knockoffs
                                data_actual = np.array(odata[: , start_batch: start_batch + training_length + prediction_length]).transpose()
                                obj = Knockoffs()
                                n = len(odata[:, 0])
                                # knockoffs = obj.Generate_Knockoffs(n, params.get("dim"), data_actual)
                                knockoff_samples = np.array(knockoffs[:, start_cause: end_cause]).transpose()
                                intervene = knockoff_samples

                                # # now restore stdout function
                                # sys.stdout = sys.__stdout__
                          
                            imse_batches.append(multi_var_point_imse)
                            imape_batches.append(multi_var_point_imse)
                            # np.random.shuffle(intervene)
                            start_batch = start_batch + 3
                        
                        imse_realization.append(np.array(imse_batches))
                        imape_realization.append(np.array(imape_batches))
                    
                    # print('Forecast Multivariate_Point')
                    # print(np.array(multi_var_point_mse).shape)

                    # print('Forecast Multivariate Realization')
                    # print(np.array(mape_realization).shape)
                    # print(np.stack(mape_realization))    
                        
                    mse_mean = np.mean(np.stack(mse_realization[:, :, start_effect: end_effect]), axis=0)
                    imse_mean = np.mean(np.stack(imse_realization), axis=0)
                    mape_mean = np.mean(np.stack(mape_realization[:, :, start_effect: end_effect]), axis=0)
                    imape_mean = np.mean(np.stack(imape_realization), axis=0)

                    mse_interventions.append(mse_mean)
                    imse_interventions.append(imse_mean)
                    mape_interventions.append(mape_mean)
                    imape_interventions.append(imape_mean)

                    # print('Forecast Multivariate')
                    # print(mape_interventions)
                    # print(mape_interventions[0][:, 0])
                
                for m in range(len(interventionlist)):
                
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
                        
                        # ------------- Invariance testing (distribution and correlation) --------------
                        
                        # Calculate Spearman correlation coefficient and its p-value
                        corr, pv_corr = spearmanr(mape_interventions[m][:, j-start_effect], imape_interventions[m][:, j-start_effect])

                        print("Intervention: " + heuristic_itn_types[m])
                        t, p = ks_2samp(np.array(mape_interventions[m][:, j-start_effect]), np.array(imape_interventions[m][:, j-start_effect]))
                        pvals.append(1-p)
                        
                        print(f'Test statistic: {round(t, 2)}, p-value: {round(p, 2)}')
                        if p < 0.05:
                            print("\033[92mNull hypothesis is rejected\033[0m")
                            causal_decision.append(1)
                            print("-------------------------------------------------------")
                        else:
                            if pv_corr > 0.05:
                                print("\033[92mNull hypothesis is rejected\033[0m")
                                causal_decision.append(1)
                                print("-------------------------------------------------------")
                            else:
                                print("\033[94mFail to reject null hypothesis\033[0m")
                                causal_decision.append(0)
                                print("-------------------------------------------------------")
                        
                    

                        pvi.append(pvals[0])

                        # --------------------------- plot residuals ----------------------------------

                        fig = plt.figure()
                        ax = fig.add_subplot(111)

                        # Calculate Spearman correlation coefficient and its p-value
                        corr, p_val = spearmanr(mape_interventions[m][:, j-start_effect], imape_interventions[m][:, j-start_effect])

                        plt.plot(mape_interventions[m][:, j-start_effect], color='g', alpha=0.7, label='Actual $\epsilon$')
                        plt.plot(imape_interventions[m][:, j-start_effect], color='r', alpha=0.7, label='Counterfactual $\epsilon$')
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
                        sns.kdeplot(mape_interventions[m][:, j-start_effect], shade=True, color="g", label="Actual")
                        sns.kdeplot(imape_interventions[m][:, j-start_effect], shade=True, color="y", label="Counterfactual")
                        # sns.distplot(mapelol[0], color='red', label='Actual')
                        # sns.distplot(mapelolint[0], color='green', label='Counterfactual')
                        
                        if len(columns) > 0:
                            # effect_var = re.sub(r'(\d+)', lambda match: f'$_{match.group(1)}$', columns[j])
                            effect_var = formatted_columns[j]
                            ax1.set_ylabel(f"{cause_group} ---> {effect_var}")
                        else:
                            # plt.ylabel(f"CSS: Z_{i + 1} ---> Z_{j + 1}")
                            ax1.set_ylabel(f"{cause_group} ---> Z_{j + 1}")

                        plt.gcf()
                        ax1.legend()
                        filename = pathlib.Path(plot_path + f"{cause_group} ---> {columns[j]}.pdf")
                        plt.savefig(filename)
                        plt.show()
                        # plt.close()
                        #-------------------------------------------------------------------------------------
                        cause_list.append(causal_decision[0])
                        indist_cause.append(causal_decision[0])
        
                        causal_decision = []
                
                    if h == group_num-1 or g==group_num-1:
                        
                        for q in range(1, end_effect-start_effect):
                        
                            mape_df = pd.DataFrame(data=mape_mean, columns=columns[start_effect: end_effect])
                            mape_int_df = pd.DataFrame(data=imape_mean, columns=columns[start_effect: end_effect])

                            # Create a single plot
                            fig = plt.figure()
                            ax2 = fig.add_subplot(111)

                            # Plot the first bivariate distribution with transparency
                            sns.kdeplot(data=mape_df, x=columns[start_effect], y=columns[start_effect+q], cmap='Greens',
                                         alpha=0.75, fill=True, levels=4, color='green', label='Actual') #fill= True, cmap="Blues", alpha=0.5

                            # Plot the second bivariate distribution on top with transparency
                            sns.kdeplot(data=mape_int_df, x=columns[start_effect], y=columns[start_effect+q], cmap='Oranges',
                                         alpha=0.60, fill=True, levels=4, color='orange', label='Counterfactual') # fill=True, cmap="Reds", fill=True, cmap="Reds",

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
                            Patch(facecolor=plt.cm.Greens(100), alpha=0.70, edgecolor='g', label='Actual'),
                            Patch(facecolor=plt.cm.Oranges(100), alpha=0.85, edgecolor='orange', label='Counterfactual')
                            ]
                            ax2.legend(handles=legend_elements)
                            filename = pathlib.Path(plot_path + f"{cause_group} ---> {columns[q+start_effect]}_2d.pdf")
                            plt.savefig(filename)
                            # plt.show()

        causal_direction.append(cause_list)
    
    pval_indist.append(pvi)

    conf_mat_indist = conf_mat_indist + indist_cause
    indist_cause, uni_cause = [], []

    pvalues.append(pval_indist)
    # print("p-values: ", pvalues)

    conf_mat.append(conf_mat_indist)
    print("--------------------------------------------------------")
    print("Discovered Causal Graphs: ", conf_mat)

    print(f'Causal direction: {causal_direction}')

    for m in range(group_num):
        for n in range(group_num):
            if m > n:
                c1, c2 = causal_criteria(causal_direction[m], causal_direction[n])

                if c1 > c2:
                    print(f'gCDMI: Group {m+1} causes Group {n+1}')
                # elif c2 > c1:
                #     print(f'gCDMI: Group {n+1} causes Group {m+1}')
                elif math.ceil(c1) & math.ceil(c2) == 0:
                    print(f'gCDMI: No causal connection found in Group {m+1} and Group {n+1}')
                elif c1 == c2:
                    print('gCDMI: Causal direction can\'t be inferred')
    print("----------*****-----------------------*****------------")
    end_time = time.time()

    return conf_mat, end_time 





