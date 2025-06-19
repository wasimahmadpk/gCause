import pathlib
import random
import numpy as np
import pandas as pd
import seaborn as sns
from preprocessing import *
from inference import modelTest
import matplotlib.pyplot as plt
from knockoffs import Knockoffs
from matplotlib.patches import Patch
from gluonts.dataset.common import ListDataset
from scipy.stats import ttest_ind, ttest_ind_from_stats, ttest_1samp, ks_2samp, ranksums, anderson_ksamp, ttest_rel, kstest, spearmanr
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


def groupCause(df, odata, model, params, ground_truth, method='Group'):

    p_val_list = [0.01] # np.arange(0.03, 0.1, 0.03)
    num_samples = params['num_samples']
    step = params['step_size']
    training_length = params['train_len']
    prediction_length = params["pred_len"]
    frequency = params["freq"]
    plot_path = params["plot_path"] # Get prior skeleton
    prior_graph = params['prior_graph']
    true_conf_mat = params["true_graph"]
    group_num = params['group_num']
    columns = params['col']
    step = params['step_size']
    num_sliding_win =  params['num_sliding_win']
    alpha = params['alpha']
    
    if method=='Group':
        groups = params['groups']
    elif method=='Full':
        groups = params['groups_fs']
        group_num = params['group_num_fs']
        print(f'Select number of groups: {group_num}')
    else:
        groups = params['groups_cc']

    formatted_columns = columns
    conf_mat, conf_mat_indist = [], []
    pvalues, pval_indist = [], []
    causal_decision, causal_decision_1tier, indist_cause = [], [], []

    # Generate Knockoffs
    data_actual = np.array(odata[: , 0: training_length + prediction_length]).transpose()
    n = len(data_actual[:, 0])
    params.update({'length': n})
    knock_obj = Knockoffs()
    knockoffs = knock_obj.Generate_Knockoffs(data_actual, params)
    
    # ------------------------------------------------------------------------------
    #         Inference for joint distribution of the multivarite system 
    # ------------------------------------------------------------------------------
    mse_realization, mape_realization = [], []
    causal_matrix, causal_matrix_1tier = [], []
    ci_matrix, ci_matrix_t1, kld_matrix = [] , [], []
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

    # ------------------------------------------------------------------------------------
    for g in range(group_num):
        ci_links, ci_links_t1 = [], []
        causal_links = []
        causal_links_1tier = []
        for h in range(group_num):
            
            ci_list, ci_list_t1 = [], []
            cause_list, cause_list_1tier = [], []
         
            # if g==h:
                # causal_links.append(1)
            if g > -1: #g!=h:
                
                start_effect, end_effect = groups.get(f'g{h+1}')[0], groups.get(f'g{h+1}')[1]
                start_cause, end_cause = groups.get(f'g{g+1}')[0], groups.get(f'g{g+1}')[1] 
                cause_group, effect_group = f'Group: {g+1}', f'Group: {h+1}'
                
                knockoff_samples = np.array(knockoffs[:, start_cause: end_cause]).transpose() 
                knockoff_samples = knockoff_samples + np.random.normal(10.1, 10.1, knockoff_samples.shape)
                # knockoff_samples = np.random.uniform(np.min(odata), np.max(odata), knockoff_samples.shape)

                pvi, mapeslol, mapeslolint = [], [], [] # p-values
                range_effect_group = list(range(start_effect, end_effect))

                intervene = knockoff_samples
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
                                                    range_effect_group, prediction_length, iter, True, 1)

                        data_actual = np.array(odata[:, start_batch: start_batch + training_length + prediction_length]).transpose()
                        knockoffs = knock_obj.Generate_Knockoffs(data_actual, params)
                        knockoff_samples = np.array(knockoffs[:, start_cause: end_cause]).transpose()
                        knockoff_samples = knockoff_samples + np.random.normal(10.1, 10.1, knockoff_samples.shape)
                        # knockoff_samples = np.random.uniform(np.min(odata), np.max(odata), knockoff_samples.shape)
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
        
                    # Perform the DM test
                    # t, pv_corr = dm_test(np.array(mape_interventions[m][:, j-start_effect]), np.array(imape_interventions[m][:, j-start_effect]))
                    # print(f"DM Statistic: {t}, p-value: {p}")

                    # -----------------------------------------------------------
                    #         Calculate CausalImpatc in terms of KLD
                    # -----------------------------------------------------------
                    kld_val = kld(mape_mean[:, j-start_effect], imape_mean[:, j-start_effect])
                    
                    # Calculate Spearman correlation coefficient and its p-value
                    corr_val, pv_corr = spearmanr(mape_mean[:, j-start_effect], imape_mean[:, j-start_effect])
                    print("Intervention: " + intervention_type)
                    # t, p = ks_2samp(np.array(mape_mean[:, j-start_effect]), np.array(imape_mean[:, j-start_effect]))
                    # t, p = ranksums(np.array(mape_mean[:, j-start_effect]), np.array(imape_mean[:, j-start_effect]))
                    t, p = ttest_rel(mape_mean[:, j-start_effect], imape_mean[:, j-start_effect])
                    # t, p = ttest_ind(mape_mean[:, j-start_effect], imape_mean[:, j-start_effect], equal_var=True, alternative = 'greater')
                    # t, p = ttest_1samp(imape_mean[:, j-start_effect], popmean=np.mean(mape_mean[:, j-start_effect]))
                    # t, p = ttest_1samp(imape_mean[:, j-start_effect], popmean=0)
                    # ad_test = anderson_ksamp(np.array(mape_mean[:, j-start_effect]), np.array(imape_mean[:, j-start_effect]))  # Anderson-Darling Test
                    # p = ad_test[2]
                    pvals.append(p)
                    
                    print(f'Test statistic: {round(t, 2)}, pv-dist: {round(p, 2)}, pv-corr: {round(pv_corr, 2)}, kld: {kld_val}')
                    if p < alpha:
                        print("\033[92mNull hypothesis is rejected\033[0m")
                        causal_decision.append(1)
                        causal_decision_1tier.append(1)
                        print("-------------------------------------------------------")
                    else:
                        causal_decision_1tier.append(0)
                        # if corr_val > 0.90:
                        print("\033[94mFail to reject null hypothesis\033[0m")
                        causal_decision.append(0)
                        print("-------------------------------------------------------")
                        # else:
                            # print("\033[92mNull hypothesis is rejected\033[0m")
                            # causal_decision.append(1)
                            # print("-------------------------------------------------------")
                    
                    pvi.append(pvals[0])
                    # -------------------------------------------------------- 
                    #                        plot residuals 
                    # --------------------------------------------------------
                    fig = plt.figure()
                    ax = fig.add_subplot(111)

                    plt.plot(mape_mean[:, j-start_effect], color='g', alpha=0.7, label='Actual $\epsilon$')
                    plt.plot(imape_mean[:, j-start_effect], color='r', alpha=0.7, label='Counterfactual $\epsilon$')
                    plt.title(f'corr: {round(corr_val, 2)}, p-val: {round(pv_corr, 2)}')
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
                
                    # ----------------------------------------------------------------
                    #                   plot residuals distribution 
                    # ----------------------------------------------------------------
                    fig = plt.figure()
                    ax1 = fig.add_subplot(111)
                    
                    # Create the KDE plot
                    sns.kdeplot(mape_mean[:, j-start_effect], shade=True, color="g", label="Actual")
                    mean_value = np.mean(mape_mean[:, j-start_effect]) # Calculate the mean of the data
                     # Add a vertical line at the mean
                    plt.axvline(mean_value, color="g", linestyle="--", label=f"Mean: {mean_value:.2f}")
                    sns.kdeplot(imape_mean[:, j-start_effect], shade=True, color="y", label="Counterfactual")
                    mean_value = np.mean(imape_mean[:, j-start_effect])
                    plt.axvline(mean_value, color="y", linestyle="--", label=f"Mean: {mean_value:.2f}")
                    
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
                    colname = make_filename_safe(columns[j])
                    filename = pathlib.Path(plot_path + f'{cause_group} ---> {colname}.pdf')
                    plt.savefig(filename)
                    print(f'plot saved to: {filename}')
                    plt.show()
                    # plt.close()
                    #-------------------------------------------------------------------------------------
                    cause_list.append(causal_decision[0])
                    cause_list_1tier.append(causal_decision_1tier[0])
                    indist_cause.append(causal_decision[0])

                    ci_list.append(kld_val)
                    causal_decision, causal_decision_1tier = [], []
                    
            
                if h == group_num-1 or g==group_num-1:
                    
                    for q in range(1, end_effect-start_effect):
                    
                        mape_df = pd.DataFrame(data=mse_mean, columns=columns[start_effect: end_effect])
                        mape_int_df = pd.DataFrame(data=imse_mean, columns=columns[start_effect: end_effect])

                        # Create a single plot
                        fig = plt.figure(figsize=(8, 6))  # Adjust size as needed
                        ax2 = fig.add_subplot(111)

                        # Plot the first bivariate distribution with transparency
                        sns.kdeplot(data=mape_df, x=columns[start_effect], y=columns[start_effect+q], cmap='Greens',
                                        alpha=0.99, fill=True, levels=6, label='Actual') #fill= True, cmap="Blues", alpha=0.5

                        # Plot the second bivariate distribution on top with transparency
                        sns.kdeplot(data=mape_int_df, x=columns[start_effect], y=columns[start_effect+q], cmap='Oranges',
                                        alpha=0.75, fill=True, levels=6, label='Counterfactual') # fill=True, cmap="Reds", fill=True, cmap="Reds",

                        if len(columns) > 0:
                            # plt.ylabel(f"CSS: {columns[i]} ---> {columns[j]}")
                            # effect_var = re.sub(r'(\d+)', lambda match: f'$_{match.group(1)}$', columns[q])
                            effect_var = formatted_columns[q]
                            # ax2.set_ylabel(f"{cause_group} ---> {effect_var}")
                            print(f'{columns[start_effect]} and {columns[start_effect+q]}')
                            plt.xlabel(columns[start_effect], fontsize=22) #
                            plt.ylabel(columns[start_effect+q], fontsize=22) #columns[start_effect+q]
                            plt.xticks(fontsize=20)
                            plt.yticks(fontsize=20)
                            plt.xticks(rotation=0)  # Rotate if values overlap
                            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))  # Format to 2 decimal places

                            plt.yticks(fontsize=18)
                            plt.tight_layout()
                        else:
                            # plt.ylabel(f"CSS: Z_{i + 1} ---> Z_{j + 1}")
                            # ax2.set_ylabel(f"{cause_group} ---> Z_{q + 1}")
                            plt.xticks(fontsize=15)
                            plt.yticks(fontsize=15)

                        # Show the plot
                        plt.gcf()
                         # Custom Legend
                        legend_elements = [
                            Patch(facecolor=plt.cm.Greens(100), alpha=0.9, label='Actual'),
                            Patch(facecolor=plt.cm.Oranges(100), alpha=0.7, label='Counterfactual')
                        ]

                        plt.legend(handles=legend_elements, fontsize=22)
                        filename = pathlib.Path(plot_path + f'{cause_group} ---> {columns[q+start_effect]}_2d_{method}_{rnd}.pdf')
                        plt.savefig(filename)
                        # plt.show()
                            
                ci_links.append(ci_list[0])
                if method=='Full':
                    causal_links.append(cause_list[0])
                else: 
                    causal_links.append(1 if 1 in cause_list else 0)

        ci_matrix.append(ci_links)
        causal_matrix.append(causal_links)
    
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
    print(f'Causal Impact Graph: {np.array(ci_matrix).shape} \n {np.array(ci_matrix)}')
    print("----------*****-----------------------*****-------------")

    print("----------*****-----------------------*****-------------")
    pred = np.array(ci_matrix)   #1 - np.array(kld_matrix)
    actual_lab = remove_diagonal_and_flatten(ground_truth)
    pred_score = remove_diagonal_and_flatten(pred)
    fmax = f1_max(actual_lab, pred_score)
     
    # Calculate metrics
    metrics = evaluate_best_predicted_graph(ground_truth, np.array([causal_matrix]))
    metrics_1tier = evaluate_best_predicted_graph(ground_truth, np.array([causal_matrix_1tier]))
    # metrics['Fscore'] = fmax[1]
    # Print metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

    return metrics, conf_mat






