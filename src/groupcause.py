import io
import sys
import time
import pickle
import pathlib
import parameters
import numpy as np
import mxnet as mx
import pandas as pd
import seaborn as sns
import preprocessing as prep
from forecast import modelTest
import matplotlib.pyplot as plt
from knockoffs import Knockoffs
from matplotlib.patches import Patch
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar._network import DeepARTrainingNetwork
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput
from scipy.stats import ttest_ind, ttest_ind_from_stats, ttest_1samp, ks_2samp, kstest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

np.random.seed(1)
mx.random.seed(2)

pars = parameters.get_flux_params()
num_samples = pars["num_samples"]
step = pars["step_size"]
training_length = pars["train_len"]
prediction_length = pars["pred_len"]
frequency = pars["freq"]
plot_path = pars["plot_path"]
# Get prior skeleton
prior_graph = pars['prior_graph']
true_conf_mat = pars["true_graph"]

group_num = pars['group_num']
groups = pars['groups']


def causal_criteria(list1, list2):

    n1, n2 = np.count_nonzero(list1), np.count_nonzero(list2)
    c1, c2 = n1/len(list1), n2/len(list2)
    return [c1, c2]

def groupCause(odata, knockoffs, model, params):
    
    print('Inside gCause')
    filename = pathlib.Path(model)
    if not filename.exists():
        print("Training model....")
        predictor = estimator.train(train_ds)
        # save the model to disk
        pickle.dump(predictor, open(filename, 'wb'))

    conf_mat, conf_mat_indist, conf_mat_uniform, causal_direction = [], [], [], []
    pvalues, pval_indist, pval_uniform = [], [], []
    var_list, causal_decision, indist_cause, uni_cause, group_cause = [], [], [], [], []

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
                
                # P-Values
                pvi, pvu = [], []
                
                knockoff_samples = np.array(knockoffs[:, start_cause: end_cause]).transpose()
                uniform_intervention = np.random.uniform(np.min(odata), np.min(odata), knockoff_samples.shape)
                
                interventionlist = [knockoff_samples, uniform_intervention]
                heuristic_itn_types = ['In-dist', 'Uniform']

                mapeslol, mapeslolint = [], []
                for j in range(start_effect, end_effect):
                
                    print("----------*****-----------------------*****------------")

                    columns = params.get('col')
                    pred_var = odata[j]
                    pred_var_name = "Z_" + str(j + 1) + ""

                    css_list = []
                    css_list_new = []
                    css_score_new = []
                    mselol = []
                    mapelol = []
                    mselolint = []
                    mapelolint = []

                    for m in range(len(interventionlist)):  # apply all types of intervention methods

                        intervene = interventionlist[m]

                        mselist = []      # list of MSE values for multiple realization without intervention
                        mselistint = []   # list of MSE values for multiple realization with intervention
                        acelist = []
                        mapelist = []     # list of MAPE values for multiple realization without intervention
                        mapelistint = []  # list of MAPE values for multiple realization with intervention
                        css_score = []    # list of causal scores for multiple realization
                        diff = []
                        start_batch = 10

                        for iter in range(20):  # 18
                            
                            mselist_batch = []
                            mselistint_batch = []
                            mapelist_batch = []
                            mapelistint_batch = []
                            for r in range(2):

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

                                mse, mape, ypred = modelTest(model, test_ds, num_samples, test_data[j], j,
                                                            prediction_length, iter, False, 0)

                                mseint, mapeint, ypredint = modelTest(model, test_dsint, num_samples,
                                                                    test_data[j], j,
                                                                    prediction_length, iter, True, m)

                                if m == 0:
                                     # create a text trap and redirect stdout
                                    text_trap = io.StringIO()
                                    sys.stdout = text_trap
                                    # Generate multiple version Knockoffs
                                    data_actual = np.array(odata[: , start_batch: start_batch + training_length + prediction_length]).transpose()
                                    obj = Knockoffs()
                                    n = len(odata[:, 0])
                                    # knockoffs = obj.Generate_Knockoffs(n, params.get("dim"), data_actual)
                                    knockoff_samples = np.array(knockoffs[:, start_cause: end_cause]).transpose()
                                    intervene = knockoff_samples

                                    # now restore stdout function
                                    sys.stdout = sys.__stdout__

                                # np.random.shuffle(intervene)
                                mselist_batch.append(mse)
                                mapelist_batch.append(mape)
                                mselistint_batch.append(mseint)
                                mapelistint_batch.append(mapeint)

                            start_batch = start_batch + 10                          # Step size for sliding window # 15
                            mselist.append(np.mean(mselist_batch))                  # mselist = mselist_batch
                            mapelist.append(np.mean(mapelist_batch))                # mapelist = mapelist_batch
                            mselistint.append(np.mean(mselistint_batch))            # mselistint = mselistint_batch
                            mapelistint.append(np.mean(mapelistint_batch))          # mapelistint = mapelistint_batch

                        msevar = np.var(mselist)
                        mapevar = np.var(mapelist)
                        mselol.append(mselist)
                        mapelol.append(mapelist)
                        mselolint.append(mselistint)
                        mapelolint.append(mapelistint)

                    var_list.append(np.var(mapelolint[0]))
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
                    
                    for z in range(len(heuristic_itn_types)):

                        print("Intervention: " + heuristic_itn_types[z])
                        t, p = ks_2samp(np.array(mapelol[z]), np.array(mapelolint[z]))
                        # t, p = kstest(np.array(mapelolint[z]), np.array(mapelol[z]))
                        pvals.append(1-p)
                        
                        print(f'Test statistic: {t}, p-value: {p}')
                        if p < 0.05:
                            print("Null hypothesis is rejected")
                            causal_decision.append(1)
                        else:
                            print("Fail to reject null hypothesis")
                            causal_decision.append(0)

                    pvi.append(pvals[0])
                    pvu.append(pvals[1])

                    # plot residuals distribution
                    fig = plt.figure()
                    ax1 = fig.add_subplot(111)
                    sns.distplot(mapelol[0], color='red', label='Actual')
                    sns.distplot(mapelolint[0], color='green', label='Counterfactual')
                    
                    mapeslol.append(mapelol[0])
                    mapeslolint.append(mapelolint[0])
                    
                    if len(columns) > 0:
                        # plt.ylabel(f"CSS: {columns[i]} ---> {columns[j]}")
                        ax1.set_ylabel(f"{cause_group} ---> {columns[j]}")
                    else:
                        # plt.ylabel(f"CSS: Z_{i + 1} ---> Z_{j + 1}")
                        ax1.set_ylabel(f"{cause_group} ---> Z_{j + 1}")

                    plt.gcf()
                    ax1.legend()
                    filename = pathlib.Path(plot_path + f"{cause_group} ---> {columns[j]}.pdf")
                    plt.savefig(filename)
                    # plt.show()
                    # plt.close()

                    cause_list.append(causal_decision[0])
                    indist_cause.append(causal_decision[0])
                    uni_cause.append(causal_decision[1])
                    causal_decision = []
                
                if h == group_num-1 or g==group_num-1:
                    
                    for q in range(1, end_effect-start_effect):
                    
                        mape_df = pd.DataFrame(data=np.transpose(mapeslol), columns=columns[start_effect: end_effect])
                        mape_int_df = pd.DataFrame(data=np.transpose(mapeslolint), columns=columns[start_effect: end_effect])

                        # Create a single plot
                        fig = plt.figure()
                        ax2 = fig.add_subplot(111)

                        # Plot the first bivariate distribution with transparency
                        sns.kdeplot(data=mape_df, x=columns[start_effect], y=columns[start_effect+q], cmap='Blues', alpha=0.80, fill=True, levels=5, color='blue', label='Actual') #fill= True, cmap="Blues", alpha=0.5

                        # Plot the second bivariate distribution on top with transparency
                        sns.kdeplot(data=mape_int_df, x=columns[start_effect], y=columns[start_effect+q], cmap='Reds', alpha=0.55, fill=True, levels=5, color='red', label='Counterfactual') # fill=True, cmap="Reds", fill=True, cmap="Reds",

                        if len(columns) > 0:
                            # plt.ylabel(f"CSS: {columns[i]} ---> {columns[j]}")
                            ax1.set_ylabel(f"{cause_group} ---> {columns[q]}")
                        else:
                            # plt.ylabel(f"CSS: Z_{i + 1} ---> Z_{j + 1}")
                            ax1.set_ylabel(f"{cause_group} ---> Z_{q + 1}")

                        # Show the plot
                        plt.gcf()

                        # Add a custom legend
                        legend_elements = [
                        Patch(facecolor=plt.cm.Blues(100), alpha=0.70, edgecolor='k', label='Actual'),
                        Patch(facecolor=plt.cm.Reds(100), alpha=0.85, edgecolor='k', label='Counterfactual')
                        ]
                        ax2.legend(handles=legend_elements)
                        filename = pathlib.Path(plot_path + f"{cause_group} ---> {columns[q+start_effect]}_2d.pdf")
                        plt.savefig(filename)
                        # plt.show()

        causal_direction.append(cause_list)
    
    pval_indist.append(pvi)
    pval_uniform.append(pvu)

    conf_mat_indist = conf_mat_indist + indist_cause
    conf_mat_uniform = conf_mat_uniform + uni_cause
    indist_cause, uni_cause = [], []

    pvalues.append(pval_indist)
    pvalues.append(pval_uniform)
    print("P-Values: ", pvalues)

    conf_mat.append(conf_mat_indist)
    conf_mat.append(conf_mat_uniform)
    print("--------------------------------------------------------")
    print("Discovered Causal Graphs: ", conf_mat)

    print(f'Causal direction: {causal_direction}')

    for m in range(group_num):
        for n in range(group_num):
            if m < n:
                c1, c2 = causal_criteria(causal_direction[m], causal_direction[n])

                if c1 > c2:
                    print(f'gCDMI: Group {m+1} causes Group {n+1}')
                elif c2 > c1:
                    print(f'gCDMI: Group {n+1} causes Group {m+1}')
                elif int(c1) & int(c2) == 0:
                    print(f'gCDMI: No causal connection found in Group {m+1} and Group {n+1}')
                elif c1 == c2:
                    print('gCDMI: Causal direction can\'t be inferred')
    
    end_time = time.time()

    return conf_mat, end_time 





