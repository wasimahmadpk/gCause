import pickle
import pathlib
import parameters
import numpy as np
import mxnet as mx
import seaborn as sns
import preprocessing as prep
import matplotlib.pyplot as plt
from knockoffs import Knockoffs
from forecast import modelTest
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar._network import DeepARTrainingNetwork
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput
from scipy.stats import ttest_ind, ttest_ind_from_stats, ttest_1samp, ks_2samp, kstest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

np.random.seed(1)
mx.random.seed(2)

pars = parameters.get_syn_params()
num_samples = pars.get("num_samples")
step = pars.get("step_size")
training_length = pars.get("train_len")
prediction_length = pars.get("pred_len")
frequency = pars.get("freq")
plot_path = pars.get("plot_path")
# Get prior skeleton
prior_graph = pars.get('prior_graph')
true_conf_mat = pars.get("true_graph")

group_num = pars.get('group_num')
groups = pars.get('groups')


def deepCause(odata, knockoffs, model, params):

    filename = pathlib.Path(model)
    if not filename.exists():
        print("Training forecasting model....")
        predictor = estimator.train(train_ds)
        # save the model to disk
        pickle.dump(predictor, open(filename, 'wb'))

    conf_mat, conf_mat_indist, conf_mat_uniform = [], [], []
    pvalues, pval_indist, pval_uniform = [], [], []
    kvalues, kval_indist, kval_uniform = [], [], []
    var_list, causal_decision, indist_cause, uni_cause = [], [], [], []

    for g in range(group_num):
        
        for h in range(group_num):

            if g != h:

                
                start_effect = groups.get(f'g{h+1}')[0]
                end_effect = groups.get(f'g{h+1}')[1]
                start_cause = groups.get(f'g{g+1}')[0]
                end_cause = groups.get(f'g{g+1}')[1]
                cause_group = f'Group: {g+1}'
                effect_group = f'Group: {h+1}' 
                
                # P-Values
                pvi, pvu = [], []

                # KL-Divergence
                kvi, kvu = [], []

                # Generate Knockoffs
                data_actual = np.array(odata[: , 0: training_length + prediction_length]).transpose()
                obj = Knockoffs()
                n = len(odata[:, 0])
                knockoffs = obj.Generate_Knockoffs(n, params.get("dim"), data_actual)
                
                knockoff_samples = np.array(knockoffs[:, start_cause: end_cause]).transpose()
                uniform_intervention = np.random.uniform(np.min(odata), np.min(odata), knockoff_samples.shape)
                
                interventionlist = [knockoff_samples, uniform_intervention]
                heuristic_itn_types = ['In-dist', 'Uniform']


                for j in range(start_effect, end_effect):
                
                    print("----------*****-----------------------*****-----------------******-----------")

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

                        for iter in range(10):  # 30
                            
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
                                    # Generate multiple version Knockoffs
                                    data_actual = np.array(odata[: , start_batch: start_batch + training_length + prediction_length]).transpose()
                                    obj = Knockoffs()
                                    n = len(odata[:, 0])
                                    knockoffs = obj.Generate_Knockoffs(n, params.get("dim"), data_actual)
                                    knockoff_samples = np.array(knockoffs[:, start_cause: end_cause]).transpose()
                                    intervene = knockoff_samples

                                # np.random.shuffle(intervene)
                                mselist_batch.append(mse)
                                mapelist_batch.append(mape)
                                mselistint_batch.append(mseint)
                                mapelistint_batch.append(mapeint)

                            start_batch = start_batch + 5                                       # Step size for sliding window # 10
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
                        print("----------*****-----------------------*****-----------------******-----------")
                        fnamehist = plot_path + "{columns[i]}_{columns[j]}:hist"
                    else:
                        print(f"Causal Link: {cause_group} --------------> Z_{j + 1}")
                        print("----------*****-----------------------*****-----------------******-----------")
                        fnamehist = plot_path + "{Z_[i + 1]}_{Z_[j + 1]}:hist"
                    
                    pvals, kvals = [], []
                    
                    for z in range(len(heuristic_itn_types)):

                        print("Intervention: " + heuristic_itn_types[z])
                        
                        # print(f"Mean: {np.mean(mapelol[z])}, Mean Intervention: {np.mean(mapelolint[z])}")
                        # print(f"Variance: {np.var(mapelol[z])}, Variance Intervention: {np.var(mapelolint[z])}")
                        # t, p = ttest_ind(np.array(mapelolint[z]), np.array(mapelol[z]), equal_var=True)
                        
                        t, p = ks_2samp(np.array(mapelol[z]), np.array(mapelolint[z]))
                        # t, p = kstest(np.array(mapelolint[z]), np.array(mapelol[z]))
                        
                        kld = prep.kl_divergence(np.array(mapelol[z]), np.array(mapelolint[z]))
                        kvals.append(kld)
                        
                        pvals.append(1-p)
                        
                        print(f'Test statistic: {t}, p-value: {p}, KLD: {kld}')
                        if p < 0.10:
                            print("Null hypothesis is rejected")
                            causal_decision.append(1)
                        else:
                            print("Fail to reject null hypothesis")
                            causal_decision.append(0)

                    pvi.append(pvals[0])
                    pvu.append(pvals[1])

                    kvi.append(kvals[0])
                    kvu.append(kvals[1])

                    # plot residuals distribution
                    fig = plt.figure()
                    ax1 = fig.add_subplot(111)
                    sns.distplot(mapelol[0], color='red', label='Actual')
                    sns.distplot(mapelolint[0], color='green', label='Counterfactual')

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

                    indist_cause.append(causal_decision[0])
                    uni_cause.append(causal_decision[1])
                    causal_decision = []
        
    pval_indist.append(pvi)
    pval_uniform.append(pvu)

    kval_indist.append(kvi)
    kval_uniform.append(kvu)

    conf_mat_indist = conf_mat_indist + indist_cause
    conf_mat_uniform = conf_mat_uniform + uni_cause
    indist_cause, uni_cause = [], []


    pvalues.append(pval_indist)
    pvalues.append(pval_uniform)
    print("P-Values: ", pvalues)

    kvalues.append(kval_indist)
    kvalues.append(kval_uniform)
    print("KL-Divergence: ", kvalues)

    conf_mat.append(conf_mat_indist)
    conf_mat.append(conf_mat_uniform)
    print("-----------------------------------------------------------------------------")
    print("Discovered Causal Graphs: ", conf_mat)

    # for ss in range(len(conf_mat)):

    #     # true_conf_mat = conf_mat[ss]
    #     fscore = round(f1_score(true_conf_mat, conf_mat[ss], average='binary'), 2)
    #     acc = accuracy_score(true_conf_mat, conf_mat[ss])
    #     tn, fp, fn, tp = confusion_matrix(true_conf_mat, conf_mat[ss], labels=[0, 1]).ravel()
    #     precision = precision_score(true_conf_mat, conf_mat[ss])
    #     recall = recall_score(true_conf_mat, conf_mat[ss])
        
    #     print("---------***-----------***----------***----------")
    #     print(f"Intervention: {heuristic_itn_types[ss]}")
    #     print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    #     print(f"Precision: {precision}")
    #     print(f"Recall: {recall}")
    #     print(f"Accuracy: {acc}")
    #     print(f"F-score: {fscore}")
    #     print("---------***-----------***----------***----------")