from HybridCORELS import *
import numpy as np 
from exp_utils import get_data
import matplotlib
import matplotlib.pyplot as plt 
import pickle
from matplotlib.lines import Line2D
import matplotlib

matplotlib.rcParams.update({'font.size': 12.0}) # default = 10.0

show_std = True

policies = ['objective', 'lower_bound', 'bfs']
cList = [0.001, 0.0001, 0.00001]
min_coverageList = [0.25, 0.50, 0.75, 0.85, 0.95] #np.concatenate([np.arange(0, 1.0, 0.05), np.arange(0.96, 0.99, 0.01)])
alphaList = np.arange(0, 11, 1) # 11 values
dataset_seeds = [0,1,2,3,4]
min_support_list = [0.01, 0.05, 0.1]
paramsList = []
datasets = ["compas", "adult", "acs_employ"]
n_iter_param = 10**9

accuracy_train_color = 'navy'
transparency_train_color = 'cyan'
accuracy_test_color = 'darkgreen'
transparency_test_color = 'lime'

legendFig = plt.figure("Legend plot")
legend_elements = []
legend_elements.append(Line2D([0], [0], marker='x', color=accuracy_train_color, lw=1, label="Prefix Accuracy (train)")) # linestyle = 'None',
legend_elements.append(Line2D([0], [0], marker='o', color=transparency_train_color, lw=1, label="Actual Prefix Coverage (train)"))
legend_elements.append(Line2D([0], [0], marker='x', color=accuracy_test_color, lw=1, label="Prefix Accuracy (test)")) # linestyle = 'None',
legend_elements.append(Line2D([0], [0], marker='o', color=transparency_test_color, lw=1, label="Actual Prefix Coverage (test)"))
legendFig.legend(handles=legend_elements, loc='center', ncol=2)
legendFig.savefig('./figures/best_prefixes_legend.pdf', bbox_inches='tight')

for dataset_name in datasets:
    accs_list = {}
    cov_list = {}
    accs_list_test = {}
    cov_list_test = {}
    for random_state_value in dataset_seeds:
        X, y, features, prediction = get_data(dataset_name, {"train" : 0.8, "test" : 0.2}, random_state_param=random_state_value)
        X_train, X_test, y_train, y_test = X['train'], X['test'], y['train'], y['test']
        
        for min_coverage in min_coverageList:

            if not min_coverage in accs_list.keys():
                accs_list[min_coverage] = []
                cov_list[min_coverage] = []
                accs_list_test[min_coverage] = []
                cov_list_test[min_coverage] = []

            dict_name = "prefixes_dict/%s_%d_%.4f" %(dataset_name, random_state_value, min_coverage)
            
            with open('%s.pickle' %dict_name, 'rb') as handle:
                best_params = pickle.load(handle)
                min_support_param = best_params['min_support']
                policy = best_params['policy']
                cValue = best_params['c']
                print("Dataset %s, Fold %d, Min Coverage %.2f, best params are :" %(dataset_name, random_state_value, min_coverage), best_params)

                fileName = "./models_part_3/prefix_%s_%d_%.3f_%.5f_%d_%.2f_%s" %(dataset_name, random_state_value, min_coverage, cValue, n_iter_param, min_support_param, policy)
                try:
                    model = HybridCORELSPreClassifier.load(fileName)
                except FileNotFoundError:
                    print("Missing model: ", fileName)
                    exit()
                assert(model.n_iter == n_iter_param)
                assert(model.policy == policy)
                assert(model.min_support == min_support_param)
                assert(model.c == cValue)
                assert(model.min_coverage == min_coverage)
                preds_train, preds_types_train  = model.predict_with_type(X_train)
                interpr_indices_train = np.where(preds_types_train == 1)
                interpr_accuracy_train = np.mean(preds_train[interpr_indices_train] == y_train[interpr_indices_train])
                preds_types_counts_train = np.unique(preds_types_train, return_counts=True)
                index_one_train = np.where(preds_types_counts_train[0] == 1)
                transparency_train = preds_types_counts_train[1][index_one_train][0]/np.sum(preds_types_counts_train[1])

                preds_test, preds_types_test  = model.predict_with_type(X_test)
                interpr_indices_test = np.where(preds_types_test == 1)
                interpr_accuracy_test = np.mean(preds_test[interpr_indices_test] == y_test[interpr_indices_test])
                preds_types_counts_test = np.unique(preds_types_test, return_counts=True)
                index_one_test = np.where(preds_types_counts_test[0] == 1)
                transparency_test = preds_types_counts_test[1][index_one_test][0]/np.sum(preds_types_counts_test[1])

                accs_list[min_coverage].append(interpr_accuracy_train)
                cov_list[min_coverage].append(transparency_train)
                accs_list_test[min_coverage].append(interpr_accuracy_test)
                cov_list_test[min_coverage].append(transparency_test)

              
    accs_list_list = np.asarray([np.mean(accs_list[min_coverage]) for min_coverage in min_coverageList])
    accs_list_list_std = np.asarray([np.std(accs_list[min_coverage]) for min_coverage in min_coverageList])
    actual_coverageList = np.asarray([np.mean(cov_list[min_coverage]) for min_coverage in min_coverageList])
    actual_coverageList_std = np.asarray([np.std(cov_list[min_coverage]) for min_coverage in min_coverageList])

    accs_list_list_test = np.asarray([np.mean(accs_list_test[min_coverage]) for min_coverage in min_coverageList])
    accs_list_list_test_std = np.asarray([np.std(accs_list_test[min_coverage]) for min_coverage in min_coverageList])
    actual_coverageList_test = np.asarray([np.mean(cov_list_test[min_coverage]) for min_coverage in min_coverageList])
    actual_coverageList_test_std = np.asarray([np.std(cov_list_test[min_coverage]) for min_coverage in min_coverageList])

    # Train Figure
    #plt.figure(figsize=(4, 3))


    fig,ax = plt.subplots()
    ax.plot(min_coverageList, accs_list_list, marker='x', c=accuracy_train_color, label="train accuracy (prefix)")
    if show_std:
        ax.fill_between(min_coverageList, accs_list_list - accs_list_list_std, accs_list_list + accs_list_list_std, color=accuracy_train_color, alpha=0.2)

    ax.set_ylabel("Prefix Accuracy")
    plt.xlabel("Min. Transparency Constraint")

    ax2=ax.twinx()

    ax2.plot(min_coverageList, actual_coverageList, marker = 'o', c=transparency_train_color, label="actual prefix coverage")
    if show_std:
        ax2.fill_between(min_coverageList, actual_coverageList - actual_coverageList_std, actual_coverageList + actual_coverageList_std, color=transparency_train_color, alpha=0.2)

    ax2.set_ylabel("Actual Prefix Coverage")

    #plt.legend(loc='best')
    plt.savefig("./figures/best_prefixes_dataset_%s_train.png" %dataset_name, bbox_inches='tight')
    plt.savefig("./figures/best_prefixes_dataset_%s_train.pdf" %dataset_name, bbox_inches='tight')

    # Test Figure
    fig,ax = plt.subplots()
    ax.plot(min_coverageList, accs_list_list_test, marker='x', c=accuracy_test_color, label="test accuracy (prefix)")
    if show_std:
        ax.fill_between(min_coverageList, accs_list_list_test - accs_list_list_test_std, accs_list_list_test + accs_list_list_test_std, color=accuracy_test_color, alpha=0.2)

    ax.set_ylabel("Prefix Accuracy")
    plt.xlabel("Min. Transparency Constraint")

    ax2=ax.twinx()

    ax2.plot(min_coverageList, actual_coverageList_test, marker = 'o', c=transparency_test_color, label="actual prefix coverage")
    if show_std:
        ax2.fill_between(min_coverageList, actual_coverageList_test - actual_coverageList_test_std, actual_coverageList_test + actual_coverageList_test_std, color=transparency_test_color, alpha=0.2)

    ax2.set_ylabel("Actual Prefix Coverage")

    #plt.legend(loc='best')
    plt.savefig("./figures/best_prefixes_dataset_%s_test.png" %dataset_name, bbox_inches='tight')
    plt.savefig("./figures/best_prefixes_dataset_%s_test.pdf" %dataset_name, bbox_inches='tight')

    plt.clf()