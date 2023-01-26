from HybridCORELS import *
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import pickle
from matplotlib.lines import Line2D
import matplotlib
from exp_utils import get_data, computeAccuracyUpperBound
import pandas as pd 

matplotlib.rcParams.update({'font.size': 12.0}) # default = 10.0

show_std = True

policies = ['objective', 'lower_bound', 'bfs']
cList = [0.001, 0.0001, 0.00001]
min_coverageList = [0.25, 0.50, 0.75, 0.85, 0.95] #np.concatenate([np.arange(0, 1.0, 0.05), np.arange(0.96, 0.99, 0.01)])
alphaList = np.arange(0, 11, 1) # 11 values
dataset_seeds = [0,1,2,3,4]
min_support_list = [0.01, 0.05, 0.1]
paramsList = []
datasets = ["acs_employ", "compas", "adult"]
n_iter_param = 10**9

accuracy_train_color = 'navy'
transparency_train_color = 'cyan'
accuracy_test_color = 'darkgreen'
transparency_test_color = 'lime'

legendFig = plt.figure("Legend plot")
legend_elements = []
legend_elements.append(Line2D([0], [0], marker='x', color=accuracy_train_color, lw=1, label="Overall Accuracy Upper Bound (train)")) #"Prefix Accuracy (train)")) # linestyle = 'None',
legend_elements.append(Line2D([0], [0], marker='o', color=transparency_train_color, lw=1, label="Actual Prefix Coverage (train)"))
legend_elements.append(Line2D([0], [0], marker='x', color=accuracy_test_color, lw=1, label="Overall Accuracy Upper Bound (test)")) #"Prefix Accuracy (test)")) # linestyle = 'None',
legend_elements.append(Line2D([0], [0], marker='o', color=transparency_test_color, lw=1, label="Actual Prefix Coverage (test)"))
legendFig.legend(handles=legend_elements, loc='center', ncol=2)
legendFig.savefig('./figures/best_prefixes_legend_collab.pdf', bbox_inches='tight')

for dataset_name in datasets:
    accs_list = {}
    cov_list = {}
    accs_list_test = {}
    cov_list_test = {}
    save_dir = "./results_part_3_collab"
    fileName = '%s/results_HybridCORELSPre_wBB_%s_with_ub_lb_collab.csv' %(save_dir, dataset_name)
    dataset_cov_res_dict = {}
    #try:
    res = pd.read_csv(fileName)
    #except:
    #    print("File not found: ", fileName)
    #    exit()

    # Iterate over results
    for index, row in res.iterrows():
        # Only care about one BB/alpha (cause BB acc UB is the same for all and only depends on the previously learnt prefix)
        if row['Alpha'] == 0 and row['Black Box'] == "random_forest":
            # Run type
            rseed = row['Seed']
            min_coverage = row['Min coverage']
            
            # Run perfs
            search_status = row['Search status']
            transparency_train = row['Training transparency']
            transparency_test = row['Test transparency']

            interpr_accuracy_train = row['Training accuracy (prefix)']
            interpr_accuracy_test = row['Test accuracy (prefix)']

            black_box_acc_upper_bound_train = row['Training accuracy (BB) UB']
            black_box_acc_upper_bound_test = row['Test accuracy (BB) UB']

            hybrid_accuracy_ub_train = (interpr_accuracy_train * transparency_train) + (black_box_acc_upper_bound_train * (1 - transparency_train))
            hybrid_accuracy_ub_test = (interpr_accuracy_test * transparency_test) + (black_box_acc_upper_bound_test * (1 - transparency_test))
            
            if not min_coverage in accs_list.keys():
                accs_list[min_coverage] = []
                cov_list[min_coverage] = []
                accs_list_test[min_coverage] = []
                cov_list_test[min_coverage] = []
                
            #accs_list[min_coverage].append(interpr_accuracy_train)
            #accs_list_test[min_coverage].append(interpr_accuracy_test)
            accs_list[min_coverage].append(hybrid_accuracy_ub_train)
            accs_list_test[min_coverage].append(hybrid_accuracy_ub_test)
            
            cov_list[min_coverage].append(transparency_train)
            cov_list_test[min_coverage].append(transparency_test)
            
    accs_list_list = np.asarray([np.mean(accs_list[min_coverage]) for min_coverage in min_coverageList])
    accs_list_list_std = np.asarray([np.std(accs_list[min_coverage]) for min_coverage in min_coverageList])
    actual_coverageList = np.asarray([np.mean(cov_list[min_coverage]) for min_coverage in min_coverageList])
    actual_coverageList_std = np.asarray([np.std(cov_list[min_coverage]) for min_coverage in min_coverageList])

    accs_list_list_test = np.asarray([np.mean(accs_list_test[min_coverage]) for min_coverage in min_coverageList])
    accs_list_list_test_std = np.asarray([np.std(accs_list_test[min_coverage]) for min_coverage in min_coverageList])
    actual_coverageList_test = np.asarray([np.mean(cov_list_test[min_coverage]) for min_coverage in min_coverageList])
    actual_coverageList_test_std = np.asarray([np.std(cov_list_test[min_coverage]) for min_coverage in min_coverageList])

    fig,ax = plt.subplots()
    ax.plot(min_coverageList, accs_list_list, marker='x', c=accuracy_train_color, label="train accuracy (prefix)")
    if show_std:
        ax.fill_between(min_coverageList, accs_list_list - accs_list_list_std, accs_list_list + accs_list_list_std, color=accuracy_train_color, alpha=0.2)

    ax.set_ylabel("Overall Accuracy Upper Bound") #"Prefix Accuracy")
    plt.xlabel("Min. Transparency Constraint")

    ax2=ax.twinx()

    ax2.plot(min_coverageList, actual_coverageList, marker = 'o', c=transparency_train_color, label="actual prefix coverage")
    if show_std:
        ax2.fill_between(min_coverageList, actual_coverageList - actual_coverageList_std, actual_coverageList + actual_coverageList_std, color=transparency_train_color, alpha=0.2)

    ax2.set_ylabel("Actual Prefix Coverage")

    #plt.legend(loc='best')
    plt.savefig("./figures/best_prefixes_dataset_%s_train_collab.png" %dataset_name, bbox_inches='tight')
    plt.savefig("./figures/best_prefixes_dataset_%s_train_collab.pdf" %dataset_name, bbox_inches='tight')

    # Test Figure
    fig,ax = plt.subplots()
    ax.plot(min_coverageList, accs_list_list_test, marker='x', c=accuracy_test_color, label="test accuracy (prefix)")
    if show_std:
        ax.fill_between(min_coverageList, accs_list_list_test - accs_list_list_test_std, accs_list_list_test + accs_list_list_test_std, color=accuracy_test_color, alpha=0.2)

    ax.set_ylabel("Overall Accuracy Upper Bound")#"Prefix Accuracy")
    plt.xlabel("Min. Transparency Constraint")

    ax2=ax.twinx()

    ax2.plot(min_coverageList, actual_coverageList_test, marker = 'o', c=transparency_test_color, label="actual prefix coverage")
    if show_std:
        ax2.fill_between(min_coverageList, actual_coverageList_test - actual_coverageList_test_std, actual_coverageList_test + actual_coverageList_test_std, color=transparency_test_color, alpha=0.2)

    ax2.set_ylabel("Actual Prefix Coverage")

    #plt.legend(loc='best')
    plt.savefig("./figures/best_prefixes_dataset_%s_test_collab.png" %dataset_name, bbox_inches='tight')
    plt.savefig("./figures/best_prefixes_dataset_%s_test_collab.pdf" %dataset_name, bbox_inches='tight')

    plt.clf()