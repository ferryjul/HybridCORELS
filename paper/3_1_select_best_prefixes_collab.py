from HybridCORELS import *
import numpy as np 
from exp_utils import get_data
import matplotlib.pyplot as plt 
import pickle
import pandas as pd 

policies = ['objective', 'lower_bound', 'bfs']
cList = [0.001, 0.0001, 0.00001]
min_coverageList = [0.25, 0.50, 0.75, 0.85, 0.95] #np.concatenate([np.arange(0, 1.0, 0.05), np.arange(0.96, 0.99, 0.01)])
alphaList = np.arange(0, 11, 1) # 11 values
dataset_seeds = [0,1,2,3,4]
min_support_list = [0.01, 0.05, 0.1]
paramsList = []
datasets = ["compas", "adult", "acs_employ"]
n_iter_param = 10**9

for dataset_name in datasets:

    fileName = './results_part_3_collab/results_HybridCORELSPreCollab_%s.csv' %(dataset_name)
    dataset_cov_res_dict = {}
    try:
        res = pd.read_csv(fileName)
    except FileNotFoundError():
        print("File not found: ", fileName)
        exit()

    # Iterate over results
    for index, row in res.iterrows():
        # Redundant checks
        assert(row['Alpha'] == 0)
        assert(row['Black Box'] == "fixed_rf")

        # Run type
        rseed = row['Seed']
        min_coverage_param = row['Min coverage']
        
        # Run perfs
        search_status = row['Search status']
        train_acc_ub = row['Train acc UB']
        transparency_train = row['Training transparency']
        transparency_test = row['Test transparency']
        #interpr_accuracy_train = row['interpr_accuracy_train']
        #interpr_accuracy_test = row['interpr_accuracy_test']

        # Run params
        cValue = row['Lambda']
        min_support_param = row['Min support']
        policy = row['Policy']
        params_dict = {"policy":policy, "c":cValue, "min_support":min_support_param}

        if not rseed in dataset_cov_res_dict.keys():
            dataset_cov_res_dict[rseed] = {}
        if not min_coverage_param in dataset_cov_res_dict[rseed].keys():
            dataset_cov_res_dict[rseed][min_coverage_param] = []
        dataset_cov_res_dict[rseed][min_coverage_param].append({'status':search_status, 'train_acc_ub':train_acc_ub, 'transparency_train': transparency_train, 'transparency_test':transparency_test, 'params_dict':params_dict})

    accs_list = {}
    cov_list = {}
    for random_state_value in dataset_seeds:
        for min_coverage in min_coverageList:
            if not min_coverage in accs_list.keys():
                accs_list[min_coverage] = []
                cov_list[min_coverage] = []
            best_acc = -1
            best_cov = -1
            best_params = {}
            best_status = "NA"
            for a_config in dataset_cov_res_dict[random_state_value][min_coverage]:
                if best_acc < a_config['train_acc_ub']:
                    best_acc = a_config['train_acc_ub']
                    best_params = a_config['params_dict']
                    best_status = a_config['status']
                    best_cov = a_config['transparency_train']
                elif (best_acc == a_config['train_acc_ub']) and (a_config['status'] == "opt" and best_status != "opt"):
                    est_acc = a_config['train_acc_ub']
                    best_params = a_config['params_dict']
                    best_status = a_config['status']
                    best_cov = a_config['transparency_train']
            accs_list[min_coverage].append(best_acc)
            cov_list[min_coverage].append(best_cov)

            dict_name = "prefixes_dict_collab/%s_%d_%.4f_collab" %(dataset_name, random_state_value, min_coverage)
            with open('%s.pickle' %dict_name, 'wb') as handle:
                pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('%s.pickle' %dict_name, 'rb') as handle:
                b = pickle.load(handle)
                print("Dataset %s, Fold %d, Min Coverage %.2f, best params are :" %(dataset_name, random_state_value, min_coverage), b, " (accuracy %.4f, status %s)" %(best_acc, best_status))
    
    accs_list_list = [np.mean(accs_list[min_coverage]) for min_coverage in min_coverageList]
    actual_coverageList = [np.mean(cov_list[min_coverage]) for min_coverage in min_coverageList]

    fig,ax = plt.subplots()
    ax.plot(min_coverageList, accs_list_list, marker='x', label="hybrid model accuracy (UB)")
    ax.set_ylabel("Hybrid Model Accuracy UB")
    plt.xlabel("Min. Transparency Constraint")

    ax2=ax.twinx()

    ax2.plot(min_coverageList, actual_coverageList, marker = 'o', c='orange', label="actual prefix coverage")
    ax2.set_ylabel("Actual Prefix Coverage")

    #plt.legend(loc='best')
    plt.savefig("./figures/best_prefixes_dataset_%s_collab.png" %dataset_name, bbox_inches='tight')
    plt.clf()