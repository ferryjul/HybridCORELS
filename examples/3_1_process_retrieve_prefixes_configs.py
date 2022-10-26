from HybridCORELS import *
import numpy as np 
from exp_utils import get_data
import matplotlib.pyplot as plt 
import pickle

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
    accs_list = {}
    cov_list = {}
    for random_state_value in dataset_seeds:
        X, y, features, prediction = get_data(dataset_name, {"train" : 0.8, "test" : 0.2}, random_state_param=random_state_value)
        X_train, X_test, y_train, y_test = X['train'], X['test'], y['train'], y['test']
        
        for min_coverage in min_coverageList:
            if not min_coverage in accs_list.keys():
                accs_list[min_coverage] = []
                cov_list[min_coverage] = []
            best_acc = -1
            best_cov = -1
            best_params = {}
            best_status = "NA"
            for policy in policies:
                for cValue in cList:
                    for min_support_param in min_support_list:
                        fileName = "./models_graham/prefix_%s_%d_%.3f_%.5f_%d_%.2f_%s" %(dataset_name, random_state_value, min_coverage, cValue, n_iter_param, min_support_param, policy)
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

                        if best_acc < interpr_accuracy_train:
                            best_acc = interpr_accuracy_train
                            best_params = {"policy":policy, "c":cValue, "min_support":min_support_param}
                            best_status = model.get_status()
                            best_cov = transparency_train
                        elif (best_acc == interpr_accuracy_train) and (model.get_status() == "opt" and best_status != "opt"):
                            best_acc = interpr_accuracy_train
                            best_params = {"policy":policy, "c":cValue, "min_support":min_support_param}
                            best_status = model.get_status()
                            best_cov = transparency_train
            accs_list[min_coverage].append(best_acc)
            cov_list[min_coverage].append(best_cov)

            dict_name = "prefixes_dict/%s_%d_%.4f" %(dataset_name, random_state_value, min_coverage)
            with open('%s.pickle' %dict_name, 'wb') as handle:
                pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('%s.pickle' %dict_name, 'rb') as handle:
                b = pickle.load(handle)
                print("Dataset %s, Fold %d, Min Coverage %.2f, best params are :" %(dataset_name, random_state_value, min_coverage), b, " (accuracy %.4f, status %s)" %(best_acc, best_status))
    
    accs_list_list = [np.mean(accs_list[min_coverage]) for min_coverage in min_coverageList]
    actual_coverageList = [np.mean(cov_list[min_coverage]) for min_coverage in min_coverageList]

    fig,ax = plt.subplots()
    ax.plot(min_coverageList, accs_list_list, marker='x', label="train accuracy (prefix)")
    ax.set_ylabel("Prefix Accuracy")
    plt.xlabel("Min. Transparency Constraint")

    ax2=ax.twinx()

    ax2.plot(min_coverageList, actual_coverageList, marker = 'o', c='orange', label="actual prefix coverage")
    ax2.set_ylabel("Actual Prefix Coverage")

    #plt.legend(loc='best')
    plt.savefig("./figures/best_prefixes_dataset_%s.png" %dataset_name, bbox_inches='tight')
    plt.clf()