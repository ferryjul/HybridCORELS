import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

save_dir = "./results/expes_alpha_v1_graham"
dataset = "compas"
fileName = '%s/results_expes_alpha_%s.csv' %(save_dir, dataset)
n_folds = 5
# header = ["fold_id", "min_coverage", "alpha_value", "train_acc", "train_coverage", "test_acc", "test_coverage", "search_status", "sparsity", "model"]
all_opt = True
results_df = pd.read_csv(fileName)
results_dict = {}
for index, row in results_df.iterrows():
    alpha_val = row["alpha_value"]
    train_acc = row["train_acc"]
    test_acc = row["test_acc"]
    min_coverage = row["min_coverage"]

    if row["search_status"] != "opt":
        all_opt = False

    if not (min_coverage in results_dict.keys()):
        results_dict[min_coverage] = {}

    if alpha_val in results_dict[min_coverage].keys():
        results_dict[min_coverage][alpha_val]["train_acc"].append(train_acc)
        results_dict[min_coverage][alpha_val]["test_acc"].append(test_acc)
    else:
        results_dict[min_coverage][alpha_val] = {}
        results_dict[min_coverage][alpha_val]["train_acc"] = [train_acc]
        results_dict[min_coverage][alpha_val]["test_acc"] = [test_acc]

plot_dicts = {}

for min_coverage in results_dict.keys():
    plot_dicts[min_coverage] = {}
    plot_dicts[min_coverage]["alpha_val_list"] = []
    plot_dicts[min_coverage]["train_accs_list"] = []
    plot_dicts[min_coverage]["test_accs_list"] = []
    for alpha_val in results_dict[min_coverage].keys():
        plot_dicts[min_coverage]["alpha_val_list"].append(alpha_val)
        assert(len(results_dict[min_coverage][alpha_val]["train_acc"]) == n_folds)
        assert(len(results_dict[min_coverage][alpha_val]["test_acc"]) == n_folds)
        plot_dicts[min_coverage]["train_accs_list"].append(np.average(results_dict[min_coverage][alpha_val]["train_acc"]))
        plot_dicts[min_coverage]["test_accs_list"].append(np.average(results_dict[min_coverage][alpha_val]["test_acc"]))

for min_coverage in plot_dicts.keys():
    plt.title("Min Coverage = %.2f" %min_coverage)
    plt.plot(plot_dicts[min_coverage]["alpha_val_list"], plot_dicts[min_coverage]["train_accs_list"], marker='x', label='train acc')
    #plt.plot(plot_dicts[min_coverage]["alpha_val_list"], plot_dicts[min_coverage]["test_accs_list"], marker='x', label='test acc')
    plt.show()

if all_opt:
    print("All experiments completed to optimality.")