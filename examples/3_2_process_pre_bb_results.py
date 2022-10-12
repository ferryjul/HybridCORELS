import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

save_dir = "./results_graham"
datasets = ["adult", "compas"]
show_lb_ub = False

for dataset in datasets:
    fileName = '%s/results_HybridCORELSPre_wBB_%s.csv' %(save_dir, dataset)
    n_folds = 5
    colors = {'random_forest':'orange', 'ada_boost':'blue', 'gradient_boost':'green'}
    # header = ["fold_id", "min_coverage", "alpha_value", "train_acc", "train_coverage", "test_acc", "test_coverage", "search_status", "sparsity", "model"]
    all_opt = True
    results_df = pd.read_csv(fileName)
    results_dict = {}
    for index, row in results_df.iterrows():
        if True: #row["fold_id"] == 1:
            alpha_val = row["Alpha"]

            train_acc = row["Training accuracy (BB)"]
            train_acc_ub = row["Training accuracy (BB) UB"]
            test_acc = row["Test accuracy (BB)"]

            model_txt = row["Model"].replace(')','').split()
            assert(model_txt[-2] == 'pred')
            train_acc_lb = float(model_txt[-1])

            min_coverage = row["Min coverage"]
            seed = row["Seed"]
            bb_type = row["Black Box"]

            if row["Search status"] != "opt":
                all_opt = False

            if not bb_type in results_dict.keys():
                results_dict[bb_type] = {}

            if not (min_coverage in results_dict[bb_type].keys()):
                results_dict[bb_type][min_coverage] = {}

            if not (alpha_val in results_dict[bb_type][min_coverage].keys()):
                results_dict[bb_type][min_coverage][alpha_val] = {}
                results_dict[bb_type][min_coverage][alpha_val]["train_acc_bb"] = [train_acc]
                results_dict[bb_type][min_coverage][alpha_val]["train_acc_ub"] = [train_acc_ub]
                results_dict[bb_type][min_coverage][alpha_val]["train_acc_lb"] = [train_acc_lb]
                results_dict[bb_type][min_coverage][alpha_val]["test_acc_bb"] = [test_acc]
            else:
                results_dict[bb_type][min_coverage][alpha_val]["train_acc_bb"].append(train_acc)
                results_dict[bb_type][min_coverage][alpha_val]["train_acc_ub"].append(train_acc_ub)
                results_dict[bb_type][min_coverage][alpha_val]["train_acc_lb"].append(train_acc_lb)
                results_dict[bb_type][min_coverage][alpha_val]["test_acc_bb"].append(test_acc)

    plot_dicts = {}


    for bbtype in results_dict.keys():
        plot_dicts[bbtype] = {}
        for min_coverage in results_dict[bbtype].keys():
            plot_dicts[bbtype][min_coverage] = {}
            plot_dicts[bbtype][min_coverage]["alpha_val_list"] = []
            plot_dicts[bbtype][min_coverage]["train_accs_list"] = []
            plot_dicts[bbtype][min_coverage]["train_accs_ub_list"] = []
            plot_dicts[bbtype][min_coverage]["train_accs_lb_list"] = []
            plot_dicts[bbtype][min_coverage]["test_accs_list"] = []
            for alpha_val in results_dict[bbtype][min_coverage].keys():
                plot_dicts[bbtype][min_coverage]["alpha_val_list"].append(alpha_val)
                assert(len(results_dict[bbtype][min_coverage][alpha_val]["train_acc_bb"]) ==  n_folds)
                assert(len(results_dict[bbtype][min_coverage][alpha_val]["train_acc_ub"]) ==  n_folds)
                assert(len(results_dict[bbtype][min_coverage][alpha_val]["train_acc_lb"]) ==  n_folds)
                assert(len(results_dict[bbtype][min_coverage][alpha_val]["test_acc_bb"]) ==  n_folds)

                # push average on the n folds
                plot_dicts[bbtype][min_coverage]["train_accs_list"].append(np.average(results_dict[bbtype][min_coverage][alpha_val]["train_acc_bb"]))
                plot_dicts[bbtype][min_coverage]["train_accs_ub_list"].append(np.average(results_dict[bbtype][min_coverage][alpha_val]["train_acc_ub"]))
                plot_dicts[bbtype][min_coverage]["train_accs_lb_list"].append(np.average(results_dict[bbtype][min_coverage][alpha_val]["train_acc_lb"]))
                plot_dicts[bbtype][min_coverage]["test_accs_list"].append(np.average(results_dict[bbtype][min_coverage][alpha_val]["test_acc_bb"]))


    #print(plot_dicts)

    for min_coverage in plot_dicts['random_forest'].keys():
        for bbtype in results_dict.keys():
            plt.title("Min Coverage = %.2f, %s" %(min_coverage, bbtype))
            plt.plot(plot_dicts[bbtype][min_coverage]["alpha_val_list"], plot_dicts[bbtype][min_coverage]["train_accs_list"], c=colors[bbtype], marker='x', label='%s' %bbtype)
            plt.plot(plot_dicts[bbtype][min_coverage]["alpha_val_list"], plot_dicts[bbtype][min_coverage]["test_accs_list"], '--', c=colors[bbtype], marker='x')#, label='%s test acc' %bbtype)
            if show_lb_ub:
                plt.plot(plot_dicts[bbtype][min_coverage]["alpha_val_list"], plot_dicts[bbtype][min_coverage]["train_accs_ub_list"], ':', c='black', marker='x')
                plt.plot(plot_dicts[bbtype][min_coverage]["alpha_val_list"], plot_dicts[bbtype][min_coverage]["train_accs_lb_list"], ':', c='black', marker='x')

            # Find best value
            best_test_index = np.argmax(plot_dicts[bbtype][min_coverage]["test_accs_list"])
            best_alpha_value = plot_dicts[bbtype][min_coverage]["alpha_val_list"][best_test_index]
            worst_acc = min([np.min(plot_dicts[bbtype][min_coverage]["train_accs_list"]), np.min(plot_dicts[bbtype][min_coverage]["test_accs_list"])])
            plt.plot([best_alpha_value, best_alpha_value], [worst_acc, plot_dicts[bbtype][min_coverage]["test_accs_list"][best_test_index]], '--', c='red')#, label='%s test acc' %bbtype)

            plt.xlabel("Alpha")
            plt.ylabel("Overall Accuracy")
            plt.legend(loc='best')
            saveName = "figures/expes_pre_min_coverage_%.2f_%s_%s.png" %(min_coverage, dataset, bbtype)
            plt.savefig(saveName, bbox_inches='tight')
            plt.clf()

    if all_opt:
        print("Dataset %s: All experiments completed to optimality." %dataset)
    else:
        print("Dataset %s: Some experiments did not reach optimality." %dataset)