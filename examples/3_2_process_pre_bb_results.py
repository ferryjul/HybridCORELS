import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

matplotlib.rcParams.update({'font.size': 12.0}) # default = 10.0

accuracy_train_color = 'navy'
accuracy_test_color = 'darkgreen'
save_dir = "./results_graham"
datasets = ["adult", "compas", "acs_employ"]
show_lb_ub = False
show_std = True
bbdict = {'random_forest':'Random Forest', 'ada_boost':'AdaBoost', 'gradient_boost':'Gradient Boost'}
min_support_txt_dict = {0.25:'Low Transparency (0.25)', 0.50:'Medium Transparency (0.50)', 0.75:'High Transparency (0.75)', 0.85:'High Transparency (0.85)', 0.95:'Very High Transparency (0.95)'}
alpha_benef_rates = {}
alpha_benef_gap = {}
legendFig = plt.figure("Legend plot")
legend_elements = []
legend_elements.append(Line2D([0], [0], marker='x', color=accuracy_train_color, lw=1, label="Black-box Accuracy (train)")) # linestyle = 'None',
legend_elements.append(Line2D([0], [0], marker='x', color=accuracy_test_color, lw=1, label="Black-box Accuracy (test)")) # linestyle = 'None',
legend_elements.append(Line2D([0], [0], marker='x', c='r', lw=1, label='Best (test)'))
legendFig.legend(handles=legend_elements, loc='center', ncol=3)
legendFig.savefig('./figures/black_boxes_legend.pdf', bbox_inches='tight')
plt.clf()
best_gap = -1
best_alphas_list = {}
for dataset in datasets:
    best_alphas_list[dataset] = {}
    fileName = '%s/results_HybridCORELSPre_wBB_%s.csv' %(save_dir, dataset)
    n_folds = 5
    #colors = {'random_forest':'orange', 'ada_boost':'blue', 'gradient_boost':'green'}
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
            plot_dicts[bbtype][min_coverage]["train_accs_list_std"] = []
            plot_dicts[bbtype][min_coverage]["train_accs_ub_list"] = []
            plot_dicts[bbtype][min_coverage]["train_accs_lb_list"] = []
            plot_dicts[bbtype][min_coverage]["test_accs_list"] = []
            plot_dicts[bbtype][min_coverage]["test_accs_list_std"] = []
            for alpha_val in results_dict[bbtype][min_coverage].keys():
                plot_dicts[bbtype][min_coverage]["alpha_val_list"].append(alpha_val)
                assert(len(results_dict[bbtype][min_coverage][alpha_val]["train_acc_bb"]) ==  n_folds)
                assert(len(results_dict[bbtype][min_coverage][alpha_val]["train_acc_ub"]) ==  n_folds)
                assert(len(results_dict[bbtype][min_coverage][alpha_val]["train_acc_lb"]) ==  n_folds)
                assert(len(results_dict[bbtype][min_coverage][alpha_val]["test_acc_bb"]) ==  n_folds)

                # push average on the n folds
                plot_dicts[bbtype][min_coverage]["train_accs_list"].append(np.average(results_dict[bbtype][min_coverage][alpha_val]["train_acc_bb"]))
                plot_dicts[bbtype][min_coverage]["train_accs_list_std"].append(np.std(results_dict[bbtype][min_coverage][alpha_val]["train_acc_bb"]))
                plot_dicts[bbtype][min_coverage]["train_accs_ub_list"].append(np.average(results_dict[bbtype][min_coverage][alpha_val]["train_acc_ub"]))
                plot_dicts[bbtype][min_coverage]["train_accs_lb_list"].append(np.average(results_dict[bbtype][min_coverage][alpha_val]["train_acc_lb"]))
                plot_dicts[bbtype][min_coverage]["test_accs_list"].append(np.average(results_dict[bbtype][min_coverage][alpha_val]["test_acc_bb"]))
                plot_dicts[bbtype][min_coverage]["test_accs_list_std"].append(np.std(results_dict[bbtype][min_coverage][alpha_val]["test_acc_bb"]))


    #print(plot_dicts)
    

    for min_coverage in plot_dicts['random_forest'].keys():
        

        for bbtype in results_dict.keys():
            if not bbtype in best_alphas_list[dataset].keys():
                best_alphas_list[dataset][bbtype] = []
            # Cast to np array to allow std + or -
            plot_dicts[bbtype][min_coverage]["train_accs_list"] = np.asarray(plot_dicts[bbtype][min_coverage]["train_accs_list"])
            plot_dicts[bbtype][min_coverage]["train_accs_list_std"] = np.asarray(plot_dicts[bbtype][min_coverage]["train_accs_list_std"])
            plot_dicts[bbtype][min_coverage]["test_accs_list"] = np.asarray(plot_dicts[bbtype][min_coverage]["test_accs_list"])
            plot_dicts[bbtype][min_coverage]["test_accs_list_std"] = np.asarray(plot_dicts[bbtype][min_coverage]["test_accs_list_std"])

            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

            #plt.title("Min Coverage = %.2f, %s" %(min_coverage, bbtype))
            #plt.title("%s black-box" %bbdict[bbtype])
            plt.title("%s" %min_support_txt_dict[min_coverage])
            plt.plot(plot_dicts[bbtype][min_coverage]["alpha_val_list"], plot_dicts[bbtype][min_coverage]["train_accs_list"], c=accuracy_train_color, marker='x', label="Train (average and std)") #, label='%s' %bbtype) #colors[bbtype]
            plt.plot(plot_dicts[bbtype][min_coverage]["alpha_val_list"], plot_dicts[bbtype][min_coverage]["test_accs_list"], c=accuracy_test_color, marker='x', label="Test (average and std)", zorder=-1)#, label='%s test acc' %bbtype)
            
            if show_std:
                plt.fill_between(plot_dicts[bbtype][min_coverage]["alpha_val_list"], plot_dicts[bbtype][min_coverage]["train_accs_list"] - plot_dicts[bbtype][min_coverage]["train_accs_list_std"], plot_dicts[bbtype][min_coverage]["train_accs_list"] + plot_dicts[bbtype][min_coverage]["train_accs_list_std"], color=accuracy_train_color, alpha=0.2)
                plt.fill_between(plot_dicts[bbtype][min_coverage]["alpha_val_list"], plot_dicts[bbtype][min_coverage]["test_accs_list"] - plot_dicts[bbtype][min_coverage]["test_accs_list_std"], plot_dicts[bbtype][min_coverage]["test_accs_list"] + plot_dicts[bbtype][min_coverage]["test_accs_list_std"], color=accuracy_test_color, alpha=0.2)
            
            if show_lb_ub:
                plt.plot(plot_dicts[bbtype][min_coverage]["alpha_val_list"], plot_dicts[bbtype][min_coverage]["train_accs_ub_list"], ':', c='black', marker='x')
                plt.plot(plot_dicts[bbtype][min_coverage]["alpha_val_list"], plot_dicts[bbtype][min_coverage]["train_accs_lb_list"], ':', c='black', marker='x')

            # Find best value
            best_test_value = np.max(plot_dicts[bbtype][min_coverage]["test_accs_list"])
            
            if bbtype == "random_forest":
                if best_gap < best_test_value-plot_dicts[bbtype][min_coverage]["test_accs_list"][0]:
                    best_gap = best_test_value-plot_dicts[bbtype][min_coverage]["test_accs_list"][0]
                    best_exp = "%s_%s" %(dataset, min_coverage)
                    old_perf = plot_dicts[bbtype][min_coverage]["test_accs_list"][0]
                    new_perf = best_test_value
            best_test_index = np.argmax(plot_dicts[bbtype][min_coverage]["test_accs_list"])
            best_alpha_value = plot_dicts[bbtype][min_coverage]["alpha_val_list"][best_test_index]
            worst_acc = min([np.min(plot_dicts[bbtype][min_coverage]["train_accs_list"]), np.min(plot_dicts[bbtype][min_coverage]["test_accs_list"])])
            
            best_alphas_list[dataset][bbtype].append(best_alpha_value)

            # Coloration du marker
            plt.scatter([best_alpha_value], [best_test_value], marker='x', c='r', zorder=1, label='Best (Test)') #facecolors='none', edgecolors='r', s=100, label="Best test value") #')#, label='%s test acc' %bbtype)

            # Trait pointillé jusqu'à l'axe des x
            plt.autoscale(False) # To avoid that the scatter changes limits
            plt.plot([best_alpha_value, best_alpha_value], [0, plot_dicts[bbtype][min_coverage]["test_accs_list"][best_test_index]], '--', c='red')#, label='%s test acc' %bbtype)

            plt.xlabel("Specialization Coefficient $\\alpha$")
            plt.ylabel("Black-Box Accuracy")
            #plt.legend(loc='best')
            saveName = "figures/expes_pre_min_coverage_%.2f_%s_%s.png" %(min_coverage, dataset, bbtype)
            plt.savefig(saveName, bbox_inches='tight')
            saveName = "figures/expes_pre_min_coverage_%.2f_%s_%s.pdf" %(min_coverage, dataset, bbtype)
            plt.savefig(saveName, bbox_inches='tight')
            plt.clf()
            #if min_coverage in [0.25, 0.50, 0.75, 0.85, 0.95]:
            if not bbtype in alpha_benef_rates.keys():
                alpha_benef_rates[bbtype] = {}
                alpha_benef_gap[bbtype] = {}
            for alpha_value in plot_dicts[bbtype][min_coverage]["alpha_val_list"]:
                if not (alpha_value in alpha_benef_rates[bbtype].keys()):
                    alpha_benef_rates[bbtype][alpha_value] = []
                    alpha_benef_gap[bbtype][alpha_value] = []
                if plot_dicts[bbtype][min_coverage]["test_accs_list"][alpha_value] > plot_dicts[bbtype][min_coverage]["test_accs_list"][0]:
                    alpha_benef_rates[bbtype][alpha_value].append(1)
                elif plot_dicts[bbtype][min_coverage]["test_accs_list"][alpha_value] == plot_dicts[bbtype][min_coverage]["test_accs_list"][0]:
                    alpha_benef_rates[bbtype][alpha_value].append(0)
                else:
                    alpha_benef_rates[bbtype][alpha_value].append(-1)
                alpha_benef_gap[bbtype][alpha_value].append(plot_dicts[bbtype][min_coverage]["test_accs_list"][alpha_value] - plot_dicts[bbtype][min_coverage]["test_accs_list"][0])
            '''plt.title("%s" %min_support_txt_dict[min_coverage])
            plt.plot(plot_dicts[bbtype][min_coverage]["alpha_val_list"], plot_dicts[bbtype][min_coverage]["test_accs_list"]-plot_dicts[bbtype][min_coverage]["test_accs_list"][0])
            saveName = "figures/test_gap_expes_pre_min_coverage_%.2f_%s_%s.png" %(min_coverage, dataset, bbtype)
            plt.savefig(saveName, bbox_inches='tight')
            plt.clf()'''
    if all_opt:
        print("Dataset %s: All experiments completed to optimality." %dataset)
    else:
        print("Dataset %s: Some experiments did not reach optimality." %dataset)

print("best_alphas_list = ", best_alphas_list)
best_alphas_average = {}

for dataset in datasets:
    best_alphas_average[dataset] = []
    for i in range(5):
        best_alphas_average[dataset].append(np.average([best_alphas_list[dataset][bbtype][i] for bbtype in best_alphas_list[dataset].keys()]))

best_alphas_average['all'] = []
for i in range(5):
    best_alphas_average['all'].append(np.average( [best_alphas_average[dataset][i] for dataset in datasets] ))

plt.plot(plot_dicts['random_forest'].keys(), best_alphas_average['all'], label=dataset)
saveName = "figures/test_better_alpha.png"
plt.savefig(saveName, bbox_inches='tight')

for bbtype in results_dict.keys():
    #assert(len(alpha_benef_rates[bbtype][0]) == 15)
    
    for alpha_value in plot_dicts[bbtype][min_coverage]["alpha_val_list"]:
        alpha_benef_rates[bbtype][alpha_value] = np.unique(alpha_benef_rates[bbtype][alpha_value], return_counts=True)
        if alpha_value >= 1:
            if ( alpha_benef_rates[bbtype][alpha_value][0].size == 1):
                alpha_benef_rates[bbtype][alpha_value] = alpha_benef_rates[bbtype][alpha_value][0][0]
            else:
                assert(alpha_benef_rates[bbtype][alpha_value][0][1] == 1)
                alpha_benef_rates[bbtype][alpha_value] = 100*alpha_benef_rates[bbtype][alpha_value][1][1]/np.sum(alpha_benef_rates[bbtype][alpha_value][1])
        alpha_benef_gap[bbtype][alpha_value] = np.average(alpha_benef_gap[bbtype][alpha_value])
    assert(alpha_benef_rates[bbtype][0][0].size == 1) # check baseline (no spe)
    alpha_benef_rates[bbtype].pop(0)
    assert(alpha_benef_gap[bbtype][0] == 0)
    alpha_benef_gap[bbtype].pop(0)
    print(alpha_benef_rates[bbtype], bbtype)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    fig,ax = plt.subplots()
    ax.plot(plot_dicts[bbtype][min_coverage]["alpha_val_list"][1:], alpha_benef_rates[bbtype].values(), c='orange')
    ax2=ax.twinx()
    ax2.plot(plot_dicts[bbtype][min_coverage]["alpha_val_list"][1:], alpha_benef_gap[bbtype].values(), c='red')
    plt.xlabel("Specialization Coefficient $\\alpha$")
    ax.set_ylabel("Improvement rate (%s)")
    ax2.set_ylabel("Absolute improvement")
    saveName = "figures/test_rate_better_expes_pre_bb_%s.png" %(bbtype)
    plt.savefig(saveName, bbox_inches='tight')
    saveName = "figures/test_rate_better_expes_pre_bb_%s.pdf" %(bbtype)
    plt.savefig(saveName, bbox_inches='tight')
    plt.clf()

all_bbs_improvements_rates = []
for alpha_value in plot_dicts[bbtype][min_coverage]["alpha_val_list"][1:]:
    all_bbs_improvements_rates.append(np.average([list(alpha_benef_rates[bbtype].values())[alpha_value-1] for bbtype in results_dict.keys()]))
print(all_bbs_improvements_rates)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.plot(plot_dicts[bbtype][min_coverage]["alpha_val_list"][1:], all_bbs_improvements_rates, c='orange')
plt.xlabel("Specialization Coefficient $\\alpha$")
plt.ylabel("Improvement rate (%s)")
saveName = "figures/test_rate_better_expes_pre_bb_all.png"
plt.savefig(saveName, bbox_inches='tight')
saveName = "figures/test_rate_better_expes_pre_bb_all.pdf"
plt.savefig(saveName, bbox_inches='tight')
plt.clf()

# best gap for rf bb
print(best_gap, best_exp, "%.4f -> %.4f" %(old_perf, new_perf))