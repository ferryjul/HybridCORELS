import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif', 'sans-serif':['Computer Modern Sans Serif'], 'size':20})
# rc('text', usetex=True)

import argparse
import os

from ndf import is_pareto_efficient



# def get_pareto_fronts(dfs):

#     dfs_valid_opt = []
#     dfs_test_opt = []
#     for df in dfs:
#         # Compute pareto front on validation
#         pareto_input = df.iloc[:, :2].to_numpy()
#         # Pareto optimal points among the non-black box points
#         mask = is_pareto_efficient(-pareto_input)
#         # Make sure we include the black box
#         mask[0] = True

#         df_valid_copy = df.iloc[mask, :].sort_values(by=['transparency_valid'])
#         df_test_copy  = df.iloc[mask, 2:].sort_values(by=['transparency_test'])   
        
#         # Re-compute the pareto front for df_test_copy
#         mask = is_pareto_efficient(-df_test_copy.to_numpy())
#         # Make sure we include the black box
#         mask[0] = True
#         df_test_copy  = df_test_copy.iloc[mask]

#         # Save results
#         dfs_valid_opt.append(df_valid_copy)
#         dfs_test_opt.append(df_test_copy)
    
#     return dfs_valid_opt, dfs_test_opt



# def compute_fronts(dfs, methods, args):#, output_file):

#     dfs_valid_opt, dfs_test_opt = get_pareto_fronts(dfs)
    
#     colors = ['blue', 'orange', 'green']
#     min = np.min([df['accuracy_valid'].min() for df in dfs_valid_opt])
#     max = np.max([df['accuracy_valid'].max() for df in dfs_valid_opt])
#     plt.figure()
#     for i in range(len(methods)):
#         plt.scatter(dfs[i]['transparency_valid'], dfs[i]['accuracy_valid'], alpha=0.2, c=colors[i])
#         plt.plot(dfs_valid_opt[i]['transparency_valid'], dfs_valid_opt[i]['accuracy_valid'], "o-", color=colors[i])
#     plt.xlabel('Coverage')
#     plt.ylabel('Accuracy')
#     plt.ylim(min-0.005, max+0.005)
#     plt.savefig(f"./images/acc_cov_valid_{args.dataset}_{args.bbox}.pdf", bbox_inches='tight')

#     plt.figure()
#     for i, method in enumerate(methods):
#         plt.scatter(dfs_valid_opt[i]['transparency_test'], dfs_valid_opt[i]['accuracy_test'], alpha=0.5, c=colors[i])
#         plt.plot(dfs_test_opt[i]['transparency_test'], dfs_test_opt[i]['accuracy_test'],"o-", color=colors[i], label=method)
#     legend_labels = plt.gca().get_legend_handles_labels()
#     plt.xlabel('Coverage')
#     plt.ylabel('Accuracy')
#     plt.savefig(f"./images/acc_cov_test_{args.dataset}_{args.bbox}.pdf", bbox_inches='tight')
#     # plt.ylim(0.6, 0.7)


#     # Plot the legend separately
#     if not os.path.exists("./images/acc_cov_test_legend.pdf"):
#         fig_leg = plt.figure(figsize=(5, 1))
#         ax_leg = fig_leg.add_subplot(111)
#         # Add the legend from the previous axes
#         ax_leg.legend(*legend_labels, loc='center', ncol=3)
#         # Hide the axes frame and the x/y labels
#         ax_leg.axis('off')
#         plt.savefig("./images/acc_cov_test_legend.pdf", bbox_inches='tight', pad_inches=0)




def get_pareto_front(df, bb_perf):

    # Compute pareto front on validation
    pareto_input = df[['transparency_test', 'accuracy_test']].to_numpy()
    # Add the black box
    pareto_input = np.vstack(pareto_input)
    # Pareto optimal points among the non-black box points
    mask = is_pareto_efficient(-pareto_input)
    pareto_front = pareto_input[mask]

    # Sort by transparency
    pareto_front = pareto_front[np.argsort(pareto_front[:, 0])]
    # Add the black box
    pareto_front =  np.vstack((np.array([[0, bb_perf]]), pareto_front))
    return pareto_front


def plot_fronts(hybrid_df, bb_post_df, args):

    plt.figure()
    colors = {'HyRS' : 'blue', 'CRL' : 'orange', 'HybridCORELSPost' : 'green',
              'HybridCORELSPre' : 'red'}
    # Linespace for the pareto curves
    coverage_span = np.linspace(0, 1, 100)
    for method, group_method in hybrid_df.groupby('method'):
        # Store one pareto curve for each fold
        pareto_curves = []
        for fold, group in group_method.groupby('fold'):
            # MultiIndex to get the BB perf
            bb_perf = bb_post_df[(fold, args.bbox)]
            # Compute the pareto front for this specific fold
            pareto_points = get_pareto_front(group, bb_perf)
            # Linear interpolation of the pareto curve
            pareto_curves.append(interp1d(pareto_points[:, 0], 
                                          pareto_points[:, 1], 
                                          fill_value="extrapolate")(coverage_span))
        pareto_curves = np.vstack(pareto_curves) # (5, 100)
        average_pareto = np.mean(pareto_curves, axis=0)
        std_pareto = np.std(pareto_curves, axis=0)

        plt.plot(coverage_span, average_pareto, color=colors[method], label=method, linewidth=3)
        plt.fill_between(coverage_span, average_pareto-std_pareto, 
                                        average_pareto+std_pareto, color=colors[method], alpha=0.09)
    
    # min = np.min([df['accuracy_valid'].min() for df in dfs_valid_opt])
    # max = np.max([df['accuracy_valid'].max() for df in dfs_valid_opt])
    # plt.figure()
    # for i, method in enumerate(methods):
    #     plt.scatter(dfs_valid_opt[i]['transparency_test'], dfs_valid_opt[i]['accuracy_test'], alpha=0.5, c=colors[i])
    #     plt.plot(dfs_test_opt[i]['transparency_test'], dfs_test_opt[i]['accuracy_test'],"o-", color=colors[i], label=method)
    legend_labels = plt.gca().get_legend_handles_labels()
    plt.xlabel('Coverage')
    plt.ylabel('Accuracy')
    plt.xlim(0, 1)
    plt.ylim(args.min_acc, args.max_acc)
    # plt.legend()
    plt.savefig(f"./images/acc_cov_test_{args.dataset}_{args.bbox}.pdf", bbox_inches='tight')


    # Plot the legend separately
    if not os.path.exists("./images/acc_cov_test_legend.pdf"):
        fig_leg = plt.figure(figsize=(5, 1))
        ax_leg = fig_leg.add_subplot(111)
        # Add the legend from the previous axes
        ax_leg.legend(*legend_labels, loc='center', ncol=4)
        # Hide the axes frame and the x/y labels
        ax_leg.axis('off')
        plt.savefig("./images/acc_cov_test_legend.pdf", bbox_inches='tight', pad_inches=0)


def main():

    # Parser initialization
    parser = argparse.ArgumentParser(description='Analysis of the results')
    parser.add_argument('--dataset', type=str, default='adult', help='adult, compas')
    parser.add_argument("--bbox", type=str, help='Black box. Options: random_forest, ada_boost, gradient_boost', default='random_forest')
    parser.add_argument("--min_acc", type=float, help='Minimum value of the accuracy', default=None)
    parser.add_argument("--max_acc", type=float, help='Maximum value of the accuracy', default=None)
    args = parser.parse_args()

    methods = ["HyRS", "CRL", "HybridCORELSPost"]
    columns_to_keep = ['model', 'fold', 'method',
                       'accuracy_test', 'transparency_test']
    folds = list(range(5))


    #### Load Intermediary Results Post ####
    path = os.path.join("..", "results", "acc_cov")
    input_file = os.path.join(path, "Intermediary", f"results_4_1_learn_post_black_boxes_{args.dataset}.csv")
    bb_post_df  = pd.read_csv(input_file)[["Seed", "Black Box", "Test accuracy"]]
    # Get a MultiIndex ('Seed', 'Black Box')
    bb_post_df = bb_post_df.groupby(["Seed", 'Black Box'])["Test accuracy"].mean()


    #### Load Final Results Post ####
    hybrid_df = []
    for method in methods:
        for fold in folds:
            input_file = os.path.join(path, f"Final",   
                                f"results_4_2_post_{method}_{args.dataset}_{fold}_{args.bbox}.csv")
            df = pd.read_csv(input_file)
            print(df.shape)
            df['fold'] = fold
            df['method'] = method
            # First row is the black box
            if df.iloc[0, 0] == "na":
                hybrid_df.append(df[columns_to_keep].iloc[1:])
            else:
                hybrid_df.append(df[columns_to_keep])


    #### Load Final Results Pre ####
    input_file = os.path.join(path, "Final", f"results_4_2_pre_HybridCORELSPre_{args.dataset}.csv")
    df  = pd.read_csv(input_file)
    # Only get results for the specified black-box
    df = df[df['Black-box type']==args.bbox]
    df = df.rename(columns={"Seed":"fold"})
    df["method"] = "HybridCORELSPre"
    hybrid_df.append(df[columns_to_keep])

    # Concatenate all hybrid model results
    hybrid_df = pd.concat(hybrid_df, axis=0, ignore_index=True)

    # Plot the pareto front
    plot_fronts(hybrid_df, bb_post_df, args)



if __name__ == '__main__':
    main()


