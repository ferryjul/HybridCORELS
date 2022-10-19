import numpy as np
import pandas as pd 
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif', 'sans-serif':['Computer Modern Sans Serif'], 'size':20})
rc('text', usetex=True)

import argparse
import csv

from ndf import is_pareto_efficient



def get_pareto_fronts(dfs):

    dfs_valid_opt = []
    dfs_test_opt = []
    for df in dfs:
        # Compute pareto front on validation
        pareto_input = df.iloc[:, :2].to_numpy()
        # Pareto optimal points among the non-black box points
        mask = is_pareto_efficient(-pareto_input)
        # Make sure we include the black box
        mask[0] = True

        df_valid_copy = df.iloc[mask, :2].sort_values(by=['transparency_valid'])
        df_test_copy  = df.iloc[mask, 2:].sort_values(by=['transparency_test'])   
        
        # Re-compute the pareto front for df_test_copy
        mask = is_pareto_efficient(-df_test_copy.to_numpy())
        # Make sure we include the black box
        mask[0] = True
        df_test_copy  = df_test_copy.iloc[mask]

        # Save results
        dfs_valid_opt.append(df_valid_copy)
        dfs_test_opt.append(df_test_copy)
    
    return dfs_valid_opt, dfs_test_opt



def compute_fronts(dfs, methods):#, output_file):

    dfs_valid_opt, dfs_test_opt = get_pareto_fronts(dfs)
    
    plt.figure()
    for i, method in enumerate(methods):
        # plt.scatter(dfs[i]['transparency_valid'], dfs[i]['accuracy_valid'], alpha=0.05)
        plt.plot(dfs_valid_opt[i]['transparency_valid'], dfs_valid_opt[i]['accuracy_valid'], "o-", label=method)
    plt.xlabel('Coverage')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.ylim(0.6, 0.7)

    plt.figure()
    for i, method in enumerate(methods):
        plt.plot(dfs_test_opt[i]['transparency_test'], dfs_test_opt[i]['accuracy_test'], "o-", label=method)
    plt.xlabel('Coverage')
    plt.ylabel('Accuracy')
    # plt.ylim(0.6, 0.7)


    plt.show()
    #df.to_csv(output_file, encoding='utf-8', index=False)



def main():

    # Parser initialization
    parser = argparse.ArgumentParser(description='Analysis of the results')
    parser.add_argument('--dataset', type=str, default='compas', help='adult, compas')
    parser.add_argument("--bbox", type=str, help='Black box. Options: random_forest, ada_boost, gradient_boost', default='random_forest')
    # parser.add_argument('--method', type=str, default='hyrs', help='hybrid model. Options: crl, hyrs, hycorels')
    args = parser.parse_args()

    # #save direcory
    # save_dir = "./results/pareto/{}".format(method)
    
    # os.makedirs(save_dir, exist_ok=True)

    methods = ["hyrs", "crl"]
    dfs = []
    for method in methods:
        input_file = f"../results/acc_cov/{args.dataset}_{args.bbox}_{method}.csv"
        df = pd.read_csv(input_file)
        dfs.append(df)

    compute_fronts(dfs, methods)



if __name__ == '__main__':
    main()


