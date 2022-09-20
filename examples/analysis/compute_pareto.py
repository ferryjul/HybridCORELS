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


# parser initialization
parser = argparse.ArgumentParser(description='Analysis of the results')
parser.add_argument('--dataset', type=str, default='compas', help='adult, compas')
parser.add_argument('--method', type=str, default='hyrs', help='hybrid model. Options: crl, hyrs, hycorels')



def compute_front(df):#, output_file):

    plt.figure()
    plt.scatter(df['transparency'], df['accuracy'])

    error      = 1.0 - df['accuracy']
    uncoverage = 1.0 - df['transparency']

    pareto_input = [[err, unc] for (err, unc) in zip(error, uncoverage)]
    pareto_input = np.array(pareto_input)
    # Pareto optimal points among the non-black box points
    msk = is_pareto_efficient(pareto_input)


    df = pd.DataFrame()          
    # Be sure to add the black box as well as the pareto optima
    df['accuracy']     = [1.0-error[0]] + [1.0 - error[i] for i in range(len(error)) if msk[i]]
    df['transparency'] =  [1-uncoverage[0]] + [1-uncoverage[i]  for i in range(len(error)) if msk[i]]

    sorted_idx = np.argsort(df['transparency'])
    plt.plot(df['transparency'].iloc[sorted_idx], df['accuracy'].iloc[sorted_idx], "ko-")
    plt.xlabel('Coverage')
    plt.ylabel('Accuracy')
    plt.ylim(0.6, 0.7)
    plt.show()
    #df.to_csv(output_file, encoding='utf-8', index=False)



def main():
    args = parser.parse_args()
    dataset = args.dataset
    method = args.method

    # #save direcory
    # save_dir = "./results/pareto/{}".format(method)
    
    # os.makedirs(save_dir, exist_ok=True)

    input_file = f"../results/acc_cov/{dataset}_{method}.csv"
    df = pd.read_csv(input_file)

    #output_file_train = '{}/{}.csv'.format(save_dir, dataset)
    compute_front(df)



if __name__ == '__main__':
    sys.exit(main())


