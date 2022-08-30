import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif', 'sans-serif':['Computer Modern Sans Serif'], 'size':20})
rc('text', usetex=True)

import numpy as np
import pandas as pd 
import os, sys

import argparse


# Parser initialization
parser = argparse.ArgumentParser(description='Analysis of the results')
parser.add_argument('--dataset', type=str, default='compas', help='adult, compas')



def main():
    args = parser.parse_args()
    dataset = args.dataset

    input_file = os.path.join("..", "results", "instability", f"{dataset}_HyRS.csv")
    df = pd.read_csv(input_file)

    plt.figure()
    beta = df['beta'] * np.random.uniform(0.95, 1.05, size=len(df))
    coverage = df['coverage'] + np.random.uniform(0, 0.025, size=len(df))
    plt.scatter(beta, coverage, alpha=0.5)
    #line = [np.min(df['beta']), np.max(df['beta'])]
    #plt.plot(line, line, 'k--')
    plt.xscale('log')
    plt.xlabel(r'$\beta$')
    plt.ylabel('Coverage')
    plt.savefig(f"instability_{dataset}.pdf", bbox_inches='tight')



if __name__ == '__main__':
    sys.exit(main())


