import csv
import numpy as np
import pandas as pd 
from six.moves import xrange
import os
import sys

import argparse
import csv

from ndf import is_pareto_efficient


# parser initialization
parser = argparse.ArgumentParser(description='Analysis of the results')
parser.add_argument('--dataset', type=str, default='compas', help='adult, compas')
parser.add_argument('--method', type=str, default='crl', help='hybrid model. Options: crl, hyrs, hycorels')



def compute_front(df, output_file):

    error      = 1.0 - df['accuracy']
    uncoverage = 1.0 - df['transparency']

    pareto_input = [[err, unc] for (err, unc) in zip(error, uncoverage)]
    pareto_input = np.array(pareto_input)
    msk = is_pareto_efficient(pareto_input)


    df = pd.DataFrame()          
    df['accuracy']      =  [1.0 - error[i] for i in xrange(len(error)) if msk[i]]
    df['transparency']  =  [1-uncoverage[i]  for i in xrange(len(error)) if msk[i]]


    df.to_csv(output_file, encoding='utf-8', index=False)



def main():
    args = parser.parse_args()
    dataset = args.dataset
    method = args.method
    

    #save direcory
    save_dir = "./results/pareto/{}".format(method)
    
    os.makedirs(save_dir, exist_ok=True)

    input_file = "../results/{}/{}.csv".format(method, dataset)
    df = pd.read_csv(input_file)

    output_file_train = '{}/{}.csv'.format(save_dir, dataset)
    compute_front(df, output_file_train)

    



if __name__ == '__main__':
    sys.exit(main())


