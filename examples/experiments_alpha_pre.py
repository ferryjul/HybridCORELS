import numpy as np
import pandas as pd
from HybridCORELS import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split #KFold
from local_config import ccanada_expes
from local_config import ccanada_expes


if ccanada_expes:
    from mpi4py import MPI

if ccanada_expes:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
else:
    rank = 231
    size = 1

#import warnings
#warnings.filterwarnings("ignore")

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type= str, help = 'Dataset name. Options: adult, compas', default = 'compas')

args = parser.parse_args()
dataset = args.dataset

# Prepare parameters (min coverage and alpha value)
min_coverages = np.concatenate([np.arange(0, 1.0, 0.05), np.arange(0.96, 0.99, 0.01)])
alpha_values = np.arange(0, 11, 1)
n_seeds = 10
train_proportion = 0.7

params_list = [] # 240
for min_coverage in min_coverages:
    for seed in range(n_seeds):
        params_list.append([min_coverage, seed])
#print("# params = ", len(params_list))
min_coverage, seed = params_list[rank]

# Set fixed parameters
random_state_param = 42
corels_params = {'policy':"lower_bound", 'max_card':1, 'c':0.001, 'n_iter':5*10**6, 'min_support':0.10, 'verbosity':[]} #"progress" # min_support was 0.05 for COMPAS experiments
#n_folds = 5

# Load and prepare data
X, y, features, prediction = load_from_csv("data/%s.csv" %dataset) 
#kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state_param)

# Generate train and test sets and run the algorithm on each fold
#fold_id = 1
res = []

#train_index, test_index in kf.split(X):
#X_train, X_test = X[train_index], X[test_index]
#y_train, y_test = y[train_index], y[test_index]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0 - train_proportion, 
                                                shuffle=True, random_state=seed)


bbox = RandomForestClassifier(random_state=seed, min_samples_leaf=10, max_depth=10)
beta_value = 1 / X_train.shape[0]

# Create and train the hybrid model
hyb_model = HybridCORELSPreClassifier(black_box_classifier=bbox, beta=beta_value, alpha=0, min_coverage=min_coverage, lb_mode='tight', **corels_params)
hyb_model.fit(X_train, y_train, features=features, prediction_name=prediction)

# Iterate over alpha values
for alpha_value in alpha_values:
    bbox = RandomForestClassifier(random_state=seed, min_samples_leaf=10, max_depth=10)
    hyb_model.refit_black_box(X_train, y_train, alpha_value, bbox)
    #print(alpha_value, hyb_model)
    # Evaluate it
    sparsity = hyb_model.get_sparsity()
    model_str = hyb_model.__str__()
    status = hyb_model.get_status()
    # Train set
    preds_train, preds_types_train = hyb_model.predict_with_type(X_train)
    accuracy_train = np.mean(preds_train == y_train)
    preds_types_counts_train = np.unique(preds_types_train, return_counts=True)
    index_one_train = np.where(preds_types_counts_train[0] == 1)
    cover_rate_train = preds_types_counts_train[1][index_one_train][0]/np.sum(preds_types_counts_train[1])
    # Test set
    preds_test, preds_types_test = hyb_model.predict_with_type(X_test)
    accuracy_test = np.mean(preds_test == y_test)
    preds_types_counts_test = np.unique(preds_types_test, return_counts=True)
    index_one_test = np.where(preds_types_counts_test[0] == 1)
    cover_rate_test = preds_types_counts_test[1][index_one_test][0]/np.sum(preds_types_counts_test[1])
    res.append([seed, min_coverage, alpha_value, accuracy_train, cover_rate_train, accuracy_test, cover_rate_test, status, sparsity, model_str])


    #fold_id += 1

# Gather the results for the 5 folds on process 0
if ccanada_expes:
    res = comm.gather(res, root=0)

if rank == 0 or not ccanada_expes:
    #save directory
    save_dir = "./results/expes_alpha_v1"
    os.makedirs(save_dir, exist_ok=True)
    # save results
    fileName = '%s/results_expes_alpha_%s.csv' %(save_dir, dataset)
    import csv
    with open(fileName, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ["fold_id", "min_coverage", "alpha_value", "train_acc", "train_coverage", "test_acc", "test_coverage", "search_status", "sparsity", "model"]
        csv_writer.writerow(header)
        for i in range(len(res)):
            if ccanada_expes:
                for j in range(len(res[i])):
                    csv_writer.writerow(res[i][j])
            else:
                csv_writer.writerow(res[i])