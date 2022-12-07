from exp_utils import get_data, to_df
import argparse
from local_config import ccanada_expes
from HybridCORELS import *
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import os

interpr_tout = 3600 #3600 #3600 # seconds
interpr_mem = 8000 # MB
n_iter_param = 10**9

if ccanada_expes: # one core performs operations for all values of alpha for all black-boxes
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    verbositylist = []
    verbosity = False
else: # for local debug, fixed parameters
    rank = 0
    size = 1
    verbositylist=["progress", "hybrid"]
    verbosity = True

parser = argparse.ArgumentParser()
parser.add_argument('--expe_id', type=int, default=0, help='dataset-min_coverage combination')
args = parser.parse_args()
expe_id = args.expe_id

# (Slurm grid) Select dataset and coverage constraint based on expe ID (fixed dataset & cov => ~ same running time, optimizes MPI CPU use)
# 36 slurm tasks
datasets = ["compas", "adult", "acs_employ"]
min_coverageList = [] # Will contain 20 values
cov = 0.1
while cov < 1.0:
    min_coverageList.append(cov)
    cov += 0.1
    cov = round(cov, 2)
min_coverageList.extend([0.925, 0.95, 0.975])
if verbosity:
    print("List of coverage constraints: ", min_coverageList, "(len = %d)" %len(min_coverageList))
expes_list = []
for dataset in datasets: # 3
    for min_cov in min_coverageList: # 12
        expes_list.append([dataset, min_cov])
dataset_name, min_coverage = expes_list[expe_id]

# (MPI grid)
# 135 MPI ranks
rseeds = [0, 1, 2, 3, 4]
min_support_list = [0.01, 0.05, 0.1]  # Min Supports of Rules in Search Space
cList = [1e-2, 1e-3, 1e-4] # Regularisation CORELS
policies = ['objective', 'lower_bound', 'bfs'] # Priority Queue Criterion
paramsList = []
for p in policies:
    for c in cList:
        for min_support_val in min_support_list:
            for dataset_seed in rseeds:
                paramsList.append([p, c, min_support_val, dataset_seed])
if verbosity:
    print("MPI # combinations of params: ", len(paramsList))
worker_params = paramsList[rank]
policy = worker_params[0]
cValue = worker_params[1]
min_support_param = worker_params[2]
rseed = worker_params[3]

X, y, features, prediction = get_data(dataset_name, {"train" : 0.6, "valid" : 0.20, "test" : 0.20}, 
                                random_state_param=rseed)
if verbosity:
    print(X['train'].shape, X['test'].shape)

X_val, y_val = X['valid'], y['valid']
X_train, y_train = X['train'], y['train']
X_test, y_test =  X['test'], y['test']

corels_params = {'policy':policy, 'max_card':1, 'c':cValue, 'n_iter':n_iter_param, 'min_support':min_support_param, 'verbosity':verbositylist} #"progress"

beta_value = min([ (1 / X_train.shape[0]) / 2, cValue / 2]) # small enough to only break ties
if verbosity:
    print("beta = ", beta_value)
    
alpha_value=0
bbox_type="fixed_rf"

# Create the hybrid model
bbox = RandomForestClassifier(random_state=42, min_samples_split=10, max_depth=5) #useless
hyb_model = HybridCORELSPreClassifier(black_box_classifier=bbox, beta=beta_value, alpha=alpha_value, min_coverage=min_coverage, lb_mode='tight', **corels_params)

# Train the hybrid model
hyb_model.fit(X_train, y_train, features=features, prediction_name=prediction, time_limit=interpr_tout, memory_limit=interpr_mem)

# Compute and save all metrics
model_path = "models/pre_prefix_%s_%d_%.3f_%.5f_%d_%.2f_%s.pickle" %(dataset_name, rseed, min_coverage, cValue, n_iter_param, min_support_param, policy)
hyb_model.save(model_path)

# Train set
preds_train, preds_types_train = hyb_model.predict_with_type(X_train)
train_acc = np.mean(preds_train == y_train)
preds_types_counts_train = np.unique(preds_types_train, return_counts=True)
index_one_train = np.where(preds_types_counts_train[0] == 1)
transparency_train = preds_types_counts_train[1][index_one_train][0]/np.sum(preds_types_counts_train[1])
bb_indices_train = np.where(preds_types_train == 0)
interpr_indices_train = np.where(preds_types_train == 1)
black_box_accuracy_train = np.mean(preds_train[bb_indices_train] == y_train[bb_indices_train])
interpr_accuracy_train = np.mean(preds_train[interpr_indices_train] == y_train[interpr_indices_train])

# Test set
preds_test, preds_types_test = hyb_model.predict_with_type(X_test)
test_acc = np.mean(preds_test == y_test)
preds_types_counts_test = np.unique(preds_types_test, return_counts=True)
index_one_test = np.where(preds_types_counts_test[0] == 1)
transparency_test = preds_types_counts_test[1][index_one_test][0]/np.sum(preds_types_counts_test[1])
bb_indices_test = np.where(preds_types_test == 0)
interpr_indices_test = np.where(preds_types_test == 1)
black_box_accuracy_test= np.mean(preds_test[bb_indices_test] == y_test[bb_indices_test])
interpr_accuracy_test = np.mean(preds_test[interpr_indices_test] == y_test[interpr_indices_test])

# Validation set
preds_val, preds_types_val = hyb_model.predict_with_type(X_val)
val_acc = np.mean(preds_val == y_val)
preds_types_counts_val = np.unique(preds_types_val, return_counts=True)
index_one_val = np.where(preds_types_counts_val[0] == 1)
transparency_val = preds_types_counts_val[1][index_one_val][0]/np.sum(preds_types_counts_val[1])
bb_indices_val = np.where(preds_types_val == 0)
interpr_indices_val = np.where(preds_types_val == 1)
black_box_accuracy_val= np.mean(preds_val[bb_indices_val] == y_val[bb_indices_val])
interpr_accuracy_val = np.mean(preds_val[interpr_indices_val] == y_val[interpr_indices_val])

status = hyb_model.get_status()
sparsity = hyb_model.get_sparsity()
print("Expe: dataset = %s, " %dataset_name, "rseed = %d" %rseed, "min_cov=%s" %min_coverage, "policy=%s" %policy, "c=%.3f" %cValue, " done.")

res = [[rseed, bbox_type, beta_value, alpha_value, policy, min_support_param, cValue, status, train_acc, val_acc, test_acc, interpr_accuracy_train, interpr_accuracy_val, interpr_accuracy_test, black_box_accuracy_train, black_box_accuracy_val, black_box_accuracy_test, transparency_train, transparency_val, transparency_test, str(hyb_model), sparsity]]

# Gather the results for the 5 folds on process 0
if ccanada_expes:
    res = comm.gather(res, root=0)

if rank == 0 or not ccanada_expes:
    # save results
    fileName = './results/results_4_1_learn_pre_prefixes_%s_%.3f.csv' %(dataset_name, min_coverage) #_proportions
    import csv
    with open(fileName, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Seed', 'Black Box', 'Beta', 'Alpha', 'Policy', 'Min support', 'Lambda', 'Search status', 'Training accuracy', 'Validation accuracy', 'Test accuracy', 'Training accuracy (prefix)', 'Validation accuracy (prefix)', 'Test accuracy (prefix)', 'Training accuracy (BB)', 'Validation accuracy (BB)','Test accuracy (BB)', 'Training transparency', 'Validation transparency', 'Test transparency', 'Model', 'Prefix length'])
        for i in range(len(res)):
            if ccanada_expes:
                for j in range(len(res[i])):
                    csv_writer.writerow(res[i][j])
            else:
                csv_writer.writerow(res[i])

