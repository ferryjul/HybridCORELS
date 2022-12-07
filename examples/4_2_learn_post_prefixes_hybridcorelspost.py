import argparse
import numpy as np
from local_config import ccanada_expes
import os
import argparse
from exp_utils import get_data, to_df
from black_box_models import BlackBox
import warnings
warnings.filterwarnings("ignore")

if ccanada_expes:
    from mpi4py import MPI
    verbositylist=[]
    verbosity = False
    models_folder = "models"
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
else:
    verbosity = True
    verbositylist=["progress", "hybrid"]
    models_folder = "models_part_4"
    size = 1
    rank = 0

parser = argparse.ArgumentParser()
parser.add_argument("--expe_id", type=int, help='dataset-split-bbtype combination', default=0)
args = parser.parse_args()
expe_id = args.expe_id

# (Slurm grid) Select dataset and coverage constraint based on expe ID (fixed dataset & cov => ~ same running time, optimizes MPI CPU use)
# 45 slurm tasks
bbox_types = ['random_forest', 'ada_boost', 'gradient_boost']
rseeds = [0, 1, 2, 3, 4]
datasets = ["compas", "adult", "acs_employ"]

expes_list = []
for d in datasets: # 3
    for r in rseeds: # 5
        for b in bbox_types: # 3
            expes_list.append([d,r,b])
dataset_name, rseed, bbox_type = expes_list[expe_id]

# Get the data
if verbosity:
    print("Loading the data...")
X, y, features, prediction = get_data(dataset_name, {"train" : 0.6, "valid" : 0.20, "test" : 0.20}, 
                                random_state_param=rseed)

X_val, y_val = X['valid'], y['valid']
X_train, y_train = X['train'], y['train']
X_test, y_test =  X['test'], y['test']

# Retrieve the BB
model_path = f"{models_folder}/{dataset_name}_{bbox_type}_{rseed}.pickle"
assert(os.path.exists(model_path))
if verbosity:
    print("Loading the Black Box")

bbox = BlackBox(bb_type=bbox_type).load(model_path)

# Interpretable part training params
time_limit = 3600 # seconds
interpr_mem = 8000 # MB
method = "HybridCORELSPost"
from HybridCORELS import *
# Train the interpretable part (HybridCORELSPost)
# (MPI grid)
# 324 MPI ranks
min_coverageList = [] # Will contain 12 values
cov = 0.1
while cov < 1.0:
    min_coverageList.append(cov)
    cov += 0.1
    cov = round(cov, 2)
min_coverageList.extend([0.925, 0.95, 0.975])
if verbosity:
    print("List of coverage constraints: ", min_coverageList, "(len = %d)" %len(min_coverageList))

min_support_list = [0.01, 0.05, 0.1]  # Min Supports of Rules in Search Space
cList = [1e-2, 1e-3, 1e-4] # Regularisation CORELS
policies = ['objective', 'lower_bound', 'bfs'] # Priority Queue Criterion

paramsList = []
for p in policies:
    for c in cList:
        for min_support_val in min_support_list:
            for min_cov in min_coverageList:
                paramsList.append([p, c, min_support_val, min_cov])
if verbosity:
    print("MPI # combinations of params: ", len(paramsList))
worker_params = paramsList[rank]
policy = worker_params[0]
cValue = worker_params[1]
min_support_param = worker_params[2]
min_coverage = worker_params[3]

n_iter_param = 10**9
corels_params = {'policy':policy, 'max_card':1, 'c':cValue, 'n_iter':n_iter_param, 'min_support':min_support_param, 'verbosity':verbositylist} #"progress"

beta_value = min([ (1 / X_train.shape[0]) / 2, cValue / 2]) # small enough to only break ties
if verbosity:
    print("beta = ", beta_value)

hyb_model = HybridCORELSPostClassifier(black_box_classifier=bbox, beta=beta_value, min_coverage=min_coverage, 
                                       bb_pretrained=True, **corels_params)
# Train the hybrid model
hyb_model.fit(X_train, y_train, time_limit=time_limit, features=features, prediction_name=prediction, memory_limit=interpr_mem)

# Valid performance
yhat, covered_index = hyb_model.predict_with_type(X_val)
overall_acc_v = np.mean(yhat == y_val)
rule_coverage_v = np.sum(covered_index) / len(covered_index)
interpr_indices = np.where(covered_index == 1)
interpr_accuracy_v = np.mean(yhat[interpr_indices] == y_val[interpr_indices])

# Test performance
yhat, covered_index = hyb_model.predict_with_type(X_test)
overall_acc_t =  np.mean(yhat == y_test)
rule_coverage_t = np.sum(covered_index) / len(covered_index)
interpr_indices = np.where(covered_index == 1)
interpr_accuracy_t = np.mean(yhat[interpr_indices] == y_test[interpr_indices])

# Train performance
yhat, covered_index = hyb_model.predict_with_type(X_train)
overall_acc_train =  np.mean(yhat == y_train)
rule_coverage_train = np.sum(covered_index) / len(covered_index)
interpr_indices = np.where(covered_index == 1)
interpr_accuracy_train = np.mean(yhat[interpr_indices] == y_train[interpr_indices])

# String description of the model
descr = hyb_model.__str__()
status = hyb_model.get_status()
sparsity = hyb_model.get_sparsity()

# Result for one MPI runner
res = [[beta_value, min_coverage, policy, min_support_param, cValue, status, overall_acc_train, interpr_accuracy_train, rule_coverage_train, overall_acc_v, interpr_accuracy_v, rule_coverage_v, overall_acc_t, interpr_accuracy_t, rule_coverage_t, descr, sparsity]]

# Gather the results for the 5 folds on process 0
if ccanada_expes:
    res = comm.gather(res, root=0)

if rank == 0 or not ccanada_expes:
    # save results
    fileName = './results/results_4_2_post_%s_%s_%d_%s.csv' %(method, dataset_name, rseed, bbox_type) #_proportions
    import csv
    with open(fileName, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['beta', 'min_coverage', 'policy', 'min support', 'lambda', 'search status', 'accuracy_train', 'prefix_accuracy_train', 'transparency_train', 'accuracy_valid', 'prefix_accuracy_valid', 'transparency_valid', 'accuracy_test', 'prefix_accuracy_test', 'transparency_test', 'model', 'prefix length'])
        for i in range(len(res)):
            if ccanada_expes:
                for j in range(len(res[i])):
                    csv_writer.writerow(res[i][j])
            else:
                csv_writer.writerow(res[i])

