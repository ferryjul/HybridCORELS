from exp_utils import get_data, computeAccuracyUpperBound
import argparse
from local_config import ccanada_expes
from HybridCORELS import *
import numpy as np 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import pickle 
from black_box_models import BlackBox

time_limit = int(15.0 * 3600)
n_iters = 100
method = "HybridCORELSPre"

if ccanada_expes:
    from mpi4py import MPI

if ccanada_expes:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    verbositylist = []
    verbosity = False
    bbox_verbose = False
    models_folder = "models"
else: # for local debug, fixed parameters
    rank = 0
    size = 1
    verbositylist=["progress", "hybrid"]
    models_folder = "models_part_4"
    verbosity = True
    bbox_verbose = True

parser = argparse.ArgumentParser(description='Learn BB part of prelearnt prefixes')
parser.add_argument('--dataset', type=int, default=0, help='1 for adult, 0 for compas')
args = parser.parse_args()
datasets = ["compas", "adult", "acs_employ"]
dataset_name = datasets[args.dataset] # 3 slurm tasks

# MPI Grid
bbox_types = ['random_forest', 'ada_boost', 'gradient_boost']
rseeds = [0, 1, 2, 3, 4]
min_coverageList = [] # Will contain 12 values
cov = 0.1
while cov < 1.0:
    min_coverageList.append(cov)
    cov += 0.1
    cov = round(cov, 2)
min_coverageList.extend([0.925, 0.95, 0.975])

paramsList = [] # Will contain 180 values
for r in rseeds:
    for c in min_coverageList:
        for bbt in bbox_types:
            paramsList.append([r, c, bbt])

if verbosity:
    print("MPI # combinations of params: ", len(paramsList))

worker_params = paramsList[rank]
rseed = worker_params[0]
min_coverage = worker_params[1]
bbox_type = worker_params[2]

n_iter_param = 10**9
dict_save_folder = '4_1_pre_best_prefixes'

# 0) Load data
X, y, features, prediction = get_data(dataset_name, {"train" : 0.6, "valid" : 0.20, "test" : 0.20}, 
                                random_state_param=rseed)
if verbosity:
    print(X['train'].shape, X['test'].shape)

X_val, y_val = X['valid'], y['valid']
X_train, y_train = X['train'], y['train']
X_test, y_test =  X['test'], y['test']

# 1) Retrieve the hybrid model with best prefix
dict_name = '%s_%d_%.3f.pickle' %(dataset_name, rseed, min_coverage)

with open('%s/%s.pickle' %(dict_save_folder, dict_name), 'rb') as handle:
    best_params_dict = pickle.load(handle)
    policy = best_params_dict['policy']
    min_support_param = best_params_dict['min_support_param']
    cValue = best_params_dict['cValue']
    print("Dataset %s, Fold %d, Min Coverage %.2f, best params are :" %(dataset_name, rseed, min_coverage), best_params_dict)

beta_value = min([ (1 / X_train.shape[0]) / 2, cValue / 2]) # small enough to only break ties
alpha_value = 1 # best value based on experiments part 3 (pre-paradigm-specific)

model_path = "%s/pre_prefix_%s_%d_%.3f_%.5f_%d_%.2f_%s.pickle" %(models_folder, dataset_name, rseed, min_coverage, cValue, n_iter_param, min_support_param, policy)
hyb_model = HybridCORELSPreClassifier.load(model_path)

if verbosity:
    print("Loaded model: ", hyb_model)
    print("Status: ", hyb_model.get_status())

# Compute weights for validation set
val_preds, val_types = hyb_model.predict_with_type(X_val)
not_captured_indices = np.where(val_types == 0)
sample_weights_val = np.ones(y_val.shape)
sample_weights_val[not_captured_indices] = np.exp(alpha_value)
sample_weights_val /= np.sum(sample_weights_val)

bbox = BlackBox(bb_type=bbox_type, verbosity=bbox_verbose, random_state_value=rseed, n_iter=n_iters, time_limit=time_limit, X_val=X_val, y_val=y_val, sample_weights_val=sample_weights_val)

hyb_model.refit_black_box(X_train, y_train, alpha_value,  bbox)

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
res = [[rseed, bbox_type, beta_value, min_coverage, policy, min_support_param, cValue, status, overall_acc_train, interpr_accuracy_train, rule_coverage_train, overall_acc_v, interpr_accuracy_v, rule_coverage_v, overall_acc_t, interpr_accuracy_t, rule_coverage_t, descr, sparsity]]

# Gather the results for the 5 folds on process 0
if ccanada_expes:
    res = comm.gather(res, root=0)

if rank == 0 or not ccanada_expes:
    # save results
    fileName = './results/results_4_2_pre_%s_%s.csv' %(method, dataset_name) #_proportions
    import csv
    with open(fileName, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Seed', 'Black-box type', 'beta', 'min_coverage', 'policy', 'min support', 'lambda', 'search status', 'accuracy_train', 'prefix_accuracy_train', 'transparency_train', 'accuracy_valid', 'prefix_accuracy_valid', 'transparency_valid', 'accuracy_test', 'prefix_accuracy_test', 'transparency_test', 'model', 'prefix length'])
        for i in range(len(res)):
            if ccanada_expes:
                for j in range(len(res[i])):
                    csv_writer.writerow(res[i][j])
            else:
                csv_writer.writerow(res[i])
