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
    verbosity = False
    models_folder = "models"
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
else:
    verbosity = True
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
            print(len(expes_list), [d,r,b])
            expes_list.append([d,r,b])
            
dataset_name, rseed, bbox_type = expes_list[expe_id]

# Get the data
if verbosity:
    print("Loading the data...")
X, y, features, prediction = get_data(dataset_name, {"train" : 0.6, "valid" : 0.20, "test" : 0.20}, 
                                random_state_param=rseed)

df_X = to_df(X, features)

# Retrieve the BB
model_path = f"{models_folder}/{dataset_name}_{bbox_type}_{rseed}.pickle"
assert(os.path.exists(model_path))
if verbosity:
    print("Loading the Black Box")

bbox = BlackBox(bb_type=bbox_type).load(model_path)

# Black box performances
bbox_acc_train = np.mean(bbox.predict(df_X["train"]) == y["train"])
bbox_acc_v = np.mean(bbox.predict(df_X["valid"]) == y["valid"])    
bbox_acc_t = np.mean(bbox.predict(df_X["test"]) == y["test"])
if verbosity:
    print("bbox_acc_train =", bbox_acc_train)
    print("bbox_acc_v =", bbox_acc_v)
    print("bbox_acc_t =", bbox_acc_t)

# Interpretable part training params
time_limit = 30 # seconds
interpr_mem = 8000 # MB
method = "HyRS"
from HyRS import HybridRuleSetClassifier

# Train the interpretable part (HyRS)
# (MPI grid)
# 100 MPI ranks
alphas = np.logspace(-3, -2, 10) # Regul in [0, 1] Higher means smaller rulesets
betas = np.logspace(-3, 0, 10)   # Regul in [0, 1] Higher means larger Coverage

paramsList = []
for a in alphas:
    for b in betas:
        paramsList.append([a, b])
if verbosity:
    print("MPI # combinations of params: ", len(paramsList))
worker_params = paramsList[rank]
alpha = worker_params[0]
beta = worker_params[1]

hparams_hyrs = {
        "alpha" : alpha,
        "beta" : beta
    }

# Process results of HyRS
def results_hyrs(bbox, df_X, y, hparams_hyrs, time_limit, init_temperature):
    # Define a hybrid model
    hyb_model = HybridRuleSetClassifier(bbox, **hparams_hyrs)

    # Train the hybrid model
    hyb_model.fit(df_X["train"], y["train"], 10**7, random_state=12, T0=init_temperature, 
                                        premined_rules=True, time_limit=time_limit)

    # Train performance
    yhat, covered_index = hyb_model.predict_with_type(df_X["train"])
    overall_acc_train = np.mean(yhat == y["train"]) 
    rule_coverage_train = np.sum(covered_index) / len(covered_index)

    # Valid performance
    yhat, covered_index = hyb_model.predict_with_type(df_X["valid"])
    overall_acc_v = np.mean(yhat == y["valid"]) 
    rule_coverage_v = np.sum(covered_index) / len(covered_index)

    # Test performance
    yhat, covered_index = hyb_model.predict_with_type(df_X["test"])
    overall_acc_t = np.mean(yhat == y["test"]) 
    rule_coverage_t = np.sum(covered_index) / len(covered_index)

    # String description of the model
    descr = hyb_model.get_description(df_X["test"], y["test"])

    # Return
    return [hparams_hyrs["alpha"], hparams_hyrs["beta"], overall_acc_train, rule_coverage_train, overall_acc_v, rule_coverage_v, overall_acc_t, rule_coverage_t, descr]
   

# Result for one MPI runner
result_run = results_hyrs(bbox, df_X, y, hparams_hyrs, time_limit, 0.01)

res = []
if rank == 0 or not ccanada_expes:
    res.append(['na', 'na', bbox_acc_train, 0, bbox_acc_v, 0, bbox_acc_t, 0, bbox.__str__()])

res.append(result_run)

# Gather the results for the 5 folds on process 0
if ccanada_expes:
    res = comm.gather(res, root=0)

if rank == 0 or not ccanada_expes:
    # save results
    fileName = './results/results_4_2_post_%s_%s_%d_%s.csv' %(method, dataset_name, rseed, bbox_type) #_proportions
    import csv
    with open(fileName, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['alpha', 'beta', 'accuracy_train', 'transparency_train', 'accuracy_valid', 'transparency_valid', 'accuracy_test', 'transparency_test', 'model'])
        for i in range(len(res)):
            if ccanada_expes:
                for j in range(len(res[i])):
                    csv_writer.writerow(res[i][j])
            else:
                csv_writer.writerow(res[i])

