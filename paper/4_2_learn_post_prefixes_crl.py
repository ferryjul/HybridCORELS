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
time_limit = 3600 # seconds
interpr_mem = 8000 # MB
method = "CRL"
from companion_rule_list import CRL

# Train the interpretable part (CRL)
# (MPI grid)
# 100 MPI ranks
temperatures = np.linspace(0.001, 0.01, num=10) # Temperature for Simulated Annealing
alphas = np.logspace(-3, -1, 10) # Regularization Parameter [0, 1] (higher means shorted rulelists)

paramsList = []
for a in alphas:
    for t in temperatures:
        paramsList.append([a, t])
if verbosity:
    print("MPI # combinations of params: ", len(paramsList))
worker_params = paramsList[rank]
alpha = worker_params[0]
temperature = worker_params[1]

# Process results of CRL
def results_crl(bbox, df_X, y, alpha, time_limit, init_temperature, random_state_param):
    hyb_model = CRL(bbox, alpha=alpha)
    # Train the hybrid model
    hyb_model.fit(df_X["train"], y["train"], 10**7, init_temperature, random_state=random_state_param, 
                                            premined_rules=True, time_limit=time_limit)
    # Evaluate the hybrid model
    output_rules, rule_coverage_v, overall_accuracy_v = hyb_model.test(df_X["valid"], y["valid"])
    _, rule_coverage_t, overall_accuracy_t = hyb_model.test(df_X["test"], y["test"])
    _, rule_coverage_train, overall_accuracy_train = hyb_model.test(df_X["train"], y["train"])

    row_list = []
    
    for i in range(len(output_rules)):
        descr = hyb_model.get_description(df_X["test"], y["test"])
        row = [alpha, init_temperature, overall_accuracy_train[i], rule_coverage_train[i], overall_accuracy_v[i], rule_coverage_v[i], overall_accuracy_t[i], rule_coverage_t[i], descr]
        row_list.append(row)

    # Return
    return row_list


# Result for one MPI runner
result_run = results_crl(bbox, df_X, y, alpha, time_limit, temperature, rseed)

res = []
if rank == 0 or not ccanada_expes:
    res.append(['na', 'na', bbox_acc_train, 0, bbox_acc_v, 0, bbox_acc_t, 0, bbox.__str__()])

res.extend(result_run)

# Gather the results for the 5 folds on process 0
if ccanada_expes:
    res = comm.gather(res, root=0)

if rank == 0 or not ccanada_expes:
    # save results
    fileName = './results/results_4_2_post_%s_%s_%d_%s.csv' %(method, dataset_name, rseed, bbox_type) #_proportions
    import csv
    with open(fileName, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['alpha', 'temperature', 'accuracy_train', 'transparency_train', 'accuracy_valid', 'transparency_valid', 'accuracy_test', 'transparency_test', 'model'])
        for i in range(len(res)):
            if ccanada_expes:
                for j in range(len(res[i])):
                    csv_writer.writerow(res[i][j])
            else:
                csv_writer.writerow(res[i])

