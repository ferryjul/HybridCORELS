import argparse
import numpy as np
from local_config import ccanada_expes
import os
#import warnings
#warnings.filterwarnings("ignore")
import argparse
from exp_utils import get_data, to_df
from black_box_models import BlackBox

# MPI grid
bbox_types = ['random_forest', 'ada_boost', 'gradient_boost']
rseeds = [0, 1, 2, 3, 4]
params = [] # will contain 3 * 5 = 15 values
for rs in rseeds:
    for bt in bbox_types:
        params.append([rs, bt]) 

if ccanada_expes:
    from mpi4py import MPI
    script_verbose = 0
    bbox_verbose = False
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
else:
    script_verbose = 1
    bbox_verbose = True
    rank = 0

param = params[rank]
rseed = param[0]
bbox_type = param[1]

parser = argparse.ArgumentParser()
# train data, last column is label
parser.add_argument("--dataset", type=int, help='Dataset ID', default=0)
#parser.add_argument("--bbox_type", type=str, help='Black box. Options: random_forest, ada_boost, gradient_boost', default='random_forest')
#parser.add_argument("--rseed", type=int, help='Random Data Split', default=0)
#parser.add_argument("--time_limit", type=int, help='Maximum run-time in seconds', default=3600)

args = parser.parse_args()


#time_limit = args.time_limit
time_limit = 20 * 3600
n_iters = 10

datasets = ["compas", "adult", "acs_employ"]
dataset_name = datasets[args.dataset]

# Get the data
if script_verbose > 0:
    print("Loading the data...")
X, y, features, prediction = get_data(dataset_name, {"train" : 0.6, "valid" : 0.20, "test" : 0.20}, 
                                random_state_param=rseed)
df_X = to_df(X, features)
if script_verbose > 0:
    print("Data loaded!")
#### Black Box ####
model_path = f"models/{dataset_name}_{bbox_type}_{rseed}.pickle"
if not os.path.exists(model_path):
    if script_verbose > 0:
        print("Fitting the Black Box\n")
    bbox = BlackBox(bb_type=bbox_type, verbosity=bbox_verbose, random_state_value=rseed, n_iter=n_iters, time_limit=time_limit, X_val=df_X["valid"], y_val=y["valid"])
    bbox.fit(df_X["train"], y["train"])
    bbox.save(model_path)
    train = 'Yes'
else:
    if script_verbose > 0:
        print("Loading the Black Box\n")
    bbox = BlackBox(bb_type=bbox_type).load(model_path)
    train = 'No'

# Black box performances
bbox_acc_train = np.mean(bbox.predict(df_X["train"]) == y["train"])
bbox_acc_v = np.mean(bbox.predict(df_X["valid"]) == y["valid"])
bbox_acc_t = np.mean(bbox.predict(df_X["test"]) == y["test"])
trials_details = bbox.trials_details
n_evals = bbox.n_evals
res = [[rseed, bbox_type, train, time_limit, n_iters, n_evals, bbox_acc_train, bbox_acc_v, bbox_acc_t, bbox.black_box_model, model_path, trials_details]]

# Gather the results for the 5 folds on process 0
if ccanada_expes:
    res = comm.gather(res, root=0)

if rank == 0 or not ccanada_expes:
    # save results
    fileName = './results/results_4_1_learn_post_black_boxes_%s.csv' %(dataset_name) #_proportions
    import csv
    with open(fileName, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Seed', 'Black Box', 'Train?', 'Time Limit', 'max#iterations', 'actual#iterations', 'Training accuracy', 'Validation accuracy', 'Test accuracy', 'Model', 'Path', 'Trials Details'])
        for i in range(len(res)):
            if ccanada_expes:
                for j in range(len(res[i])):
                    csv_writer.writerow(res[i][j])
            else:
                csv_writer.writerow(res[i])

