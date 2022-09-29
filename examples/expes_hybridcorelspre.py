from exp_utils import get_data
import argparse
from local_config import ccanada_expes
from HybridCORELS import *
import numpy as np 
from black_box_models import BlackBox

bb_tout = 3600 # seconds
interpr_tout = 3600 # seconds
interpr_mem = 4000 # MB

if ccanada_expes: # one core performs operations for all values of alpha for all black-boxes
    from mpi4py import MPI

if ccanada_expes:
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
    #verbositylist = []
    #verbosity = False

parser = argparse.ArgumentParser(description='Instability of HyRS')
parser.add_argument('--dataset', type=int, default=0, help='1 for adult, 0 for compas')
args = parser.parse_args()
datasets = ["compas", "adult"]
dataset_name = datasets[args.dataset]

policies = ['objective', 'lower_bound', 'bfs']
cList = [0.001, 0.0001, 0.00001]
min_coverageList = [0.25, 0.50, 0.75, 0.95] #np.concatenate([np.arange(0, 1.0, 0.05), np.arange(0.96, 0.99, 0.01)])
alphaList = np.arange(0, 11, 1) # 11 values
bbtypes = ['random_forest'] #, 'ada_boost', 'gradient_boost']
dataset_seeds = [0,1,2,3,4]
paramsList = []

for p in policies:
    for c in cList:
        #for alpha in alphaList:
        for m in min_coverageList:
            #for bbtype in bbtypes:
            for dataset_seed in dataset_seeds:
                paramsList.append([p, c, m, dataset_seed])

#print(len(paramsList))

worker_params = paramsList[rank]
policy = worker_params[0]
cValue = worker_params[1]
min_coverage = worker_params[2]
random_state_value = worker_params[3]

X_train, X_test, y_train, y_test, features, prediction = get_data(dataset_name, random_state_param=random_state_value)

corels_params = {'policy':policy, 'max_card':1, 'c':cValue, 'n_iter':10**7, 'min_support':0.05, 'verbosity':verbositylist} #"progress"

beta_value = min([ (1 / X_train.shape[0]) / 2, cValue / 2]) # small enough to only break ties
if verbosity:
    print("beta = ", beta_value)

res = []

is_fitted_prefix = False
for alpha_value in alphaList:
    for black_box in bbtypes:
        if not is_fitted_prefix:
            # Create the hybrid model
            bbox = BlackBox(black_box, verbosity=verbosity, random_state_value=random_state_value, time_limit=bb_tout)
            hyb_model = HybridCORELSPreClassifier(black_box_classifier=bbox, beta=beta_value, alpha=alpha_value, min_coverage=min_coverage, lb_mode='tight', **corels_params)

            # Train the hybrid model
            hyb_model.fit(X_train, y_train, features=features, prediction_name=prediction, time_limit=interpr_tout, memory_limit=interpr_mem)

            # Compute and save all metrics
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

            status = hyb_model.get_status()
            sparsity = hyb_model.get_sparsity()
            is_fitted_prefix = True
        else: # only refit BB
            bbox = BlackBox(black_box, verbosity=verbosity, random_state_value=random_state_value, time_limit=bb_tout)
            hyb_model.refit_black_box(X_train, y_train, alpha_value,  bbox)

            # Recompute some metrics
            # Train set
            preds_train = hyb_model.predict(X_train)
            train_acc = np.mean(preds_train == y_train)
            black_box_accuracy_train = np.mean(preds_train[bb_indices_train] == y_train[bb_indices_train])

            # Test set
            preds_test = hyb_model.predict(X_test)
            test_acc = np.mean(preds_test == y_test)
            black_box_accuracy_test= np.mean(preds_test[bb_indices_test] == y_test[bb_indices_test])
        print("Expe: min_cov=%s" %min_coverage, "policy=%s" %policy, "c=%.3f" %cValue, "bb=%s"%black_box, "alpha=%.3f" %alpha_value, " done.")
        res.append([random_state_value, black_box, min_coverage, beta_value, alpha_value, policy, cValue, status, train_acc, test_acc, interpr_accuracy_train, interpr_accuracy_test, black_box_accuracy_train, black_box_accuracy_test, transparency_train, transparency_test, str(hyb_model), sparsity])

# Gather the results for the 5 folds on process 0
if ccanada_expes:
    res = comm.gather(res, root=0)

if rank == 0:
    # save results
    fileName = './results/results_HybridCORELSPre_%s.csv' %(dataset_name) #_proportions
    import csv
    with open(fileName, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Seed', 'Black Box', 'Min coverage', 'Beta', 'Alpha', 'Policy', 'Lambda', 'Search status', 'Training accuracy', 'Test accuracy', 'Training accuracy (prefix)', 'Test accuracy (prefix)', 'Training accuracy (BB)', 'Test accuracy (BB)', 'Training transparency', 'Test transparency', 'Model', 'Prefix length'])
        for i in range(len(res)):
            if ccanada_expes:
                for j in range(len(res[i])):
                    csv_writer.writerow(res[i][j])
            else:
                csv_writer.writerow(res[i])

