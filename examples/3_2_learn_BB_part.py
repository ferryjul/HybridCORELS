from exp_utils import get_data, computeAccuracyUpperBound
import argparse
from local_config import ccanada_expes
from HybridCORELS import *
import numpy as np 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import pickle 


if ccanada_expes: # one core performs operations for all values of alpha for all black-boxes
    from mpi4py import MPI

if ccanada_expes:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    verbositylist = []
    verbosity = False
    models_folder = "models_save"
else: # for local debug, fixed parameters
    rank = 0
    size = 1
    verbositylist=["progress", "hybrid"]
    models_folder = "models_graham"
    verbosity = True

parser = argparse.ArgumentParser(description='Learn BB part of prelearnt prefixes')
parser.add_argument('--dataset', type=int, default=0, help='1 for adult, 0 for compas')
args = parser.parse_args()
datasets = ["compas", "adult", "acs_employ"]
dataset_name = datasets[args.dataset]

min_coverageList = [0.25, 0.50, 0.75, 0.85, 0.95] # 5 values
alphaList = np.arange(0, 11, 1) # 11 values
dataset_seeds = [0,1,2,3,4] # 5 values
paramsList = []

bbtypes = ["random_forest", "ada_boost", "gradient_boost"]

for dataset_seed in dataset_seeds:
    for m in min_coverageList:       
        for alpha in alphaList:
            paramsList.append([dataset_seed, m, alpha])

if not ccanada_expes:
    print("# combinations of params: ", len(paramsList))

worker_params = paramsList[rank]
random_state_value = worker_params[0]
min_coverage = worker_params[1]
alpha_value = worker_params[2]

X, y, features, prediction = get_data(dataset_name, {"train" : 0.8, "test" : 0.2}, random_state_param=random_state_value)

if not ccanada_expes:
    print(X['train'].shape, X['test'].shape)

X_train, X_test, y_train, y_test = X['train'], X['test'], y['train'], y['test']

res = []

# => Retrieve the hybrid model with best prefix
n_iter_param = 10**9
dict_name = "prefixes_dict/%s_%d_%.4f" %(dataset_name, random_state_value, min_coverage)
with open('%s.pickle' %dict_name, 'rb') as handle:
    best_params = pickle.load(handle)
    min_support_param = best_params['min_support']
    policy = best_params['policy']
    cValue = best_params['c']
    print("Dataset %s, Fold %d, Min Coverage %.2f, best params are :" %(dataset_name, random_state_value, min_coverage), best_params)


beta_value = min([ (1 / X_train.shape[0]) / 2, cValue / 2]) # small enough to only break ties

hyb_model = HybridCORELSPreClassifier.load("./%s/prefix_%s_%d_%.3f_%.5f_%d_%.2f_%s" %(models_folder, dataset_name, random_state_value, min_coverage, cValue, n_iter_param, min_support_param, policy))

if verbosity:
    print("Loaded model: ", hyb_model)
    print("Status: ", hyb_model.get_status())

# Try different black-boxes
for black_box in bbtypes:
    if black_box == "random_forest":
        bbox = RandomForestClassifier(random_state=42, min_samples_split=10, max_depth=10)
    elif black_box == "ada_boost":
        bbox = AdaBoostClassifier(random_state=42)
    elif black_box == "gradient_boost":
        bbox = GradientBoostingClassifier(random_state=42)
    
    hyb_model.refit_black_box(X_train, y_train, alpha_value,  bbox)

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

    y_bb_train_counts = np.unique(y_train[bb_indices_train], return_counts=True)[1]
    black_box_accuracy_train_lb = max(y_bb_train_counts)/sum(y_bb_train_counts)
    assert(black_box_accuracy_train_lb == hyb_model.black_box_majority) # just a double check

    status = hyb_model.get_status()
    sparsity = hyb_model.get_sparsity()

    ##
    black_box_acc_upper_bound = computeAccuracyUpperBound(X_train[bb_indices_train], y_train[bb_indices_train])
    if verbosity:
        print("Accuracy upper bound = ", black_box_acc_upper_bound)

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

    y_bb_test_counts = np.unique(y_test[bb_indices_test], return_counts=True)[1]
    black_box_accuracy_test_lb = max(y_bb_test_counts)/sum(y_bb_test_counts)
    status = hyb_model.get_status()
    sparsity = hyb_model.get_sparsity()

    black_box_acc_upper_bound_test = computeAccuracyUpperBound(X_test[bb_indices_test], y_test[bb_indices_test])

    print("Expe: dataset = %s, " %dataset_name, "min_cov=%s" %min_coverage, "policy=%s" %policy, "c=%.3f" %cValue, "bb=%s"%black_box, "alpha=%.3f" %alpha_value, " done.")
    res.append([random_state_value, black_box, min_coverage, beta_value, alpha_value, policy, min_support_param, cValue, status, train_acc, test_acc, interpr_accuracy_train, interpr_accuracy_test, black_box_accuracy_train, black_box_acc_upper_bound, black_box_accuracy_train_lb, black_box_accuracy_test, black_box_acc_upper_bound_test, black_box_accuracy_test_lb, transparency_train, transparency_test, str(hyb_model), sparsity])

# Gather the results for the 5 folds on process 0
if ccanada_expes:
    res = comm.gather(res, root=0)

if rank == 0 or not ccanada_expes:
    # save results
    fileName = './results/results_HybridCORELSPre_wBB_%s_with_ub_lb.csv' %(dataset_name) #_proportions
    import csv
    with open(fileName, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Seed', 'Black Box', 'Min coverage', 'Beta', 'Alpha', 'Policy', 'Min support', 'Lambda', 'Search status', 'Training accuracy', 'Test accuracy', 'Training accuracy (prefix)', 'Test accuracy (prefix)', 'Training accuracy (BB)', 'Training accuracy (BB) UB',  'Training accuracy (BB) LB', 'Test accuracy (BB)', 'Test accuracy (BB) UB', 'Test accuracy (BB) LB', 'Training transparency', 'Test transparency', 'Model', 'Prefix length'])
        for i in range(len(res)):
            if ccanada_expes:
                for j in range(len(res[i])):
                    csv_writer.writerow(res[i][j])
            else:
                csv_writer.writerow(res[i])

