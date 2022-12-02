import pickle

min_coverageList = [0.25, 0.50, 0.75, 0.85, 0.95] #np.concatenate([np.arange(0, 1.0, 0.05), np.arange(0.96, 0.99, 0.01)])
dataset_seeds = [0,1,2,3,4]
datasets = ["compas", "adult", "acs_employ"]

for dataset_name in datasets:
    accs_list = {}
    cov_list = {}
    accs_list_test = {}
    cov_list_test = {}
    for random_state_value in dataset_seeds:
        
        for min_coverage in min_coverageList:

            if not min_coverage in accs_list.keys():
                accs_list[min_coverage] = []
                cov_list[min_coverage] = []
                accs_list_test[min_coverage] = []
                cov_list_test[min_coverage] = []

            dict_name = "prefixes_dict/%s_%d_%.4f" %(dataset_name, random_state_value, min_coverage)
            
            with open('%s.pickle' %dict_name, 'rb') as handle:
                best_params = pickle.load(handle)
                min_support_param = best_params['min_support']
                policy = best_params['policy']
                cValue = best_params['c']
                print("Dataset %s, Fold %d, Min Coverage %.2f, best params are :" %(dataset_name, random_state_value, min_coverage), best_params)
