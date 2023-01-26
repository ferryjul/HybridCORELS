import numpy as np 
from exp_utils import get_data
import matplotlib.pyplot as plt 
import pickle
import pandas as pd

dict_save_folder = '4_1_pre_best_prefixes'

datasets = ["compas", "adult", "acs_employ"]
n_iter_param = 10**9
min_coverageList = [] # Will contain 20 values
cov = 0.1
while cov < 1.0:
    min_coverageList.append(cov)
    cov += 0.1
    cov = round(cov, 2)
min_coverageList.extend([0.925, 0.95, 0.975])
criterion_column = 'Val acc UB' #'Validation accuracy (prefix)'

for dataset_name in datasets:
    for min_coverage in min_coverageList:
        # Find per dataset-fold-coverage best prefix selection
        fileName = './results_part_4/results_4_1_learn_pre_prefixes_%s_%.3f_collab.csv' %(dataset_name, min_coverage) #_proportions
        dataset_cov_res_dict = {}
        try:
            res = pd.read_csv(fileName)
        except FileNotFoundError():
            print("File not found: ", fileName)
            exit()

        # Iterate over results
        for index, row in res.iterrows():
            rseed = row['Seed']
            cValue = row['Lambda']
            min_support_param = row['Min support']
            policy = row['Policy']
            validation_accuracy_prefix = row[criterion_column]
            if not rseed in dataset_cov_res_dict.keys():
                dataset_cov_res_dict[rseed] = {'cValue':cValue, 
                                                'min_support_param':min_support_param, 
                                                'policy':policy,
                                                'validation_accuracy_prefix':validation_accuracy_prefix}
            else:
                if validation_accuracy_prefix > dataset_cov_res_dict[rseed]['validation_accuracy_prefix']:
                    dataset_cov_res_dict[rseed] = {'cValue':cValue, 
                                                    'min_support_param':min_support_param, 
                                                    'policy':policy,
                                                    'validation_accuracy_prefix':validation_accuracy_prefix}
        # Save best configs
        for rseed in dataset_cov_res_dict.keys():
            dict_name = '%s_%d_%.3f_collab.pickle' %(dataset_name, rseed, min_coverage)
            local_dict = {'dataset_name':dataset_name, 
                            'rseed':rseed, 
                            'min_coverage':min_coverage, 
                            'cValue':dataset_cov_res_dict[rseed]['cValue'], 
                            'n_iter_param':n_iter_param, 
                            'min_support_param':dataset_cov_res_dict[rseed]['min_support_param'], 
                            'policy':dataset_cov_res_dict[rseed]['policy'],
                            'validation_accuracy_prefix':dataset_cov_res_dict[rseed]['validation_accuracy_prefix']}
        
            with open('%s/%s.pickle' %(dict_save_folder, dict_name), 'wb') as handle:
                pickle.dump(local_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('%s/%s.pickle' %(dict_save_folder, dict_name), 'rb') as handle:
                b = pickle.load(handle)
                print(b)