from HybridCORELS import *


dataset_name = "compas"
min_coverage = 0.975

rseeds = [0, 1, 2, 3, 4]
min_support_list = [0.01, 0.05, 0.1]  # Min Supports of Rules in Search Space
cList = [1e-2, 1e-3, 1e-4] # Regularisation CORELS
policies = ['objective', 'lower_bound', 'bfs'] # Priority Queue Criterion
paramsList = []
for policy in policies:
    for cValue in cList:
        for min_support_param in min_support_list:
            for rseed in rseeds:
                model_path = "models/pre_prefix_%s_%d_%.3f_%.5f_%d_%.2f_%s.pickle" %(dataset_name, rseed, min_coverage, cValue, n_iter_param, min_support_param, policy)
                hyb_model = HybridCORELSPreClassifier.load("models/%s"%model_file)
                sparsity = hyb_model.get_sparsity()
                if sparsity <= 1:
                    print(model_path)
                    print(hyb_model)