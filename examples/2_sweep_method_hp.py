import numpy as np
import pandas as pd
import argparse
import os

import warnings
warnings.filterwarnings("ignore")

# Local
from companion_rule_list import CRL
from HyRS import HybridRuleSetClassifier
from exp_utils import get_data, to_df
from black_box_models import BlackBox
from HybridCORELS import HybridCORELSPostClassifier



# Process results of HybridCORELSPostClassifier
def results_hybrid_post(bbox, df_X, y, min_coverage, **corels_params):
    hyb_model = HybridCORELSPostClassifier(black_box_classifier=bbox, beta=0.0, min_coverage=min_coverage, 
                                       bb_pretrained=True, **corels_params)
    # Train the hybrid model
    hyb_model.fit(df_X["train"].to_numpy(), y["train"], time_limit=120)
    print(hyb_model.get_status())

    # Valid performance
    yhat, covered_index = hyb_model.predict_with_type(df_X["valid"].to_numpy())
    overall_acc_v = np.mean(yhat == y["valid"])
    rule_coverage_v = np.sum(covered_index) / len(covered_index)

    # Test performance
    yhat, covered_index = hyb_model.predict_with_type(df_X["test"].to_numpy())
    overall_acc_t =  np.mean(yhat == y["test"])
    rule_coverage_t = np.sum(covered_index) / len(covered_index)

    # Store in dataframe
    df_res = pd.DataFrame([[overall_acc_v, rule_coverage_v, overall_acc_t, rule_coverage_t]], 
                            columns=['accuracy_valid', 'transparency_valid',
                                     'accuracy_test', 'transparency_test'])

    return df_res



# Process results of CRL
def results_crl(bbox, df_X, y, alpha, init_temperature):
    hyb_model = CRL(bbox, alpha=alpha)
    # Train the hybrid model
    hyb_model.fit(df_X["train"], y["train"], 50000, init_temperature, random_state=3, premined_rules=True)
    # Evaluate the hybrid model
    output_rules, rule_coverage_v, overall_accuracy_v = hyb_model.test(df_X["valid"], y["valid"])
    _, rule_coverage_t, overall_accuracy_t = hyb_model.test(df_X["test"], y["test"])

    row_list = []
    
    for i in range(len(output_rules)):
        row = {}
        row['accuracy_valid'] = overall_accuracy_v[i]
        row['transparency_valid'] = rule_coverage_v[i]
        row['accuracy_test'] = overall_accuracy_t[i]
        row['transparency_test'] = rule_coverage_t[i]
        row_list.append(row)
    df_res = pd.DataFrame(row_list)

    return df_res



# Process results of HyRS
def results_hyrs(bbox, df_X, y, hparams_hyrs, init_temperature, seed):
    # Define a hybrid model
    hyb_model = HybridRuleSetClassifier(bbox, **hparams_hyrs)

    # Train the hybrid model
    hyb_model.fit(df_X["train"], y["train"], 500, random_state=seed, T0=init_temperature, premined_rules=True)

    # Valid performance
    yhat, covered_index = hyb_model.predict_with_type(df_X["valid"])
    overall_acc_v = np.mean(yhat == y["valid"]) 
    rule_coverage_v = np.sum(covered_index) / len(covered_index)

    # Test performance
    yhat, covered_index = hyb_model.predict_with_type(df_X["test"])
    overall_acc_t = np.mean(yhat == y["test"]) 
    rule_coverage_t = np.sum(covered_index) / len(covered_index)

    # Store in dataframe
    df_res = pd.DataFrame([[overall_acc_v, rule_coverage_v, overall_acc_t, rule_coverage_t]], 
                            columns=['accuracy_valid', 'transparency_valid',
                                     'accuracy_test', 'transparency_test'])

    return df_res



def main():
    parser = argparse.ArgumentParser()
    # train data, last column is label
    parser.add_argument("--dataset", type=str, help='Dataset name. Options: adult, compas', default='compas')
    parser.add_argument("--bbox", type=str, help='Black box. Options: random_forest, ada_boost, gradient_boost', default='random_forest')
    args = parser.parse_args()

    # Save direcory
    save_dir = os.path.join("results", "acc_cov")
    os.makedirs(save_dir, exist_ok=True)

    # Get the data
    X, y, features, _ = get_data(args.dataset, {"train" : 0.7, "valid" : 0.15, "test" : 0.15})
    df_X = to_df(X, features)

    #### Black Box ####
    print("Fitting the Black Box\n")
    bbox = BlackBox(bb_type=args.bbox, verbosity=True, n_iter=2, X_val=df_X["valid"], y_val=y["valid"])
    bbox.fit(df_X["train"], y["train"])
    bbox_acc_v = np.mean(bbox.predict(df_X["valid"]) == y["valid"])
    bbox_acc_t = np.mean(bbox.predict(df_X["test"]) == y["test"])


    #### HybridCORELSPostClassifier ####
    print("Fitting HybridCORELSPostClassifier\n")
    # Where to store results
    df = pd.DataFrame([[bbox_acc_v, 0, bbox_acc_t, 0]], columns=['accuracy_valid', 'transparency_valid',
                                                                 'accuracy_test', 'transparency_test'])
    # Set parameters
    corels_params = {
        'max_card' : 1,
        'min_support' : 0.01,
        'n_iter' : 10**7,
        'verbosity': []
    }

    cs = [1e-2, 1e-3, 1e-4]
    policies = ['objective', 'lower_bound', 'bfs']
    min_coverages = [0.25, 0.50, 0.75, 0.85, 0.95]
    min_supports = [0.01, 0.05]
    for policy in policies:
        for c in cs:
            for min_coverage in min_coverages:
                for min_support in min_supports:
                    print(policy, c, min_coverage, min_support)
                    corels_params["c"] = c
                    corels_params["policy"] = policy
                    corels_params["min_support"] = min_support
                    df = pd.concat([df, results_hybrid_post(bbox, df_X, y, min_coverage, **corels_params)])
    
    filename = os.path.join(save_dir, f"{args.dataset}_{args.bbox}_hybrid_post.csv")
    df.to_csv(filename, encoding='utf-8', index=False)



    #### CRL ####
    print("Fitting CRL\n")
    # Where to store results
    df = pd.DataFrame([[bbox_acc_v, 0, bbox_acc_t, 0]], columns=['accuracy_valid', 'transparency_valid',
                                                                 'accuracy_test', 'transparency_test'])
    # Set parameters
    temperatures = np.linspace(0.001, 0.01, num=10)
    alphas = np.logspace(-3, -1, 10)
    for temperature in temperatures:
        for alpha in alphas:
            print(temperature, alpha)
            df = pd.concat([df, results_crl(bbox, df_X, y, alpha, temperature)])
    
    filename = os.path.join(save_dir, f"{args.dataset}_{args.bbox}_crl.csv")
    df.to_csv(filename, encoding='utf-8', index=False)



    #### HyRS ####
    print("Fitting HyRS\n")
    # Where to store results
    df = pd.DataFrame([[bbox_acc_v, 0, bbox_acc_t, 0]], columns=['accuracy_valid', 'transparency_valid',
                                                                 'accuracy_test', 'transparency_test'])
    # Set parameters
    hparams_hyrs = {
        "alpha" : 0.001,
        "beta" : 0.015
    }
    #temperatures = np.linspace(0.001, 0.01, num=10)
    alphas = np.logspace(-3, -2, 10)
    betas = np.logspace(-3, 0, 10)
    #for temperature in temperatures:
    for alpha in alphas:
        for beta in betas:
            for seed in range(5):
                print(alpha, beta, seed)
                hparams_hyrs['alpha'] = alpha
                hparams_hyrs['beta'] = beta
                df = pd.concat([df, results_hyrs(bbox, df_X, y, hparams_hyrs, 0.01, seed)])

    filename = os.path.join(save_dir, f"{args.dataset}_{args.bbox}_hyrs.csv")
    df.to_csv(filename, encoding='utf-8', index=False)





if __name__ == '__main__':
    main()
