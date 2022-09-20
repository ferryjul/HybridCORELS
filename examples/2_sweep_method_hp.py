import numpy as np
import pandas as pd
from companion_rule_list import CRL
from HyRS import HybridRuleSetClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

import argparse
import os



# Process results of CRL
def results_crl(bbox, X_train, y_train, X_test, y_test, hparams_crl, init_temperature):
    hyb_model = CRL(bbox, **hparams_crl)
    # Train the hybrid model
    hyb_model.fit(X_train, y_train, 50000, init_temperature, random_state=3, print_progress=True)
    # Evaluate the hybrid model
    output_rules, rule_coverage, overall_accuracy = hyb_model.test(X_test, y_test)

    row_list = []
    
    for i in range(len(output_rules)):
        row = {}
        row['accuracy'] = overall_accuracy[i]
        row['transparency'] = rule_coverage[i]
        row_list.append(row)
        print(f"acc:  {str(overall_accuracy[i])}, transp.:  {str(rule_coverage[i])}")
    df_res = pd.DataFrame(row_list)

    return df_res



# Process results of CRL
def results_hyrs(bbox, X_train, y_train, X_test, y_test, hparams_hyrs, init_temperature, seed):
    # Define a hybrid model
    hyb_model = HybridRuleSetClassifier(bbox, **hparams_hyrs)

    # Train the hybrid model
    hyb_model.fit(X_train, y_train, 100, random_state=seed, T0=init_temperature)
    # Test performance
    yhat, covered_index = hyb_model.predict_with_type(X_test)
    overall_accuracy = np.mean(yhat == y_test) 
    rule_coverage = np.sum(covered_index) / len(covered_index)

    # Store in dataframe
    df_res = pd.DataFrame([[overall_accuracy, rule_coverage]], columns=['accuracy', 'transparency'])

    return df_res



def main():
    parser = argparse.ArgumentParser()
    # train data, last column is label
    parser.add_argument("--dataset", type=str, help='Dataset name. Options: adult, compas', default='compas')
    parser.add_argument("--method", type=str, help='Method name. Options: hyrs, crl', default='hyrs')
    args = parser.parse_args()

    df = pd.read_csv(f"data/{args.dataset}.csv", sep = ',')
    X = df.iloc[:, :-1]
    y = np.array(df.iloc[:, -1])

    # Generate train and test sets
    random_state_param = 42
    train_proportion = 0.8
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0 - train_proportion, 
                                                            shuffle=True, random_state=random_state_param+1)

    # Fit a black-box
    bbox = RandomForestClassifier(random_state=42, min_samples_leaf=10, max_depth=10)
    bbox.fit(X_train, y_train)
    bbox_accuracy = np.mean(bbox.predict(X_test) == y_test) 
    # Evaluate the black box
    df = pd.DataFrame([[bbox_accuracy, 0]], columns=['accuracy', 'transparency'])

    # Explore the space of possible hybrid models
    if args.method == "crl":
        # Set parameters
        hparams_crl = {
            "min_support" : 0.05,
            "max_card" : 2,
            "alpha" : 0.001
        }
        temperatures = np.linspace(0.001, 0.01, num=10)
        alphas = np.logspace(0.001, -1, 10)
        for temperature in temperatures:
            for alpha in alphas:
                print(temperature, alpha)
                hparams_crl['alpha'] = alpha
                df = pd.concat([df, results_crl(X_train, y_train, X_test, y_test, hparams_crl, temperature)])
    
    elif args.method == "hyrs":
        # Set parameters
        hparams_hyrs = {
            "n_rules" : 5000,
            "min_support" : 1,
            "max_card" : 2,
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
                    df = pd.concat([df, results_hyrs(bbox, X_train, y_train, X_test, y_test, hparams_hyrs, 0.01, seed)])

    #save direcory
    save_dir = os.path.join("results", "acc_cov")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{args.dataset}_{args.method}.csv")
    df.to_csv(filename, encoding='utf-8', index=False)



if __name__ == '__main__':
    main()
