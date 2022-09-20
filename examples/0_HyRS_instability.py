import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from HyRS import HybridRuleSetClassifier
import os
import warnings
warnings.filterwarnings("ignore")
import argparse
from exp_utils import get_data

def main():
    
    parser = argparse.ArgumentParser(description='Instability of HyRS')
    parser.add_argument('--dataset', type=str, default='compas', help='adult, compas')
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, _ = get_data(args.dataset)


    # Fit a black-box
    bbox = RandomForestClassifier(random_state=42, min_samples_leaf=10, max_depth=10)
    bbox.fit(X_train, y_train)
    # Test performance
    print("BB Accuracy : ", np.mean(bbox.predict(X_test) == y_test), "\n")


    # Set parameters
    hparams = {
        "alpha" : 0.001,
        "beta" : 0.015
    }


    def sweep(beta):
        # Define a hybrid model
        hparams["beta"] = beta
        nreps = 10
        res = {'coverage' : [], 'seed' : np.arange(nreps), 'beta' : beta * np.ones(nreps)}
        for i in res['seed']:
            hyb_model = HybridRuleSetClassifier(bbox, **hparams)
            # Train the hybrid model
            hyb_model.fit(X_train, y_train, 200, random_state=23+i, T0=0.01, premined_rules=True)

            # Test performance
            _, covered_index = hyb_model.predict_with_type(X_test)
            res['coverage'].append(np.sum(covered_index) / len(covered_index))
        return pd.DataFrame(res)


    # Sweep over beta
    res = []
    for beta in np.logspace(-3, 0, 20):
        print(f"Beta : {beta:.2f}\n")
        res.append(sweep(beta))

    df = pd.concat(res)
    print(df)

    # Save results
    save_dir = os.path.join("results", "instability")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{args.dataset}_HyRS.csv")
    df.to_csv(filename, encoding='utf-8', index=False)



if __name__ == '__main__':
    main()

