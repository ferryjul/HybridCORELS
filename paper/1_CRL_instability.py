import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from companion_rule_list import CRL
import os
import warnings
warnings.filterwarnings("ignore")
import argparse
from exp_utils import get_data, to_df

def main():
    
    parser = argparse.ArgumentParser(description='Instability of CRL')
    parser.add_argument('--dataset', type=str, default='compas', help='adult, compas')
    args = parser.parse_args()

    X, y, features, _ = get_data(args.dataset, {"train" : 0.8, "test" : 0.2})
    df_X = to_df(X, features)

    # Fit a black-box
    bbox = RandomForestClassifier(random_state=42, min_samples_leaf=10, max_depth=10)
    bbox.fit(df_X["train"], y["train"])
    # Test performance
    print("BB Accuracy : ", np.mean(bbox.predict(df_X["test"]) == y["test"]), "\n")


    nreps = 10
    res = {'coverage' : []}
    for i in range(nreps):
        hyb_model = CRL(bbox)
        # Train the hybrid model
        hyb_model.fit(df_X["train"], y["train"], 10000, random_state=23+i, 
                        init_temperature=0.01, premined_rules=True)
        # Get the coverages
        _, rule_coverage, _ = hyb_model.test(df_X["test"], y["test"])
        # Save Test performance
        res['coverage'].append(rule_coverage)

    res = pd.DataFrame(res)
    # Save results
    save_dir = os.path.join("results", "instability")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{args.dataset}_CRL.csv")
    res.to_csv(filename, encoding='utf-8', index=False)



if __name__ == '__main__':
    main()

