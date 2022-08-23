import numpy as np
import pandas as pd
from HybridCORELS import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

import argparse
import os

parser = argparse.ArgumentParser()
 # train data, last column is label
parser.add_argument("--dataset", type= str, help = 'Dataset name. Options: adult, compas', default = 'compas')
parser.add_argument("--method", type= str, help = 'pre or post, depending on the chosen paradigm', default = 'pre')

args = parser.parse_args()
method = args.method
random_state_param = 42
train_proportion = 0.8
alpha_value = 10
beta_value = 0.0
dataset = args.dataset
#df = pd.read_csv("data/{}.csv".format(dataset), sep = ',')
X, y, features, prediction = load_from_csv("data/%s.csv" %dataset) 


# Generate train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0 - train_proportion, 
                                                    shuffle=True, random_state=random_state_param+1)


#min_support=0.05, max_card=2, alpha=0.001
# Set parameters
corels_params = {'policy':"objective", 'max_card':1, 'c':0.001, 'n_iter':10**7, 'min_support':0.05, 'verbosity':["progress", "hybrid"]} #"progress"
bbox = RandomForestClassifier(random_state=42)#, min_samples_leaf=10, max_depth=10)

# Define a hybrid model


def process(model, X, y):
    preds, preds_types = model.predict_with_type(X)

    overall_accuracy = np.mean(preds == y)

    preds_types_counts = np.unique(preds_types, return_counts=True)
    index_one = np.where(preds_types_counts[0] == 1)
    cover_rate = preds_types_counts[1][index_one][0]/np.sum(preds_types_counts[1])
    row_list = []
    row = {}
    row['accuracy'] = overall_accuracy
    row['transparency'] = cover_rate
    row_list.append(row)
    print("acc:  {}, transp.:  {}".format(str(overall_accuracy), str(cover_rate)))
    df_res = pd.DataFrame(row_list)

    return df_res

def sweep(min_coverage):
    # To use the interp-then-bb-training paradigm:
    if method == "pre":
        hyb_model = HybridCORELSPreClassifier(black_box_classifier=bbox, beta=beta_value, alpha=alpha_value, min_coverage=min_coverage, lb_mode='tight', **corels_params)#"progress"
    # To use the bb-then-interpr-training paradigm:
    elif method == "post":
        hyb_model = HybridCORELSPostClassifier(black_box_classifier=bbox, beta=beta_value, min_coverage=min_coverage, bb_pretrained=False, **corels_params)#"progress"
   
    # Train the hybrid model
    hyb_model.fit(X_train, y_train, features=features, prediction_name=prediction)
    #print("===================>> train perfs")
    #process(hyb_model, X_train, y_train)
    #print("===================>> test perfs")
    print(hyb_model)
    return process(hyb_model, X_test, y_test)

#save direcory
save_dir = "./results/hycorels"
os.makedirs(save_dir, exist_ok=True)

min_coverages = [0.80] # np.linspace(0.40, 0.99, num=20)

df = pd.DataFrame()

for min_coverage in min_coverages:
    print("===================>> min_coverage {}".format(min_coverage))
    df = pd.concat([df, sweep(min_coverage)])

filename = '{}/{}.csv'.format(save_dir, args.dataset)

df.to_csv(filename, encoding='utf-8', index=False)