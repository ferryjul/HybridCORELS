import numpy as np
import pandas as pd
from HybridCORELS import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
from exp_utils import get_data

parser = argparse.ArgumentParser()
 # train data, last column is label
parser.add_argument("--dataset", type= int, help = '0 for COMPAS, 1 for Adult, 2 for ACS Employment', default = 0)
parser.add_argument("--method", type= str, help = 'pre or post, depending on the chosen paradigm', default = 'pre')
parser.add_argument("--alpha_value", type= int, help = 'when method is pre, value for the alpha hyperparameter (specialization coefficient)', default = 3)
parser.add_argument("--min_coverage", type=float, help = 'min_coverage constraint', default = 0.0)

args = parser.parse_args()
method = args.method
random_state_param = 0 #42
train_proportion = 0.8
alpha_value = args.alpha_value
beta_value = 0.0

datasets = ["compas", "adult", "acs_employ"]

dataset_name = datasets[args.dataset]

#df = pd.read_csv("data/{}.csv".format(dataset), sep = ',')
#X, y, features, prediction = load_from_csv("data/%s.csv" %dataset) 
X, y, features, prediction = get_data(dataset_name, {"train" : 0.8, "test" : 0.2}, random_state_param=random_state_param)

X_train, X_test, y_train, y_test = X['train'], X['test'], y['train'], y['test']
print(X_train.shape)
# Generate train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0 - train_proportion, shuffle=True, random_state=random_state_param+1)


#min_support=0.05, max_card=2, alpha=0.001
# Set parameters
corels_params = {'policy':"objective", 'max_card':1, 'c':0.001, 'n_iter':10**7, 'min_support':0.05, 'verbosity':["progress", "hybrid", "mine"]} # 

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
    from black_box_models import BlackBox
    # supported black-box types are: "random_forest", "ada_boost", "gradient_boost"
    bbox = BlackBox("gradient_boost", verbosity=True, random_state_value=42, n_iter=10, X_val=X_test, y_val=y_test, time_limit=120) #RandomForestClassifier(random_state=42, min_samples_split=10, max_depth=10)
    """bbox.fit(X_train, y_train) # Block to assess BB reproducibility
    print(bbox, bbox.score(X_test, y_test))
    bbox = BlackBox("gradient_boost", verbosity=True, random_state_value=42, n_iter=20, X_val=X_test, y_val=y_test)#, time_limit=10) #RandomForestClassifier(random_state=42, min_samples_split=10, max_depth=10)
    bbox.fit(X_train, y_train)
    print(bbox, bbox.score(X_test, y_test))"""

    # To use the interp-then-bb-training paradigm:
    if method == "pre":
        hyb_model = HybridCORELSPreClassifier(black_box_classifier=bbox, beta=beta_value, alpha=alpha_value, min_coverage=min_coverage, obj_mode='collab', **corels_params)#"progress"
    # To use the bb-then-interpr-training paradigm:
    elif method == "post":
        hyb_model = HybridCORELSPostClassifier(black_box_classifier=bbox, beta=beta_value, min_coverage=min_coverage, bb_pretrained=False, **corels_params)#"progress"
    
    # Train the hybrid model
    hyb_model.fit(X_train, y_train, features=features, prediction_name=prediction, time_limit=120, memory_limit=4000) # max. 10 sec or 1GB memory use to train the prefix (interpretable part of the hybrid model)

    print(hyb_model)
    print("Status = ", hyb_model.get_status())

    process(hyb_model, X_test, y_test)

    # test save / load with pickle
    #hyb_model.save("test_save_load")
    #hyb_model = HybridCORELSPreClassifier.load("test_save_load")


    #hyb_model.refit_black_box(X_train, y_train, alpha_value,  bbox)
    #print("===================>> train perfs")
    #process(hyb_model, X_train, y_train)
    #print("===================>> test perfs")
    #print(hyb_model)
    return process(hyb_model, X_test, y_test)

#save direcory
save_dir = "./results/pre_tests"
os.makedirs(save_dir, exist_ok=True)

min_coverages = [args.min_coverage] # np.linspace(0.40, 0.99, num=20)

df = pd.DataFrame()

for min_coverage in min_coverages:
    print("===================>> min_coverage {}".format(min_coverage))
    df = pd.concat([df, sweep(min_coverage)])

filename = '{}/{}.csv'.format(save_dir, dataset_name)

df.to_csv(filename, encoding='utf-8', index=False)