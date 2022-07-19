from HybridCORELS import *
from corels import *
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import argparse

# parser initialization
parser = argparse.ArgumentParser(description='HybridCORELS experiments')
parser.add_argument('--method', type=int, default=0, help='0 = black-box, 1 = HybridCORELS, 2 = CORELS')
parser.add_argument('--min_coverage', type=float, default=0.0, help='When method==1, value of the min_coverage parameter.')
args = parser.parse_args()
method = args.method


# Parameters
random_state_param = 42
debug = False
train_proportion = 0.8
#black_box = MLPClassifier(max_iter=500, random_state=42)
black_box = RandomForestClassifier(min_samples_leaf=10, max_depth=10, random_state=random_state_param)
corels_params = {'policy':"lower_bound", 'max_card':2, 'c':0.0001, 'n_iter':10**7, 'min_support':0.05, 'verbosity':["progress"]} #"progress"
beta_value = 1/30000 #0.0 #35
alpha_value = 3
min_coverage_value = args.min_coverage# 0.4

dataset = 'adult' #'compas'
X, y, features, prediction = load_from_csv("data/%s.csv" %dataset) 
if dataset == 'compas':
    sens_attr = 'Race=Caucasian' #'sex_male' # 'race_white'
elif dataset == 'adult':
    sens_attr = 'sex_male'
else:
    raise ValueError("Unknown dataset: %s" %dataset)

majority_group = features.index(sens_attr)
print("Majority group is ", features[majority_group], "(covers %.3f of data)" %(np.average(X[:, majority_group])))

# Generate train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0 - train_proportion, shuffle=True, random_state=random_state_param+1)

# print("Using %d examples for training" %(X_train.shape[0]))

# Classifier creation
if method == 0:
    print("Training black-box model: ", black_box)
    c = black_box
elif method == 1:
    print("Training HybridCORELS classifier - Beta = %.3f - Alpha = %.3f" %(beta_value, alpha_value))
    c = HybridCORELSClassifier(black_box_classifier=black_box, beta=beta_value, alpha=alpha_value, min_coverage=min_coverage_value, **corels_params)#"progress"
elif method == 2:
    print("Training CORELS classifier")
    c = CorelsClassifier(**corels_params)
else:
    raise ValueError("Method value %d unknown." %method)

print("Training set: ", X_train.shape)
# Classifier training
if method == 0:
    c.fit(X_train, y_train)
elif method == 1 or method == 2:
    # Fit the model. Features is a list of the feature names
    c.fit(X_train, y_train, features=features, prediction_name=prediction)
    # Print the model
    print(c)
else:
    raise ValueError("Method value %d unknown." %method)

# Score the model on the train set
a = c.score(X_train, y_train)

print("Train Accuracy: " + str(a))

# Score the model on the test set
a = c.score(X_test, y_test)

print("Test Accuracy: " + str(a))

# Test predict_proba
if method in [0,1] and debug:
    print(" === CHECK predict_proba CORRECTNESS === ")
    proba_preds = c.predict_proba(X_train)
    preds = np.ones(shape=y_train.shape)
    preds[np.where(proba_preds[:,0] > 0.5)] = 0
    print("Train Accuracy (computed from probas): ", np.mean(preds == y_train))

    proba_preds = c.predict_proba(X_test)
    preds = np.ones(shape=y_test.shape)
    preds[np.where(proba_preds[:,0] > 0.5)] = 0
    print("Test Accuracy (computed from probas): ", np.mean(preds == y_test))

'''
# Test predict_with_type
if method == 1 and debug:
    print(" === CHECK preds_with_type CORRECTNESS === ")
if method == 1:
    for data_split in ['train', 'test']:
        if data_split == 'train':
            X = X_train
        else:
            X = X_test
        preds, preds_types = c.predict_with_type(X)
        preds_types_counts = np.unique(preds_types, return_counts=True)
        index_one = np.where(preds_types_counts[0] == 1)
        print("%.3f %s examples classified by interpretable part." %(preds_types_counts[1][index_one]/np.sum(preds_types_counts[1]), data_split))
        majority_indices = np.where(X[:,majority_group] == 1)
        minority_indices = np.where(X[:,majority_group] == 0)
        assert(majority_indices[0].size + minority_indices[0].size == X.shape[0])
        print("Average access to interpretable part (%s) for %s = %.3f (%d/%d)" %(data_split, sens_attr, np.average(preds_types[majority_indices]), np.sum(preds_types[majority_indices]), majority_indices[0].size))
        print("Average access to interpretable part (%s) for not(%s) = %.3f (%d/%d)" %(data_split, sens_attr, np.average(preds_types[minority_indices]), np.sum(preds_types[minority_indices]), minority_indices[0].size))
print("----------------------------------------------------------")
'''