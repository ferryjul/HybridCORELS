from HybridCORELS import *
from corels import *
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import argparse

# parser initialization
parser = argparse.ArgumentParser(description='HybridCORELS experiments')
parser.add_argument('--method', type=int, default=0, help='0 = black-box, 1 = HybridCORELS, 2 = CORELS')
args = parser.parse_args()
method = args.method


# Parameters
random_state_param = 42
debug = False
train_proportion = 0.8
#black_box = MLPClassifier(max_iter=500, random_state=42)
black_box = RandomForestClassifier(min_samples_split=10, random_state=random_state_param)
corels_params = {'policy':"lower_bound", 'max_card':2, 'c':0.0001, 'n_iter':10**6, 'min_support':0.05, 'verbosity':[]} #"progress"

X, y, features, prediction = load_from_csv("data/adult.csv")
sens_attr = 'sex_male' # 'race_white'
majority_group = features.index(sens_attr)
print("Majority group is ", features[majority_group], "(covers %.3f of data)" %(np.average(X[majority_group])))

# Generate train and test sets
train_split = int(train_proportion * X.shape[0])

X_train = X[:train_split]
y_train = y[:train_split]

X_test = X[train_split:]
y_test = y[train_split:]

acc_list = []
acc_list_train = []
interpretable_part_list = []
interpretable_part_list_train = []

beta_value = 0.25
beta_list = np.arange(0,15,0.5)

for alpha_value in beta_list:
    # print("Using %d examples for training" %(X_train.shape[0]))

    # Classifier creation
    if method == 0:
        print("Training black-box model: ", black_box)
        c = black_box
    elif method == 1:
        print("Training HybridCORELS classifier - Beta = %.3f" %(beta_value))
        c = HybridCORELSClassifier(black_box_classifier=black_box, beta=beta_value, alpha=alpha_value, **corels_params)#"progress"
    elif method == 2:
        print("Training CORELS classifier")
        c = CorelsClassifier(**corels_params)
    else:
        raise ValueError("Method value %d unknown." %method)

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
    acc_train = c.score(X_train, y_train)

    print("Train Accuracy: " + str(acc_train))

    # Score the model on the test set
    acc_test = c.score(X_test, y_test)

    print("Test Accuracy: " + str(acc_test))

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

    # Test predict_with_type
    if method == 1 and debug:
        print(" === CHECK preds_with_type CORRECTNESS === ")
    if method == 1:
        preds, preds_types = c.predict_with_type(X_test)
        preds_types_counts = np.unique(preds_types, return_counts=True)
        index_one = np.where(preds_types_counts[0] == 1)
        interpretable_part = preds_types_counts[1][index_one]/np.sum(preds_types_counts[1])
        print("%.3f test examples classified by interpretable part." %(interpretable_part))
        preds, preds_types = c.predict_with_type(X_train)
        preds_types_counts = np.unique(preds_types, return_counts=True)
        index_one = np.where(preds_types_counts[0] == 1)
        interpretable_part_train = preds_types_counts[1][index_one]/np.sum(preds_types_counts[1])
        print("%.3f test examples classified by interpretable part." %(interpretable_part_train))
   
        acc_list.append(acc_test)
        acc_list_train.append(acc_train)
        interpretable_part_list.append(interpretable_part)
        interpretable_part_list_train.append(interpretable_part_train)
    print("----------------------------------------------------------")

import matplotlib.pyplot as plt 

plt.plot(beta_list, acc_list_train, label='Accuracy (train)', marker='x')
plt.xlabel("Alpha")
plt.show()
plt.plot(beta_list, interpretable_part_list_train, label='%Interpretable (train)', marker='x')
plt.xlabel("Alpha")
plt.show()
plt.xlabel("Alpha")
plt.plot(beta_list, acc_list, label='Accuracy (test)', marker='x')
plt.show()
plt.xlabel("Alpha")
plt.plot(beta_list, interpretable_part_list, label='%Interpretable (test)', marker='x')
plt.show()

