import argparse
import numpy as np
import pandas as pd
from corels import CorelsClassifier
from sklearn.ensemble import RandomForestClassifier
from HyRS import HybridRuleSetClassifier
from companion_rule_list import CRL
#from HybridCORELS import HybridCORELSPreClassifier, HybridCORELSPostClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('data/compas.csv', sep = ',')
X = df.iloc[:, :-1]
y = np.array(df.iloc[:, -1])
features = list(X.columns)

# Generate train and test sets
random_state_param = 42
train_proportion = 0.8
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0 - train_proportion, 
                                                    shuffle=True, random_state=random_state_param+1)


print("--------------- CORELS ---------------\n")
rulelist = CorelsClassifier(max_card=2, min_support=0.01).fit(X_train.to_numpy(), y_train, features=features)

print("CORELS Accuracy : ", np.mean(rulelist.predict(X_test.to_numpy()) == y_test), "\n")



print("--------------- BB ---------------\n")
# Fit a black-box
bbox = RandomForestClassifier(random_state=42, min_samples_leaf=10, max_depth=10)
bbox.fit(X_train, y_train)
# Test performance
print("BB Accuracy : ", np.mean(bbox.predict(X_test) == y_test), "\n")



print("--------------- HyRS ---------------\n")
# Set parameters
hparams = {
    "n_rules" : 5000,
    "min_support" : 1,
    "max_card" : 2,
    "alpha" : 0.001,
    "beta" : 0.015
}

# Define a hybrid model
hyb_model = HybridRuleSetClassifier(bbox, **hparams)

# Train the hybrid model
hyb_model.fit(X_train, y_train, 100, random_state=random_state_param, T0=0.01)

# Print the RuleSet
positive_rules = [hyb_model.prules[i] for i in hyb_model.positive_rule_set]
negative_rules = [hyb_model.nrules[i] for i in hyb_model.negative_rule_set]
print("Positive rules : \n", positive_rules)
print("Negative rules : \n", negative_rules, "\n")

# Test performance
yhat, covered_index = hyb_model.predict_with_type(X_test)
print("HyRS Accuracy : ", np.mean(yhat == y_test)) 
print("Coverage of RuleSet : ", np.sum(covered_index) / len(covered_index), "\n")



print("--------------- CRL ---------------\n")
# Set parameters
hparams = {
    "min_support" : 0.01,
    "max_card" : 2,
    "alpha" : 0.001
}

# Define a hybrid model
hyb_model = CRL(bbox, **hparams)

# Train the hybrid model
hyb_model.fit(X_train, y_train, random_state=random_state_param+1)
output_rules, rule_coverage, test_acc = hyb_model.test(X_test, y_test)

# Print the RuleList
print(" ACC ,  C  ,  Rule")
for (i,j,k) in zip(test_acc, output_rules, rule_coverage):
    print(f"{i:.3f}", f"{k:.3f}", j)



print("---------------  HybridCORELSPreClassifier ---------------")
# Set parameters
corels_params = {
    'policy' : "objective", 
    'max_card' : 1, 
    'c' : 0.001, 
    'n_iter' : 10**7, 
    'min_support' : 0.05,
    'verbosity': []
}

bbox = RandomForestClassifier(random_state=42, min_samples_leaf=10, max_depth=10)
alpha_value = 10
beta_value = 0.0
# Define a hybrid model
hyb_model = HybridCORELSPreClassifier(black_box_classifier=bbox, beta=beta_value, alpha=alpha_value, 
                                      min_coverage=0.4, lb_mode='tight', **corels_params)
# Train the hybrid model
hyb_model.fit(X_train.to_numpy(), y_train, features=features)

# Print the RuleList
print("\n", hyb_model, "\n")

# Test performance
yhat, covered_index = hyb_model.predict_with_type(X_test.to_numpy())
print("HybridCORELSPreClassifier Accuracy : ", np.mean(yhat == y_test)) 
print("Coverage of RuleList : ", np.sum(covered_index) / len(covered_index), "\n")



print("---------------  HybridCORELSPostClassifier ---------------")

bbox = RandomForestClassifier(random_state=42, min_samples_leaf=10, max_depth=10)
hyb_model = HybridCORELSPostClassifier(black_box_classifier=bbox, beta=beta_value, min_coverage=0.4, 
                                       bb_pretrained=False, **corels_params)
# Train the hybrid model
hyb_model.fit(X_train.to_numpy(), y_train, features=features)

# Print the RuleList
print("\n", hyb_model, "\n")

# Test performance
yhat, covered_index = hyb_model.predict_with_type(X_test.to_numpy())
print("HybridCORELSPreClassifier Accuracy : ", np.mean(yhat == y_test)) 
print("Coverage of RuleList : ", np.sum(covered_index) / len(covered_index), "\n")