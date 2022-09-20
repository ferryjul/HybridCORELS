
import numpy as np
import pandas as pd
from corels import CorelsClassifier
from sklearn.ensemble import RandomForestClassifier
from HybridCORELS import HybridCORELSPreClassifier, HybridCORELSPostClassifier
from exp_utils import get_data
from HyRS import HybridRuleSetClassifier
from companion_rule_list import CRL

random_state_param = 42
X_train, X_test, y_train, y_test, features = get_data("compas")


print("--------------- CORELS ---------------\n")
rulelist = CorelsClassifier(max_card=1).fit(X_train.to_numpy(), y_train, features=features)

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
    "alpha" : 0.001,
    "beta" : 0.02
}

# Define a hybrid model
hyb_model = HybridRuleSetClassifier(bbox, **hparams)

# Train the hybrid model
hyb_model.fit(X_train, y_train, 200, T0=0.01, premined_rules=True, random_state=3)

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
    "max_card" : 2,
    "alpha" : 0.001
}

# Define a hybrid model
hyb_model = CRL(bbox, **hparams)

# Train the hybrid model
hyb_model.fit(X_train, y_train, random_state=random_state_param+1, premined_rules=True)
output_rules, rule_coverage, test_acc = hyb_model.test(X_test, y_test)

# Print the RuleList
print(" ACC ,  C  ,  Rule")
for (i,j,k) in zip(test_acc, output_rules, rule_coverage):
    print(f"{i:.3f}", f"{k:.3f}", j)
print("\n")



print("---------------  HybridCORELSPreClassifier ---------------")
# Set parameters
corels_params = {
    'policy' : "objective", 
    'max_card' : 1, 
    'c' : 0.001, 
    'n_iter' : 10**6,
    'verbosity': []
}

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