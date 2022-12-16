
import numpy as np
import pandas as pd
from corels import CorelsClassifier
from sklearn.ensemble import RandomForestClassifier
from HybridCORELS import HybridCORELSPreClassifier, HybridCORELSPostClassifier
from exp_utils import get_data, to_df
from HyRS import HybridRuleSetClassifier
from companion_rule_list import CRL

random_state_param = 42
X, y, features, _ = get_data("compas", {"train" : 0.8, "test" : 0.2})


print("--------------- CORELS ---------------\n")
rulelist = CorelsClassifier(max_card=1).fit(X["train"], y["train"], features=features)

print("CORELS Accuracy : ", np.mean(rulelist.predict(X["test"]) == y["test"]), "\n")



print("--------------- BB ---------------\n")
df_X = to_df(X, features)
# Fit a black-box
bbox = RandomForestClassifier(random_state=42, min_samples_leaf=10, max_depth=10)
bbox.fit(df_X["train"], y["train"])
# Test performance
print("BB Accuracy : ", np.mean(bbox.predict(df_X["test"]) == y["test"]), "\n")



print("--------------- HyRS ---------------\n")
# Set parameters
hparams = {
    "alpha" : 0.001,
    "beta" : 0.02
}

# Define a hybrid model
hyb_model = HybridRuleSetClassifier(bbox, **hparams)

# Train the hybrid model
hyb_model.fit(df_X["train"], y["train"], 100, T0=0.01, premined_rules=True, 
                                            random_state=3, time_limit=10)
print(hyb_model.get_description(df_X["test"], y["test"]))



print("--------------- CRL ---------------\n")
# Set parameters
hparams = {
    "max_card" : 2,
    "alpha" : 0.01
}

# Define a hybrid model
hyb_model = CRL(bbox, **hparams)

# Train the hybrid model
hyb_model.fit(df_X["train"], y["train"], n_iteration=50000, random_state=random_state_param+1, 
                                                            premined_rules=True, time_limit=10)
print(hyb_model.get_description(df_X["test"], y["test"]))


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
                                      min_coverage=0.4, **corels_params)
# Train the hybrid model
hyb_model.fit(X["train"], y["train"], features=features)

# Print the RuleList
print("\n", hyb_model, "\n")

# Test performance
yhat, covered_index = hyb_model.predict_with_type(X["test"])
print("HybridCORELSPreClassifier Accuracy : ", np.mean(yhat == y["test"])) 
print("Coverage of RuleList : ", np.sum(covered_index) / len(covered_index), "\n")



print("---------------  HybridCORELSPostClassifier ---------------")

hyb_model = HybridCORELSPostClassifier(black_box_classifier=bbox, beta=beta_value, min_coverage=0.4, 
                                       bb_pretrained=False, **corels_params)
# Train the hybrid model
hyb_model.fit(X["train"], y["train"], features=features)

# Print the RuleList
print("\n", hyb_model, "\n")

# Test performance
yhat, covered_index = hyb_model.predict_with_type(X["test"])
print("HybridCORELSPreClassifier Accuracy : ", np.mean(yhat == y["test"])) 
print("Coverage of RuleList : ", np.sum(covered_index) / len(covered_index), "\n")