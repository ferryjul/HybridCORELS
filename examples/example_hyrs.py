import numpy as np
import pandas as pd
from HyRS import HybridRuleSetClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

random_state_param = 42
train_proportion = 0.8
dataset = 'compas' #'compas'
df = pd.read_csv('data/compas.csv', sep = ',')
X = df.iloc[:, :-1]
y = np.array(df.iloc[:, -1])

# Generate train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0 - train_proportion, 
                                                    shuffle=True, random_state=random_state_param+1)


# Fit a black-box
bb_model = RandomForestClassifier(random_state=42, min_samples_leaf=10, 
                                  max_depth=10)
bb_model.fit(X_train, y_train)


# Set parameters
hparams = {
    "n_rules" : 5000,
    "min_support" : 5,
    "max_card" : 4,
    "T0" : 0.01,
    "alpha" : 0.001,
    "beta" : 0.015
}

# Define a hybrid model
hyb_model = HybridRuleSetClassifier(bb_model, **hparams)

# Train the hybrid model
hyb_model.fit(X_train, y_train, 100, random_state=3, print_progress=True)
yhat, covered_index = hyb_model.predict_with_type(X_test) 

print("BB Accuracy : ", np.mean(bb_model.predict(X_test) == y_test))
print("Hybrid Accuracy : ", np.mean(yhat == y_test)) 
print("Coverage of RuleSet : ", np.sum(covered_index) / len(covered_index))

positive_rules = [hyb_model.prules[i] for i in hyb_model.positive_rule_set]
negative_rules = [hyb_model.nrules[i] for i in hyb_model.negative_rule_set]

print("Positive rules : \n", positive_rules)
print("Negative rules : \n", negative_rules)
