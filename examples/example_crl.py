import numpy as np
import pandas as pd
from companion_rule_list import CRL, my_debug
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")



random_state_param = 42
train_proportion = 0.8
dataset = "compas"
df = pd.read_csv("data/{}.csv".format(dataset), sep = ',')
X = df.iloc[:, :-1]
y = np.array(df.iloc[:, -1])

# Generate train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0 - train_proportion, 
                                                    shuffle=True, random_state=random_state_param+1)


# Fit a black-box
bbox = RandomForestClassifier(random_state=42, min_samples_leaf=10, max_depth=10)
bbox.fit(X_train, y_train)

#min_support=0.05, max_card=2, alpha=0.001
# Set parameters
hparams = {
    "min_support" : 0.05,
    "max_card" : 2,
    "alpha" : 0.001
}

# Define a hybrid model
hyb_model = CRL(bbox, **hparams)


# Train the hybrid model
hyb_model.fit(X_train, y_train, 1000, 0.001, random_state=3, print_progress=True)

def process(X, y):
    overall_accuracy, output_rules, _, _, cover_rate = hyb_model.test(X, y)

    row_list = []
            
    for i in range(len(output_rules)):
        row = {}
        row['accuracy'] = overall_accuracy[i]
        row['transparency'] = cover_rate[i]
        row_list.append(row)
        print("acc:  {}, transp.:  {}".format(str(overall_accuracy[i]), str(cover_rate[i])))
    df_res = pd.DataFrame(row_list)


print("===================>> train perfs")
process(X_train, y_train)


print("===================>> test perfs")
process(X_test, y_test)


