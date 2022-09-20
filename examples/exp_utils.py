import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rule_mining import mine_rules_preprocessing

def get_data(dataset, max_card=2, min_support=1, n_rules=300):
    df = pd.read_csv(f"data/{dataset}.csv", sep = ',')
    X = df.iloc[:, :-1]
    y = np.array(df.iloc[:, -1])
    X = mine_rules_preprocessing(X, y, max_card, min_support, n_rules)
    features = list(X.columns)

    # Generate train and test sets
    random_state_param = 42
    train_proportion = 0.8
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0 - train_proportion, 
                                                        shuffle=True, random_state=random_state_param+1)
    return X_train, X_test, y_train, y_test, features