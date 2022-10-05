import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rule_mining import mine_rules_preprocessing



def get_data(dataset, splits, max_card=2, min_support=1, n_rules=300, random_state_param=42):
    df = pd.read_csv(f"data/{dataset}.csv", sep = ',')
    X = df.iloc[:, :-1]
    prediction = df.iloc[:, -1].name

    y = np.array(df.iloc[:, -1])
    X = mine_rules_preprocessing(X, y, max_card, min_support, n_rules)
    features = list(X.columns)

    X = np.array(X)

    # Generate splits
    assert len(splits) <= 3, "We only support splitting the data to up to 3 folds"
    split_names = list(splits.keys())
    split_ratios = list(splits.values())
    assert np.sum(split_ratios) == 1, "The split ratios must sum up to one"
    X_dict = {}
    y_dict = {}
    X_1, X_2, y_1, y_2 = train_test_split(X, y, train_size=split_ratios[0],
                                          shuffle=True, random_state=random_state_param)
    X_dict[split_names[0]] = X_1
    y_dict[split_names[0]] = y_1
    if len(splits) == 2:
        X_dict[split_names[1]] = X_2
        y_dict[split_names[1]] = y_2
    else:
        sub_ratio = split_ratios[1] / (split_ratios[1] + split_ratios[2])
        X_2, X_3, y_2, y_3 = train_test_split(X_2, y_2, train_size=sub_ratio,
                                          shuffle=True, random_state=random_state_param)
        X_dict[split_names[1]] = X_2
        y_dict[split_names[1]] = y_2
        X_dict[split_names[2]] = X_3
        y_dict[split_names[2]] = y_3
    return X_dict, y_dict, features, prediction



def to_df(X, features):
    df_X = {}
    for key, val in X.items():
        df_X[key] = pd.DataFrame(val, columns=features)
    return df_X