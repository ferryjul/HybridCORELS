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

def computeAccuracyUpperBound(X, y, verbose=0):
    import pandas as pd
    import numpy as np
    """
    Parameters
    ----------
    X : Features vector
    y : Labels vector
    verbose : Int
    0 -> No display
    1 -> Minimal Display
    1 -> Debug (also performs additional checks)
    Returns
    -------
    Int, Array, Array
    Int : Minimum number of instances that can not be classified correctly due to dataset inconsistency
    Array of e_r: for each inconsistent group of examples r, e_r is a representative example of this group (its index in X)
    Array of k_r: k_r is the minimum number of instances that can not be classified correctly due to dataset inconsistency, among group r
    Array of i_r: all instances that will be misclassified in the best case (for all inconsistent group, those representing minority for their label)
    """
    representatives = []
    cardinalities = []
    misclassified = []
    values, counts = np.unique(X, axis=0, return_counts=True)
    values = values[counts > 1]
    counts = counts[counts > 1]
    if verbose >= 1:
        print("Found ", values.shape[0], " unique duplicates.")
    incorrCnt = 0
    for ii, anEl in enumerate(list(values)):
        occurences = np.where((X == anEl).all(axis=1))
        representant = occurences[0][0]
        if verbose >= 2:
            print("Value ", anEl, " appears ", counts[ii], " times. (CHECK = ", occurences[0].shape[0], ")")
            print("Occurences: ", occurences, "(representant is instance#", representant, ")")
            # Additional check
            if counts[ii] != occurences[0].shape[0]:
                exit(-1)
        labels = y[occurences[0]]
        if verbose >= 2:
            print(labels)
            # Additional check
            els = X[occurences[0]]
            elsC = np.unique(els, axis=0, return_counts=True)
            if elsC[0].shape[0] > 1:
                exit(-1)
        labelsData = np.unique(labels, return_counts = True)
        if labelsData[0].size > 1:
            if labelsData[0].size != 2: # only two possible values as we work with binary labels -> this case should never happen
                exit(-1)
            minErrors = np.min(labelsData[1])
            if labelsData[1][0] == minErrors: # less 0's
                indicesInLabels = np.where((labels == 0))
                indicesX = occurences[0][indicesInLabels]
                misclassified.extend(indicesX)
                if verbose >= 2:
                    print("Less zeros!")
                    print("associated id label:", indicesInLabels)              
                    print("associated X ids:", indicesX)
            elif labelsData[1][1] == minErrors: # less 1's
                indicesInLabels = np.where((labels == 1))
                indicesX = occurences[0][indicesInLabels]
                misclassified.extend(indicesX)
                if verbose >= 2:
                    print("Less ones!")
                    print("associated id label:", indicesInLabels)     
                    print("associated X ids:", indicesX)
            else:
                print("internal error, exiting")
                exit(-1)
            if verbose >= 2:
                print("min errors possible : ", minErrors)
            incorrCnt += minErrors
            representatives.append(representant)
            cardinalities.append(minErrors)
            #print("Representant = ", representant, ", min errors = ", minErrors)
        else:
            if verbose >= 2:
                print("no inconsistency")
    if verbose >= 1:
        print("At least ", incorrCnt, " elements can not be classified correctly.")
        print("accuracy upper bound = 1 - ", incorrCnt, "/", X.shape[0], " (", 1.0-(incorrCnt/X.shape[0]), ")")        
    return 1.0-(incorrCnt/X.shape[0])