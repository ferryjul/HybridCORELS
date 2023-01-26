import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from rule_mining import mine_rules_preprocessing

import os 
import numpy as np 

random_state_value = 42
test_size_ratio = 0.3

# # Load the new Adult dataset (using the provided folktables module)
# from folktables import ACSDataSource, ACSEmployment, employment_filter


# categories = {
#     "AGEP" : {
#         1.0 : "age_low",
#         2.0 : "age_medium",
#         3.0 : "age_high"
#     },
#     "SCHL" : {
#         1.0: "No schooling completed",
#         2.0: "Nursery school, preschool",
#         3.0: "Kindergarten",
#         4.0: "Grade 1",
#         5.0: "Grade 2",
#         6.0: "Grade 3",
#         7.0: "Grade 4",
#         8.0: "Grade 5",
#         9.0: "Grade 6",
#         10.0: "Grade 7",
#         11.0: "Grade 8",
#         12.0: "Grade 9",
#         13.0: "Grade 10",
#         14.0: "Grade 11",
#         15.0: "12th grade - no diploma",
#         16.0: "Regular high school diploma",
#         17.0: "GED or alternative credential",
#         18.0: "Some college, but less than 1 year",
#         19.0: "1 or more years of college credit, no degree",
#         20.0: "Associate's degree",
#         21.0: "Bachelor's degree",
#         22.0: "Master's degree",
#         23.0: "Professional degree beyond a bachelor's degree",
#         24.0: "Doctorate degree",
#     },
#     "MAR": {
#         1.0: "Married",
#         2.0: "Widowed",
#         3.0: "Divorced",
#         4.0: "Separated",
#         5.0: "Never married or under 15 years old",
#     },
#     "SEX": {1.0: "Male", 2.0: "Female"},
#     "RAC1P": {
#         1.0: "White alone",
#         2.0: "Black or African American alone",
#         3.0: "American Indian alone",
#         4.0: "Alaska Native alone",
#         5.0: (
#             "American Indian and Alaska Native tribes specified;"
#             "or American Indian or Alaska Native,"
#             "not specified and no other"
#         ),
#         6.0: "Asian alone",
#         7.0: "Native Hawaiian and Other Pacific Islander alone",
#         8.0: "Some Other Race alone",
#         9.0: "Two or More Races",
#     },
#     "ESP" : {
#         0.0 : "N/A (not own child of householder, and not child in subfamily)",
#         1.0 : "Living with two parent : both employed",
#         2.0 : "Living with two parent : Father employed",
#         3.0 : "Living with two parent : Mother employed",
#         4.0 : "Living with two parent : None employed",
#         5.0 : "Living with Father : Employed",
#         6.0 : "Living with Father : Not employed",
#         7.0 : "Living with Mother : Employed",
#         8.0 : "Living with Mother : Not employed",
#     },
#     "DIS" : {
#         1.0 : "Disability",
#         2.0 : "No disability"
#     },
#     "NATIVITY" : {
#         1.0 : "Native",
#         2.0 : "Foreign born"
#     },
#     "DREM" : {
#         1.0 : "Cognitive difficulty",
#         2.0 : "No Cognitive difficulty"
#     },
#     "RELP" : {
#         0.0 : "Reference person",
#         1.0 : "Husband/wife",
#         2.0 : "Biological son or daughter",
#         3.0 : "Adopted son or daughter",
#         4.0 : "Stepson or stepdaughter",
#         5.0 : "Brother or sister",
#         6.0 : "Father or mother",
#         7.0 : "Grandchild",
#         8.0 : "Parent-in-law",
#         9.0 : "Son-in-law or daughter-in-law",
#         10.0 : "Other relative",
#         11.0 : "Roomer or boarder",
#         12.0 : "Housemate or roommate",
#         13.0 : "Unmarried partner",
#         14.0 : "Foster child",
#         15.0 : "Other nonrelative",
#         16.0 : "Institutionalized group quarters population",
#         17.0 : "Noninstitutionalized group quarters population"
#     }
# }


# def binarize_age(X):
#     quantiles = np.quantile(X[:, 0], [0, 0.33, 0.66, 1])
#     for q in range(3):
#         index = np.where((quantiles[q] <= X[:, 0]) & (X[:, 0] <= quantiles[q+1]))[0]
#         X[index, 0] = q+1


# def generate_acs_data():
#     data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
#     acs_data = data_source.get_data(states=["TX"], download=True)
#     acs_data = employment_filter(acs_data)
#     features = ACSEmployment.features
#     X, y, _ = ACSEmployment.df_to_numpy(acs_data)
#     binarize_age(X)

#     # Reorder features according to categories dict
#     keep_features = list(categories.keys())
#     keep_features_idx = [ features.index(f) for f in keep_features]
#     X = X[:, keep_features_idx]

#     # OHE the categorical features
#     ohe = OneHotEncoder(sparse=False).fit_transform(X)
#     ohe_features = []
#     for feature_cat in categories.values():
#         ohe_features += list(feature_cat.values())

#     # Save the dataset
#     ohe_df = pd.DataFrame(np.column_stack((ohe, y)).astype(int), 
#                          columns=ohe_features+["Employed"])
#     filename = os.path.join("data", "acs_employ.csv")
#     ohe_df.to_csv(filename, encoding='utf-8', index=False)



def get_data(dataset, splits, max_card=2, min_support=1, n_rules=300, random_state_param=42):
    # # Generate the acs_data if it is not already there
    # if dataset == "acs_employ":# and not os.path.exists(f"data/{dataset}.csv"):
    #     generate_acs_data()

    # Mine the dataset set if it has not already been done
    if not os.path.exists(f"data/{dataset}_mined.csv"):

        df = pd.read_csv(f"data/{dataset}.csv", sep = ',')
        X = df.iloc[:, :-1]
        prediction = df.iloc[:, -1].name
        y = np.array(df.iloc[:, -1])

        # Mine the rules
        X = mine_rules_preprocessing(X, y, max_card, min_support, n_rules)
        features = list(X.columns)
        X = np.array(X)

        # Save the dataset
        df = pd.DataFrame(np.column_stack((X, y)), columns=features+[prediction])
        df.to_csv(f"data/{dataset}_mined.csv", encoding='utf-8', index=False)

    # Rules have already been mined
    else:
        df = pd.read_csv(f"data/{dataset}_mined.csv", sep = ',')
        X = df.iloc[:, :-1]
        features = list(X.columns)
        X = np.array(X)
        prediction = df.iloc[:, -1].name
        y = np.array(df.iloc[:, -1])


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