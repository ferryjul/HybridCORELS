from __future__ import print_function, division, with_statement
import numpy as np
import pickle

def compute_inconsistent_groups(X, y, verbose=0):
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
    minority_card = []
    majority_card = []
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
                minority_card.append(labelsData[1][0])
                majority_card.append(labelsData[1][1])
                if verbose >= 2:
                    print("Less zeros!")
                    print("associated id label:", indicesInLabels)              
                    print("associated X ids:", indicesX)
            elif labelsData[1][1] == minErrors: # less 1's
                indicesInLabels = np.where((labels == 1))
                indicesX = occurences[0][indicesInLabels]
                minority_card.append(labelsData[1][1])
                majority_card.append(labelsData[1][0])
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
    return np.asarray(representatives), np.asarray(minority_card), np.asarray(majority_card)


def check_array(x, ndim=None):
    if not hasattr(x, 'shape') and \
       (type(x) == str or not hasattr(x, '__len__')) and \
       not hasattr(x, '__array__'):
       raise TypeError("Array must be provided, got: " + str(type(x)))

    x = np.array(x, order='C', copy=False)

    if ndim and ndim != x.ndim:
        raise ValueError("Array must be " + str(ndim) + "-dimensional in shape, got " + str(x.ndim) +
                         " dimensions instead")

    asbool = x.astype(np.bool)

    if not np.array_equal(x, asbool):
        raise ValueError("Array must contain only binary members (0 or 1), got " + str(x));

    return asbool

def check_consistent_length(x, y):
    if x.ndim < 1 or y.ndim < 1:
        raise ValueError("Arrays must have at least one dimension")

    return x.shape[0] == y.shape[0]

def check_is_fitted(o, n):
    if not hasattr(o, n) or not getattr(o, n):
        raise ValueError("Model not fitted yet")

def get_feature(features, i):
    if not features or abs(i) > len(features):
        return ""

    if i < 0:
        return "not " + features[-i - 1]
    else:
        return features[i - 1]
    
def check_in(name, allowed, val):
    if not val.lower() in allowed:
        allowed_str = "'" + "' '".join(allowed) + "'"
        raise ValueError(name + " must be chosen from " + allowed_str +
                         ", got: " + val)

def check_features(features):
    if not isinstance(features, list):
        raise TypeError("Features must be a list, got: " + str(type(features)))
    
    for i in range(len(features)):
        if not isinstance(features[i], str):
            raise TypeError("Each feature much be a string, got: " + str(type(features[i])))

def check_rulelist(rl):
    if not hasattr(rl, "rules") or not hasattr(rl, "features") or not hasattr(rl, "prediction_name"):
        raise ValueError("Rulelist must have the following attributes: 'rules', 'features', 'prediction_name'")

    if not isinstance(rl.rules, list):
        raise TypeError("Rulelist rules must be a list, got: " + str(type(rl.rules)))
    
    if not isinstance(rl.prediction_name, str):
        raise TypeError("Prediction name must be a string, got: " + str(type(rl.prediction_name)))

    check_features(rl.features)
    n_features = len(rl.features)

    if len(rl.rules) < 1:
        raise ValueError("Rulelist must contain at least the default rule")

    for r in range(len(rl.rules)):
        if not isinstance(rl.rules[r], dict):
            raise TypeError("Each rule must be a dict, got: " + str(type(rl.rules[r])))
        
        if not "prediction" in rl.rules[r]:
            raise ValueError("Rule dicts must contain 'prediction' key")
        if not "antecedents" in rl.rules[r]:
            raise ValueError("Rule dicts must contain 'antecedents' key")
            
        if not isinstance(rl.rules[r]["prediction"], (bool, int)):
            raise TypeError("Rule predictions must be bools or ints, got: " + str(type(rl.rules[r]["prediction"])))
        if not isinstance(rl.rules[r]["antecedents"], list): 
            raise TypeError("Rule antecedents must be lists, got: " + str(type(rl.rules[r]["antecedents"])))
        

        a_len = len(rl.rules[r]["antecedents"])
        for i in range(a_len):
            rule = rl.rules[r]["antecedents"][i]
            if not isinstance(rule, int):
                raise TypeError("Rule id must be an int, got: " + str(type(rule)))
            if abs(rule) > n_features:
                raise ValueError("Rule id out of bounds: " + str(rule))

        if r == (len(rl.rules) - 1) and (a_len != 1 or rl.rules[r]["antecedents"][0] != 0):
            raise ValueError("Last rule in the rulelist must be the default prediction,"
                             " with antecedents '[0]', got: " + str(rl.rules[r]["antecedents"]))

class RuleList:
    """This class represents a rulelist. It is used to store the model generated by 
    `CorelsClassifier.fit`.
    
    Attributes
    ----------
    rules : list
        Set of rule indices (into the features array) that comprise the rulelist.
    
    features : list
        Set of all features. An array of strings.
    
    prediction_name : str
        Name of the feature being predicted.
    """

    def __init__(self, rules=[], features=[], prediction_name=""):
        self.rules = rules
        self.features = features
        self.prediction_name = prediction_name

    def save(self, fname):
        """
        Save the rulelist to a file, using python's pickle module.

        Parameters
        ----------
        fname : string
            File name to store the rulelist in
        
        Returns
        -------
        self : obj
        """

        check_rulelist(self)

        with open(fname, "wb") as f:
            pickle.dump({ "f": self.features, "r": self.rules, "p": self.prediction_name }, f)

        return self

    def load(self, fname):
        """
        Load a rulelist from a file, using python's pickle module.
        
        Parameters
        ----------
        fname : string
            File name to load the rulelist from
        
        Returns
        -------
        self : obj
        """

        with open(fname, "rb") as f:
            rl_dict = pickle.load(f)
            if not "r" in rl_dict or not "f" in rl_dict or not "p" in rl_dict:
                raise ValueError("Invalid rulelist file")
            
            rl = RuleList()
            rl.rules = rl_dict["r"]
            rl.features = rl_dict["f"]
            rl.prediction_name = rl_dict["p"]
            check_rulelist(rl)

            self.rules = rl.rules
            self.features = rl.features
            self.prediction_name = rl.prediction_name

        return self

    def __str__(self):
        check_rulelist(self)

        tot = ""#"RULELIST:\n"
        
        if len(self.rules) == 1:
            tot += self.prediction_name + " = " + str(self.rules[0]["prediction"])
        else:    
            for i in range(len(self.rules) - 1):
                feat = get_feature(self.features, self.rules[i]["antecedents"][0])
                for j in range(1, len(self.rules[i]["antecedents"])):
                    feat += " && " + get_feature(self.features, self.rules[i]["antecedents"][j])
                rule_support = self.rules[i]["support"]
                rule_accuracy = self.rules[i]["accuracy"]
                tot += "if [" + feat + "]:\n  " + self.prediction_name + " = " + str(bool(self.rules[i]["prediction"])) + "(support %d, accuracy %.3f)\nelse " %(rule_support, rule_accuracy)

            # HybridCORELS: NO default decision
            #tot += "\n  " + self.prediction_name + " = " + str(bool(self.rules[-1]["prediction"]))


        return tot
    
    def __repr__(self):
        return self.__str__() + "\nAll features: (" + str(self.features) + ")"

def load_from_csv(fname):
    """
    Load a dataset from a csv file. The csv file must contain n_samples+1 rows, each with n_features+1
    columns. The last column of each sample is its prediction class, and the first row of the file
    contains the feature names and prediction class name.
    
    Parameters
    ----------
    fname : str
        File name of the csv data file
    
    Returns
    -------
    X : array-like, shape = [n_samples, n_features]
        The sample data

    y : array-line, shape = [n_samples]
        The target values for the sample data
    
    features : list
        A list of strings of length n_features. Specifies the names of each of the features.

    prediction_name : str
        The name of the prediction class
    """

    import csv
    features = []
    prediction_name = ""

    with open(fname, "r") as f:
        features = f.readline().strip().split(",")
        prediction_name = features[-1]
        features = features[0:-1]

    data = np.genfromtxt(fname, dtype=np.uint8, skip_header=1, delimiter=",")

    X = data[:, 0:-1]
    y = data[:, -1]

    return X, y, features, prediction_name
