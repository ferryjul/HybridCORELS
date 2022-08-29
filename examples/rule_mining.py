import pandas as pd 
import numpy as np
from scipy.sparse import csc_matrix
from sklearn.ensemble import RandomForestClassifier
from fim import fpgrowth
import itertools
import operator
from scipy.sparse import csc_matrix



def generate_rulespace(X, y, max_card=2, neg_columns=False, method='fpgrowth', random_state=42):
    """
    Generate the set of all rules for a given task

    Arguments
    ---------
    X : pd.DataFrame, shape = [n_samples, n_features]
        The training input samples. All features must be binary.
    y : array-like,  shape = [n_samples,]
        The binary target for the task
    neg_columns : Boolean
        Include the negation of X columns as well
    max_card : int
        The maximal cardinality of the rules
    method : str
        Mining method is either 'rf' of 'fpgrowth'

    Returns
    -------
    X : pd.DataFrame, shape = [n_samples, 2*n_features]
        The training input samples new features representing negation of existing columns.
    prules, n_rules : List(str), lenght = n_rules
        p_rule: Rules associated with positive predictions
        p_rule: Rules associated with negative predictions
    """
    
    # Rule Mining
    if method == 'fpgrowth': # Generate rules with fpgrowth

        if neg_columns:
            # Add negation before applying FPGrowth
            X_neg = 1 - X
            X_neg.columns = ['neg_' + name.strip() for name in X.columns]
            X = pd.concat([X, X_neg], axis=1)

        # Rules frequent with positive and negative predictions
        pindex = np.where(y==1)[0]
        nindex = np.where(y!=1)[0]
        
        # FPGrowth treats each instance as a set of items [[1 2 3], [3 4], [4 6], ...]
        itemMatrix = [[item for item in X.columns if row[item]==1] for _,row in X.iterrows()] 

        prules = fpgrowth([itemMatrix[i] for i in pindex], supp=-1, zmin=1, zmax=max_card)
        prules = [np.sort(x[0]).tolist() for x in prules]
        
        nrules= fpgrowth([itemMatrix[i] for i in nindex], supp=-1, zmin=1, zmax=max_card)
        nrules = [np.sort(x[0]).tolist() for x in nrules]
    
    else: # Using Random Forests to mine rules

        # Rules associated with positive preditions
        prules = []
        for length in range(2, max_card+1, 1):
            n_estimators = 250 * length
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=length, random_state=random_state)
            clf.fit(X, y)
            for n in range(n_estimators):
                prules.extend(extract_rules_rf(clf.estimators_[n], X.columns))

        # Rules associated with negative predictions
        nrules = []
        for length in range(2, max_card, 1):
            n_estimators = 250 * length
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=length, random_state=random_state)
            clf.fit(X, 1-y)
            for n in range(n_estimators):
                nrules.extend(extract_rules_rf(clf.estimators_[n], X.columns))

        # Add negation after fitting the rf
        X_neg = 1 - X
        X_neg.columns = ['neg_' + name.strip() for name in X.columns]
        X = pd.concat([X, X_neg], axis=1)


    return X, prules, nrules




def screen_rules(rules, X, N_rules, min_support):
    """
    Filter rules based on minimum support and return
    new binary data.

    Arguments
    ---------
    X : pd.DataFrame, shape = [n_samples, n_features]
        The training input samples. All features must be binary.
    N_rules : int
        Maximum number of rules to consider
    min_support : int
        Minimum coverage of the rules (in percent e.g. 1, 5, etc.)

    Returns
    -------
    rules : List(str)
        list of all considered rules
    RMatrix : np.array, shape = [n_samples, N_rules]
        Binary matrix indicating if instance i satisfies rule j
    """
    N = X.shape[0]
    min_covered_examples = min_support / 100 * N
    # Map column_name to column index
    itemInd = {}
    for i, name in enumerate(X.columns):
        itemInd[name] = int(i)
    
    len_rules = [len(rule) for rule in rules]
    indices = np.array(list(itertools.chain.from_iterable([[itemInd[x] for x in rule] for rule in rules])))
    indptr = list(accumulate(len_rules))
    indptr.insert(0, 0) # insert 0 at 0 position/necessary for building csc-matrix
    indptr = np.array(indptr)
    data = np.ones(len(indices))
    ruleMatrix = csc_matrix((data,indices,indptr), shape=(len(X.columns), len(rules)))
    
    mat = np.matrix(X) * ruleMatrix # a matrix of data sum wrt rules
    lenMatrix = np.matrix([len_rules for _ in range(X.shape[0])])
    # Z_ij == 1 if instance i respects rule j
    Z = (mat == lenMatrix).astype(int)
    # Coverage of each rule
    C = np.array(np.sum(Z, axis=0))[0]
    
    # Only keep the N_rules rules with sufficient coverage
    supp_select = np.array([i for i in range(len(rules)) if C[i]>min_covered_examples], dtype=np.int32)
    select = np.argsort(C[supp_select])[::-1][:N_rules].tolist()
    ind = list(supp_select[select])
    rules = [rules[i] for i in ind]
    
    RMatrix = np.array(Z[:, ind]) 
    
    return rules, RMatrix



# Implementation from HyRS
# def __screen_rules(self, rules, df, y, criteria='precision'):
    #     """
    #     Return a list of M rules and a (N, M) binary array where
    #     a_ij represents wheter or not instance i respects rule j
    #     """
    #     # Store rules in a sparse matrix
    #     itemInd = {}
    #     for i, name in enumerate(df.columns):
    #         itemInd[name] = int(i)
    #     len_rules = [len(rule) for rule in rules]
    #     indices = np.array(list(itertools.chain.from_iterable([[itemInd[x] for x in rule] for rule in rules])))
    #     indptr = list(accumulate(len_rules))
    #     indptr.insert(0, 0)
    #     indptr = np.array(indptr)
    #     data = np.ones(len(indices))
    #     ruleMatrix = csc_matrix( (data, indices, indptr), shape=(len(df.columns), len(rules)) )

    #     # mat = sparse.csr_matrix.dot(df,ruleMatrix)
    #     # Multiply by the binarized data matrix to see which rules cover which instance
    #     mat = np.matrix(df) * ruleMatrix
    #     lenMatrix = np.matrix([len_rules for _ in range(df.shape[0])])
    #     Z = (mat == lenMatrix).astype(int) # (n_instances, n_rules) binary matrix of cover(R_j, x_i)
    #     Z_support = np.array(np.sum(Z, axis=0))[0] # Number of instances covered by each rule

    #     # Compute the precision of each rule
    #     Zpos = Z[y>0]
    #     TP = np.array(np.sum(Zpos, axis=0))[0]
    #     supp_select = np.where(TP >= self.min_support*sum(y)/100)[0] # Not sure what is going on !!!???
    #     FP = Z_support - TP
    #     precision = TP.astype(float) / (TP + FP)

    #     # Select N rules with highest precision
    #     supp_select = supp_select[precision[supp_select] > np.mean(y)]
    #     select = np.argsort(-precision[supp_select])[:self.n_rules].tolist()
    #     ind = list(supp_select[select])
    #     rules = [rules[i] for i in ind]
    #     RMatrix = np.array(Z[:, ind])

    #     return rules, RMatrix



def extract_rules_rf(tree, feature_names):
    left     = tree.tree_.children_left
    right    = tree.tree_.children_right
    features = [feature_names[i] for i in tree.tree_.feature]
    # Get ids of leaf nodes
    idx = np.argwhere(left == -1)[:, 0]

    def recurse(left, right, child, lineage=None):          
        if lineage is None:
            lineage = []
        if child in left:
            parent = np.where(left == child)[0].item()
            suffix = 'neg_'
        else:
            parent = np.where(right == child)[0].item()
            suffix = ''

        lineage.append((suffix + features[parent].strip()))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)
    
    rules = []
    for child in idx:
        rule = []
        for node in recurse(left, right, child):
            rule.append(node)
        rules.append(rule)
    return rules



def accumulate(iterable, func=operator.add):
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total