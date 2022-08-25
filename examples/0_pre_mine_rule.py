
import pandas as pd 
import numpy as np
from scipy.sparse import csc_matrix
from sklearn.ensemble import RandomForestClassifier
from fim import fpgrowth
import itertools
import operator
from scipy.sparse import csc_matrix
import argparse



def generate_rulespace(X, y, max_card=2, method='fpgrowth', random_state=42):

    # Rule Mining
    if method == 'fpgrowth': # Generate rules with fpgrowth

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
        print(len(prules))
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



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help='Dataset name. Options: adult, compas', default='compas')
    parser.add_argument("--method", type=str, help='Rule Mining Method. Options: fpgrowth, rf', default='fpgrowth')
    parser.add_argument("--min_support", type=int, help='Minimum Support in Percentage', default=5)
    parser.add_argument("--max_card", type=int, help='Maximum Length', default=2)
    parser.add_argument("--n_rules", type=int, help='Maximum Number of Rules', default=2000)
    args = parser.parse_args()

    df = pd.read_csv(f"data/{args.dataset}.csv", sep = ',')
    X = df.iloc[:, :-1]
    y = np.array(df.iloc[:, -1])
    prediction_name = df.columns[-1]
    print(X.shape)
    
    X, prules, nrules = generate_rulespace(X, y, max_card=args.max_card, method=args.method)

    prules, pRMatrix = screen_rules(prules, X, N_rules=args.n_rules // 2, min_support=args.min_support)
    nrules, nRMatrix = screen_rules(nrules, X, N_rules=args.n_rules // 2, min_support=args.min_support)

    # Reformat rules
    prules = [" && ".join(rule) for rule in prules]
    nrules = [" && ".join(rule) for rule in nrules]
    
    # Save processed data
    new_columns = prules + nrules + [prediction_name]
    df = pd.DataFrame(np.hstack((pRMatrix, nRMatrix, y.reshape((-1, 1)))), columns=new_columns)
    print(df.shape)
    print(df.head())

    fname = f"{args.dataset}_mined_{args.method}_{args.min_support}_{args.max_card}_{args.n_rules}"
    df.to_csv(f"data/{fname}.csv", encoding='utf-8', index=False)



if __name__ == "__main__":
    main()
