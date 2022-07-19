import pandas as pd 
#from fim import fpgrowth,fim # you can comment this out if you do not use fpgrowth to generate rules
import numpy as np
import itertools
from numpy.random import random
from bisect import bisect_left
from random import sample, seed
from sklearn.metrics import confusion_matrix
import operator
from scipy.sparse import csc_matrix
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class Coverage:
    """ Store results on the Coverage of a RuleSet"""
    p: np.array # Is instance i covered by R+ ?
    n: np.array # Is instance i covered by R- ?
    overlap: np.array   # Is instance i covered by both R+ and R- ?
    pcovered: np.array  # Is instance i ONLY covered by R+ ?
    ncovered: np.array  # Is instance i ONLY covered by R- ?
    covered: np.array   # Which instance are covered by R+ U R- ?


class hyb(object):
    def __init__(self, binary_data, Y, bb_model, alpha=1, beta=0.1):
        self.df = binary_data
        self.bb_model = bb_model
        self.Y = Y
        self.N = float(len(Y))
        self.Yb = bb_model.predict(binary_data)
        self.alpha = alpha
        self.beta = beta


    def generate_rulespace(self, supp, maxlen, N, need_negcode = False, method = 'fpgrowth'):
        if method == 'fpgrowth': # generate rules with fpgrowth
            if need_negcode:
                df = 1-self.df 
                df.columns = [name.strip() + 'neg' for name in self.df.columns]
                df = pd.concat([self.df,df],axis = 1)
            else:
                df = 1 - self.df
            pindex = np.where(self.Y==1)[0]
            nindex = np.where(self.Y!=1)[0]
            itemMatrix = [[item for item in df.columns if row[item] ==1] for i,row in df.iterrows() ]  
            prules= fpgrowth([itemMatrix[i] for i in pindex],supp = supp,zmin = 1,zmax = maxlen)
            prules = [np.sort(x[0]).tolist() for x in prules]
            nrules= fpgrowth([itemMatrix[i] for i in nindex],supp = supp,zmin = 1,zmax = maxlen)
            nrules = [np.sort(x[0]).tolist() for x in nrules]
        else: # if you cannot install the package fim, then use random forest to generate rules
            print('Using random forest to generate rules ...')
            prules = []
            for length in range(2, maxlen + 1, 1):
                n_estimators = 250 * length
                clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=length, random_state=42)
                clf.fit(self.df, self.Y)
                for n in range(n_estimators):
                    prules.extend(extract_rules(clf.estimators_[n], self.df.columns))
            prules = [list(x) for x in set(tuple(np.sort(x)) for x in prules)]
            nrules = []
            for length in range(2,maxlen+1,1):
                n_estimators = 250 * length# min(5000,int(min(comb(df.shape[1], length, exact=True),10000/maxlen)))
                clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=length, random_state=42)
                clf.fit(self.df, 1 - self.Y)
                for n in range(n_estimators):
                    nrules.extend(extract_rules(clf.estimators_[n], self.df.columns))
            nrules = [list(x) for x in set(tuple(np.sort(x)) for x in nrules)]   
            df = 1 - self.df 
            df.columns = [name.strip() + 'neg' for name in self.df.columns]
            df = pd.concat([self.df, df], axis=1)
        self.prules, self.pRMatrix, self.psupp, self.pprecision, self.perror = self.screen_rules(prules, df, self.Y, N, supp)
        self.nrules, self.nRMatrix, self.nsupp, self.nprecision, self.nerror = self.screen_rules(nrules, df, 1-self.Y, N, supp)


    def screen_rules(self, rules, df, y, N, supp, criteria='precision'):
        # Store rules in a sparse matrix
        itemInd = {}
        for i, name in enumerate(df.columns):
            itemInd[name] = int(i)
        len_rules = [len(rule) for rule in rules]
        indices = np.array(list(itertools.chain.from_iterable([[itemInd[x] for x in rule] for rule in rules])))
        indptr = list(accumulate(len_rules))
        indptr.insert(0, 0)
        indptr = np.array(indptr)
        data = np.ones(len(indices))
        ruleMatrix = csc_matrix( (data, indices, indptr), shape=(len(df.columns), len(rules)) )

        # mat = sparse.csr_matrix.dot(df,ruleMatrix)
        # Multiply by the binarized data matrix to see which rules cover which instance
        mat = np.matrix(df) * ruleMatrix
        lenMatrix = np.matrix([len_rules for i in range(df.shape[0])])
        Z = (mat == lenMatrix).astype(int) # (n_instances, n_rules) binary matrix of cover(R_j, x_i)
        Z_support = np.array(np.sum(Z, axis=0))[0] # Number of instances covered by each rule

        # Compute the precision of each rule
        Zpos = Z[y>0]
        TP = np.array(np.sum(Zpos, axis=0))[0]
        supp_select = np.where(TP >= supp*sum(y)/100)[0] # Not sure what is going on !!!???
        FP = Z_support - TP
        precision = TP.astype(float)/(TP + FP)

        # Select N rules with highest precision
        supp_select = supp_select[precision[supp_select] > np.mean(y)]
        select = np.argsort(-precision[supp_select])[:N].tolist()
        ind = list(supp_select[select])
        rules = [rules[i] for i in ind]
        RMatrix = np.array(Z[:, ind])
        #rules_len = [len(set([name.split('_')[0] for name in rule])) for rule in rules]


        return rules, RMatrix, Z_support[ind], precision[ind], FP[ind]


    def train(self, Niteration=5000, interpretability='size', print_progress=False, random_state=42):

        # Set the seed for reproducability
        np.random.seed(random_state)
        seed(random_state)
        
        # Setup
        self.maps = []
        int_flag = int(interpretability =='size')
        T0 = 0.01
        obj_curr = 1000000000
        obj_opt = obj_curr

        # We store the rule sets using their indices
        prs_curr = sample(list( range(len(self.prules)) ), 3)
        nrs_curr = sample(list( range(len(self.nrules)) ), 3)
        self.maps.append([-1, obj_curr, prs_curr, nrs_curr]) #[iter, obj, prs, nrs])

        # Coverage of P-N rules
        coverage_curr = self.compute_rules_coverage(prs_curr, nrs_curr)

        # Compute the loss function
        Yhat_curr, TP, FP, TN, FN  = self.compute_loss(coverage_curr)

        # Count the number of features
        nfeatures = len(np.unique([con.split('_')[0] for i in prs_curr for con in self.prules[i]])) + \
                    len(np.unique([con.split('_')[0] for i in nrs_curr for con in self.nrules[i]]))

        # Compute the objective function
        # New objective function
        o1_curr, o2_curr, o3_curr = self.compute_objective(FN, FP, int_flag, nfeatures, \
                                                            prs_curr, nrs_curr, coverage_curr)
        obj_curr = o1_curr + self.alpha * o2_curr + self.beta * o3_curr
        

        # Main Loop
        self.actions = []
        for iter in tqdm(range(Niteration), disable=print_progress):
            ## What is this????
            #if iter > 0.75 * Niteration:
            #    prs_curr,nrs_curr,pcovered_curr,ncovered_curr,overlap_curr,covered_curr, Yhat_curr = prs_opt[:],nrs_opt[:],pcovered_opt[:],ncovered_opt[:],overlap_opt[:],covered_opt[:], Yhat_opt[:] 
            
            # Propose new RuleSets
            prs_new, nrs_new, coverage_new = self.propose_rs(Yhat_curr, prs_curr, nrs_curr, coverage_curr)
    
            # Compute the new loss
            Yhat_new, TP, FP, TN, FN = self.compute_loss(coverage_new)
            
            # New number of features
            nfeatures = len(np.unique([con.split('_')[0] for i in prs_new for con in self.prules[i]])) + \
                        len(np.unique([con.split('_')[0] for i in nrs_new for con in self.nrules[i]]))
            
            # New objective function
            o1_new, o2_new, o3_new = self.compute_objective(FN, FP, int_flag, nfeatures, \
                                                            prs_new, nrs_new, coverage_new)
            obj_new = o1_new + self.alpha * o2_new + self.beta * o3_new

            # Decrease Temperature
            T = T0 ** (iter / Niteration)
            # Acceptance Probability
            alpha = np.exp(float(obj_curr - obj_new) / T)

            # If ?????
            #if obj_new < self.maps[-1][1]:
            # We decreased the optimal objective
            if obj_new < obj_opt:
                #prs_opt, nrs_opt, obj_opt,pcovered_opt,ncovered_opt,overlap_opt,covered_opt, Yhat_opt = \
                #    prs_new[:],nrs_new[:],obj_new,pcovered_new[:],ncovered_new[:],overlap_new[:],covered_new[:], Yhat_new[:]
                # Update optimal sol
                prs_opt = prs_new
                nrs_opt = nrs_new
                obj_opt = obj_new
                
                #perror, nerror, oerror, berror = self.diagnose(pcovered_new, ncovered_new, overlap_new, covered_new, Yhat_new)
                #accuracy_min = float(TP+TN)/self.N
                #explainability_min = sum(covered_new)/self.N
                #covered_min = covered_new
            
            #if print_message:
            #    perror, nerror, oerror, berror = self.diagnose(pcovered_new,ncovered_new,overlap_new,covered_new,Yhat_new)
            #    print('\niter = {}, alpha = {}, {}(obj) = {}(error) + {}(intepretability) + {}(exp)\n accuracy = {}, explainability = {}, nfeatures = {}\n perror = {}, nerror = {}, oerror = {}, berror = {}\n '.format(iter,round(alpha,2),round(obj_new,3),(FP+FN)/self.N, self.alpha*(len(prs_new) + len(nrs_new)), self.beta*sum(~covered_new)/self.N, (TP+TN+0.0)/self.N,sum(covered_new)/self.N, nfeatures,perror,nerror,oerror,berror ))
            #    print('prs = {}, nrs = {}'.format(prs_new, nrs_new))

            # Accept the change is probability alpha
            if random() <= alpha:
                # Update current solution
                prs_curr = prs_new
                nrs_curr = nrs_new
                obj_curr = obj_new
                coverage_curr = coverage_new
                Yhat_curr = Yhat_new
                self.maps.append([iter, obj_new, prs_new, nrs_new])

                if print_progress:
                    print(f"\n** iter={iter} ** \n {obj_new:.3f}(obj) = " +\
                        f"{o1_new:.3f}(error) + alpha * {o2_new:d}(interp) + beta * {o3_new:.3f}(1-coverage)")
        
        # Save the optimal rule sets as attributes
        self.positive_rule_set = prs_opt
        self.negative_rule_set = nrs_opt



    def compute_rules_coverage(self, prs, nrs):
        p = np.sum(self.pRMatrix[:, prs], axis=1) > 0  # Is instance i covered by R+ ?
        n = np.sum(self.nRMatrix[:, nrs], axis=1) > 0  # Is instance i covered by R- ?
        overlap = np.multiply(p, n)    # Is instance i covered by both R+ and R- ?
        pcovered = p ^ overlap         # Is instance i ONLY covered by R+ ?
        ncovered = n ^ overlap         # Is instance i ONLY covered by R- ?
        covered  = np.logical_or(p, n) # Which instance are covered by R+ U R- ?
        assert sum(overlap) <= sum(covered)
        return Coverage(p, n, overlap, pcovered, ncovered, covered)


    # def diagnose(self, pcovered, ncovered, overlapped, covered, Yhat):
    #     perror = sum(self.Y[pcovered]!=Yhat[pcovered])
    #     nerror = sum(self.Y[ncovered]!=Yhat[ncovered])
    #     oerror = sum(self.Y[overlapped]!=Yhat[overlapped])
    #     berror = sum(self.Y[~covered]!=Yhat[~covered])
    #     return perror, nerror, oerror, berror


    def compute_loss(self, coverage):
        # pcovered : is x covered by R+ ?
        # covered : is x covered by R+ U R- ?
        Yhat = np.zeros(int(self.N))
        Yhat[coverage.pcovered] = 1
        Yhat[~coverage.covered] = self.Yb[~coverage.covered]
        TN, FP, FN, TP = confusion_matrix(Yhat, self.Y).ravel()
        return  Yhat, TP, FP, TN, FN


    def compute_objective(self, FN, FP, int_flag, nfeatures, prs, nrs, coverage):
        return (FN + FP) / self.N, \
                (int_flag *(len(prs) + len(nrs)) + (1 - int_flag) * nfeatures),\
                sum(~coverage.covered) / self.N


    def propose_rs(self, Yhat, prs, nrs, coverage):# vt, print_message = False):
        # Error because of the interpretable model
        incorr = np.where(Yhat[coverage.covered] != self.Y[coverage.covered])[0]
        # Error because of the blackbox model
        incorrb = np.where(Yhat[~coverage.covered] != self.Y[~coverage.covered])[0]
        ex = -1
        # Covering all examples
        if sum(coverage.covered) == self.N:
            move = ['cut']
            self.actions.append(0)
            if len(prs) == 0:
                sign = [0]
            elif len(nrs) == 0:
                sign = [1]
            else:
                sign = [int(random()<0.5)]
        # Covering no examples
        if sum(coverage.covered) == 0:
            move = ['extend']
            self.actions.append(1)
            sign = [int(random()<0.5)]
        # # What the heck ???
        # elif len(incorr) == 0 and (len(incorrb)==0 or sum(coverage.overlap)==self.N): #\or sum(coverage.overlap) > sum(coverage.covered):
        #     self.actions.append(1)
        #     move = ['cut']
        #     sign = [int(random()<0.5)]
        # elif (len(incorr) == 0 and (sum(covered)>0)) or len(incorr)/sum(covered) >= len(incorrb)/sum(~covered):
        #     if print_message:
        #         print(' ===== 2 ===== ')
        #     self.actions.append(2)
        #     ex = sample(list(np.where(~covered)[0]) + list(np.where(overlapped)[0]),1)[0] 
        #     if overlapped[ex] or len(prs) + len(nrs) >= (vt + self.beta)/self.alpha:
        #         # print('2')
        #         move = ['cut']
        #         sign = [int(random()<0.5)]
        #     else:
        #         # print('3')
        #         move = ['expand']
        #         sign = [int(random()<0.5)]
        else:
            # if sum(overlapped)/sum(pcovered)>.5 or sum(overlapped)/sum(ncovered)>.5:
            #     if print_message:
            #         print(' ===== 3 ===== ')
            #     # print('4')
            #     move = ['cut']
            #     sign = [int(len(prs)>len(nrs))]
            # else:  
            t = random()
            if t < 1./3: # Try to decrease errors
                self.actions.append(3)
                ex = sample(list(incorr) + list(incorrb), 1)[0]
                if ex in incorr: # Incorrectly classified by the interpretable model
                    if self.Y[ex] == 0: # Negative example
                        move = ['cut']  # Remove the positive rule that covers it
                        sign = [1]
                    else: # Positive example
                        if random() < 0.5: # Add a rule to cover it
                            move = ['add']
                            sign = [1]
                        else: # Remove a negative rule that covers it
                            move = ['cut']
                            sign = [0]
                    #rs_indicator = (coverage.pcovered[ex]).astype(int) # covered by prules
                    #if random() < 0.5:
                    #    move = ['cut']
                    #    sign = [rs_indicator]
                    #else:
                    #    move = ['cut', 'add']
                    #    sign = [rs_indicator, rs_indicator]
                # elif overlapped[ex]: 
                #     if random()<0.5 :
                #         # print('5')
                #         move = ['cut']
                #         sign = [1 - self.Y[ex]]
                #     else:
                #         # print('6')
                #         move = ['cut','add']
                #         sign = [1 - self.Y[ex],1 - self.Y[ex]]
                else: # Incorrectly classified by the black box model
                    move = ['add']
                    sign = [int(self.Y[ex]==1)] # Add a consistent rule that covers it
            elif t < 2./3: # decrease coverage
                self.actions.append(4)
                move = ['cut']
                sign = [round(random())]
            else: # increase coverage
                self.actions.append(5)
                move = ['expand']
                sign = [round(random())]
                # if random()<0.5:
                #     move.append('add')
                #     sign.append(1-rs_indicator)
                # else:
                #     move.extend(['cut','add'])
                #     sign.extend([1-rs_indicator,1-rs_indicator])

        # Update the sets R+ and R-
        #for j in range(len(move)):
        j = 0
        if sign[j] == 1:
            prs_new = self.action(move[j], 1, ex, prs, Yhat, coverage.pcovered)
            nrs_new = nrs
        else:
            prs_new = prs
            nrs_new = self.action(move[j], 0, ex, nrs, Yhat, coverage.ncovered)

        coverage_new = self.compute_rules_coverage(prs_new, nrs_new)

        return prs_new, nrs_new, coverage_new


    def action(self, move, rs_indicator, ex, rules, Yhat, covered):
        RMatrix = self.pRMatrix if rs_indicator else self.nRMatrix
        Y = self.Y if rs_indicator else 1 - self.Y

        # Removing a rule
        if move == 'cut' and len(rules) > 0:
            # Remove a rule to decrease the error at ex
            if random() < 0.25 and ex >= 0:
                # Remove a rule that covers ex
                candidate = list(set(np.where(RMatrix[ex, :]==1)[0]).intersection(rules))
                if len(candidate) == 0:
                    candidate = rules
                cut_rule = sample(candidate, 1)[0]
            # Remove any rule
            else:
                p = []
                all_sum = np.sum(RMatrix[:, rules], axis=1)
                # Compute the precision of the RuleSet if we remove a rule
                for index, rule in enumerate(rules):
                    Yhat = ((all_sum - np.array(RMatrix[:, rule])) > 0).astype(int)
                    TP, FP, TN, FN = confusion_matrix(Yhat, Y).ravel()
                    p.append(TP.astype(float) / (TP + FP))
                p = np.exp(-np.array([x - min(p) for x in p]))
                p = np.insert(p, 0, 0)
                p = np.array(list(accumulate(p)))
                # Sample rules with Softmax prob based on precision
                if p[-1] == 0:
                    cut_rule = sample(rules, 1)[0]
                else:
                    p = p / p[-1]
                    index = find_lt(p, random())
                    cut_rule = rules[index]
            # Remove the selected rule
            return_rules = [rule for rule in rules if not rule == cut_rule]

        # WHAT ???!!!
        elif move == 'add' and ex >= 0:
            score_max = -self.N * 10000000
            if self.Y[ex] * rs_indicator + (1 - self.Y[ex]) * (1 - rs_indicator) == 1:
                # select = list(np.where(RMatrix[ex] & (error +self.alpha*self.N < self.beta * supp))[0]) # fix
                select = list(np.where(RMatrix[ex])[0])
            else:
                # select = list(np.where( ~RMatrix[ex]& (error +self.alpha*self.N < self.beta * supp))[0])
                select = list(np.where(~RMatrix[ex])[0])
            self.select = select
            if len(select) > 0:
                if random() < 0.25:
                    add_rule = sample(select, 1)[0]
                else: 
                    # cover = np.sum(RMatrix[(~covered)&(~covered2), select],axis = 0)
                    # =============== Use precision as a criteria ===============
                    # Yhat_neg_index = np.where(np.sum(RMatrix[:,rules],axis = 1)<1)[0]
                    # mat = np.multiply(RMatrix[Yhat_neg_index.reshape(-1,1),select].transpose(),Y[Yhat_neg_index])
                    # TP = np.sum(mat,axis = 1)
                    # FP = np.array(np.sum(RMatrix[Yhat_neg_index.reshape(-1,1),select],axis = 0) - TP)
                    # TN = np.sum(Y[Yhat_neg_index]==0)-FP
                    # FN = sum(Y[Yhat_neg_index]) - TP
                    # p = (TP.astype(float)/(TP+FP+1)) + self.alpha * supp[select]
                    # add_rule = select[sample(list(np.where(p==max(p))[0]),1)[0]]
                    # =============== Use objective function as a criteria ===============
                    # What is going on here ???
                    for ind in select:
                        z = np.logical_or(RMatrix[:, ind], Yhat)
                        TN, FP, FN, TP = confusion_matrix(z, self.Y).ravel()
                        score = FP + FN - self.beta * sum(RMatrix[~covered ,ind])
                        if score > score_max:
                            score_max = score
                            add_rule = ind
            return_rules = rules + [add_rule]
        
        # Expand by adding any rule
        else:
            candidates = [x for x in range(RMatrix.shape[1])]
            select = list(set(candidates).difference(rules))
            if random() < 0.25:
                add_rule = sample(select, 1)[0]
            else:
                # What is going on in here !!!????
                # Yhat_neg_index = np.where(np.sum(RMatrix[:,rules],axis = 1)<1)[0]
                Yhat_neg_index = np.where(~covered)[0]
                mat = np.multiply(RMatrix[Yhat_neg_index.reshape(-1, 1),select].transpose(), Y[Yhat_neg_index])
                # TP = np.array(np.sum(mat,axis = 0).tolist()[0])
                TP = np.sum(mat, axis=1)
                FP = np.array(np.sum(RMatrix[Yhat_neg_index.reshape(-1, 1), select], axis=0) - TP)
                TN = np.sum(Y[Yhat_neg_index]==0) - FP
                FN = sum(Y[Yhat_neg_index]) - TP
                score = (FP + FN) + self.beta * (TN + FN)
                # score = (TP.astype(float)/(TP+FP+1)) + self.alpha * supp[select] # using precision as the criteria
                add_rule = select[sample(list(np.where(score==min(score))[0]), 1)[0]] 
            
            return_rules = rules + [add_rule]
        return return_rules


    def predict(self, df, Y):

        # Black box model
        Yb = self.bb_model.predict(df)

        prules = [self.prules[i] for i in self.positive_rule_set]
        nrules = [self.nrules[i] for i in self.negative_rule_set]

        # if isinstance(self.df, scipy.sparse.csc.csc_matrix)==False:
        dfn = 1 - df #df has negative associations
        dfn.columns = [name.strip() + 'neg' for name in df.columns]
        df_test = pd.concat([df, dfn], axis=1)

        # Interpretable model
        if len(prules):
            # Does R+ cover these instances
            p = [[] for _ in prules]
            for i, rule in enumerate(prules):
                p[i] = (np.sum(df_test[list(rule)], axis=1)==len(rule)).astype(int)
            p = (np.sum(p, axis=0) > 0).astype(int)
        else:
            p = np.zeros(len(Y))
        if len(nrules):
            # Does R- cover these instances
            n = [[] for _ in nrules]
            for i, rule in enumerate(nrules):
                n[i] = (np.sum(df_test[list(rule)], axis=1)==len(rule)).astype(int)
            n = (np.sum(n, axis=0) > 0).astype(int)
        else:
            n = np.zeros(len(Y))
        pind = list(np.where(p)[0])
        nind = list(np.where(n)[0])
        covered = np.logical_or(p, n)

        # Black box model
        Yhat = Yb
        Yhat[nind] = 0
        Yhat[pind] = 1
        return Yhat, covered


def accumulate(iterable, func=operator.add):
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total


def find_lt(a, x):
    """ Find rightmost value less than x"""
    i = bisect_left(a, x)
    if i:
        return int(i-1)
    else:
        return 0


def extract_rules(tree, feature_names):
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
            suffix = 'neg'
        else:
            parent = np.where(right == child)[0].item()
            suffix = ''

        lineage.append((features[parent].strip() + suffix))

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
