# -*- coding: utf-8 -*-
"""
This is the code for paper Companion rule list
We make small modification to unify the API with other models
"""


import numpy as np
import pandas as pd 
from fim import fpgrowth 
import itertools
import random
import operator
from scipy.sparse import csc_matrix
import math
from bitarray import bitarray
import argparse
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier


import warnings
warnings.filterwarnings("ignore")


def my_debug(msg):
    print("====================================> {}".format(msg))

class CRL(object):
    
    def __init__(self, bbox, min_support=0.05, max_card=2, alpha=0.001):
        """Hybrid Rule List/Black-box based classifier.

        This class implements an Hybrid interpretable/black-box model.
        It considers that the black-box model has already been fitted and then fits a rulelist on top of
        it in order to gain free transparency. At inference, instances covered by the rules are labeled
        according to those rules while instances not covered are send to the black box for prediction.

        Attributes
        ----------

        bbox: already fitted classifier

        min_support: minimal support of the rules

        max_card: maximal cardinality of the rules

        alpha: regularizaition parameter

        References
        ----------
        Danqing Pan, Tong Wang, Satoshi Hara. Interpretable Companions for Black-Box Models. 
        In Proceedings of the Twenty Third International Conference on Artificial Intelligence 
        and Statistics, PMLR 108:2444-2454, 2020.
        """
        self.bbox = bbox
        self.min_support = min_support
        self.max_card = max_card
        self.alpha = alpha

    def __screen_rules_2(self, rules, df, y, N, supp):
    
        # This function screen rules by supporting rate
        
        itemInd = {}
        for i,name in enumerate(df.columns):
            itemInd[name] = int(i)
        
        len_rules = [len(rule) for rule in rules]
        indices = np.array(list(itertools.chain.from_iterable([[itemInd[x] for x in rule] for rule in rules])))
        indptr = list(accumulate(len_rules))
        indptr.insert(0,0) # insert 0 at 0 position/necessary for building csc-matrix
        indptr = np.array(indptr)
        data = np.ones(len(indices))
        ruleMatrix = csc_matrix((data,indices,indptr),shape = (len(df.columns),len(rules)))
        
        mat = np.matrix(df)*ruleMatrix # a matrix of data sum wrt rules
        
        lenMatrix = np.matrix([len_rules for i in range(df.shape[0])])
        Z = (mat == lenMatrix).astype(int) # matrix with 0 and 1/'match' rule when 1
        
        Zpos = [Z[i] for i in np.where(np.array(y)>0)][0]
        TP = np.array(np.sum(Zpos,axis=0).tolist()[0])
        
        supp_select = np.where(TP>=supp*sum(y)/100)[0]
        FP = np.array(np.sum(Z,axis = 0))[0] - TP
        p1 = TP.astype(float)/(TP+FP)
        
        supp_select = np.array([i for i in supp_select if p1[i]>np.mean(y)],dtype=np.int32)
        select = np.argsort(p1[supp_select])[::-1][:N].tolist()
        ind = list(supp_select[select])
        rules = [rules[i] for i in ind]
        
        RMatrix = np.array(Z[:,ind]) 
        supp = np.array(np.sum(Z,axis=0).tolist()[0])[ind] # support/number of data covered
        
        return rules, RMatrix, supp, p1[ind], FP[ind]

    
    def __generate_rulespace(self, need_negcode=False):
        if need_negcode:
            df = 1-self.df 
            df.columns = [name.strip() + 'neg' for name in self.df.columns]
            df = pd.concat([self.df,df],axis = 1)
        else:
            df = 1 - self.df
        
        pindex = np.where(self.Y==1)[0]
        nindex = np.where(self.Y!=1)[0]

        itemMatrix = [[item for item in df.columns if row[item] ==1] for i,row in df.iterrows() ]  
        prules= fpgrowth([itemMatrix[i] for i in pindex], supp = self.min_support, zmin = 1,zmax = self.max_card)
        prules = [np.sort(x[0]).tolist() for x in prules]
        nrules= fpgrowth([itemMatrix[i] for i in nindex], supp = self.min_support, zmin = 1, zmax = self.max_card)
        nrules = [np.sort(x[0]).tolist() for x in nrules]
        
        return prules + nrules


    def __screen_rules(self, rules, df, y, criteria='precision'):
        print(rules)
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
        lenMatrix = np.matrix([len_rules for _ in range(df.shape[0])])
        Z = (mat == lenMatrix).astype(int) # (n_instances, n_rules) binary matrix of cover(R_j, x_i)
        Z_support = np.array(np.sum(Z, axis=0))[0] # Number of instances covered by each rule

        # Compute the precision of each rule
        Zpos = Z[y>0]
        TP = np.array(np.sum(Zpos, axis=0))[0]
        supp_select = np.where(TP >= self.min_support*sum(y)/100)[0] # Not sure what is going on !!!???
        FP = Z_support - TP
        precision = TP.astype(float) / (TP + FP)

        # Select N rules with highest precision
        supp_select = supp_select[precision[supp_select] > np.mean(y)]
        select = np.argsort(-precision[supp_select])[:self.n_rules].tolist()
        ind = list(supp_select[select])
        rules = [rules[i] for i in ind]
        RMatrix = np.array(Z[:, ind])

        return rules, RMatrix

    def __propose_rule(self, rule_sequence, premined):
        # This function propose a new rule based on the previous rule
        
        premined_rules = premined.copy()
        rule_seq = rule_sequence.copy()
        
        # No rule reuse
        for i in range(len(rule_seq)):
            premined_rules.remove(rule_seq[i])
        
        rand = random.random()
        
        # We use 4 operations to generate a new rule
        
        if (rand < 0.25):
            #print('add')
            if len(premined_rules)>1:
                # randomly choose a rule in premined rules
                rule_to_add = random.sample(premined_rules,1)[0]
                # insert to a random position in the list
                rule_seq.insert(random.randint(0,len(rule_seq)),rule_to_add)
            
        elif (rand < 0.5):
            #print('remove')
            if len(rule_seq)>1: # at least have 2 rules in the list
                # randomly choose a rule from the list
                rule_to_remove = random.sample(rule_seq,1)[0]
                # remove it
                rule_seq.remove(rule_to_remove)

        elif (rand < 0.75):
            #print('swap')
            if len(rule_seq)>1: # at least have 2 rules in the list
                # randomly choose 2 rules in the list
                swap_num = random.sample(list(range(len(rule_seq))),2)
                # swap them
                rule_seq[swap_num[0]],rule_seq[swap_num[1]] = rule_seq[swap_num[1]],rule_seq[swap_num[0]]

        else:
            #print('replace')
            if (len(rule_seq)>0 and len(premined_rules)>1):
                # randomly choose a rule in the list
                replace_num = random.sample(list(range(len(rule_seq))),1)[0]
                # randomly choose a premined rule
                rule_selected = random.sample(premined_rules,1)[0]
                # replace the rule in the list
                rule_seq[replace_num] = rule_selected
                
        return rule_seq

    def __compute_start(self, new_rule, prev_rule):
        # This function is to find where to start the computation
        # We use this mechanism to speed up the algorithm
        # before start, it is copy
        
        start = 0
        match = False # to check if two rule list have different rule
        len_min = min(len(new_rule),len(prev_rule))
        
        for i in range(len_min):
            if prev_rule[i] != new_rule[i]:
                start = i
                match = True
                break    
        
        if match == False:
            # if no different rule, then the last rule must be removed/added
            start = max(len(new_rule),len(prev_rule))-1
        
        return start

    def __compute_support(self, rule, cover_sets, support_map, cover_map, cov_blx, start):
    
        # this function generates support map and cover map by less computation
        # only copy before start
        
        # new rule, previous support, previous cover and a start point
        # start should not be larger than len(rule_list)-1 or len(support_map)
        
        new_support_map = {} # accumulate set
        new_cover_map = {} # non accumulate set
        new_cov_blx = []
        
        # copy before start
        for i in range(start): 
            
            new_support_map[i] = support_map[i]
            new_cover_map[i] = cover_map[i]
            new_cov_blx.append(cov_blx[i])
        
        # compute after start
        for i in range(start,len(rule)): # only do set computation from start

            if i  == 0:
                new_support_map[i] = cover_sets[rule[i]]
                new_cover_map[i] = cover_sets[rule[i]]
            
            else:
                new_support_map[i] = (cover_sets[rule[i]])|(new_support_map[i-1]) # time cost 3
                new_cover_map[i] = (cover_sets[rule[i]])&~(new_support_map[i-1]) # time cost 4
            
            new_cov_blx.append(~new_support_map[i])
            
        return new_support_map, new_cover_map, new_cov_blx
    
    def __obj_func(self, cov, Acc, Alpha, acc_blx, c_blx):
        
        k = len(cov)
        
        # The objective function
        # cov: the cover rate of each rule
        # Acc: the accuracy of each rule
        # acc_blx: black box accuracy
        # c_blx: the cover rate of black box
        
        acc = [sum([Acc[i]*cov[i] for i in range(j+1)])+acc_blx[j]*c_blx[j] for j in range(k)]
        obj_list = [0.5*(acc[i]+acc[i-1])*cov[i] for i in range(1,k)]
        
        obj = sum(obj_list)+0.5*(acc[0]+self.BLX_ACC)*cov[0]#
        obj = obj - Alpha*k

        return obj

    def __compute_obj(self, new_rule_n, prev_rule_n, support_map, cover_map, cov_blx, c, A, choose):
        # This function compute objective function
        k = len(new_rule_n)
        
        # prev_rule and support_map match
        c_new = [] # cover rate of each rule
        A_new = [] # accuracy of each rule
        choose_new = []

        # find start position by comparing new rule and previous rule

        start_position = self.__compute_start(new_rule_n, prev_rule_n)
        
        new_support_map,new_cover_map,new_cov_blx = self.__compute_support(new_rule_n, self.cover_sets, support_map, cover_map, cov_blx, start_position)

        # copy before start position
        for i in range(0,start_position):
            c_new.append(c[i])
            A_new.append(A[i])
            choose_new.append(choose[i])
            
        # compute after start position
        for i in range(start_position,k):
            
            pos_label_num = sum(self.Y&(new_cover_map[i])) # time cost 1
            covered_num = sum(new_cover_map[i])
                    
            pos_acc = pos_label_num/(covered_num+0.0001)
            neg_acc = (covered_num-pos_label_num)/(covered_num+0.0001)
            
            c_new.append(covered_num/self.X.shape[0])
            
            #A_new.append(max(pos_acc,neg_acc))
            if pos_acc > neg_acc:
                A_new.append(pos_acc)
                choose_new.append(1)
            
            else:
                A_new.append(neg_acc)
                choose_new.append(0)
        
        blx_cov_num = [sum(new_cov_blx[i]) for i in range(len(new_support_map))]
        
        acc_blx = [sum(new_cov_blx[i]&self.Y_c)/(blx_cov_num[i]+0.0001) for i in range(len(new_support_map))]
        c_blx = [blx_cov_num[i]/self.X.shape[0] for i in range(len(new_support_map))]

        # compute object
        obj = self.__obj_func(c_new, A_new, self.alpha, acc_blx, c_blx)
        return obj, c_new, A_new, choose_new, new_support_map, new_cover_map, new_cov_blx
    
    def __simulated_annealing(self, iteration, init_T):
    
        # The main loop of simulated annealing
        
        # temperature
        temperature = init_T
        
        obj_p, chosen, A = 0.1, [], []

        obj_best, cover_best, c_best, A_best = 0, 0, [], []
        rule_best, chosen_best = [], []
        
        # init support map and cover map
        support_map, cover_map, cov_blx = self.__compute_support(self.init_rule, self.cover_sets, {}, {}, [], start=0)

        # init Accuracy, cover and chosen
        c = [sum(cover_map[i])/self.X.shape[0] for i in range(len(self.init_rule))]
        
        for i in range(len(self.init_rule)):
            init_pos_acc = sum(self.Y&(cover_map[i]))/(sum(cover_map[i])+0.001)
            init_neg_acc = (sum(cover_map[i])-sum(self.Y&(cover_map[i])))/(sum(cover_map[i])+0.001)
            
            if init_pos_acc>init_neg_acc:
                A.append(init_pos_acc)
                chosen.append(1)
            else:
                A.append(init_neg_acc)
                chosen.append(0)
        
        prev_rule = self.init_rule
        
        # main iteration
        for t in range(iteration):
            rule_proposed = self.__propose_rule(prev_rule, list(range(len(self.premined_rules))))
            obj_c, c_new, A_new, chosen_new, new_support_map, new_cover_map, new_cov_blx  = self.__compute_obj(rule_proposed, prev_rule, support_map, cover_map, cov_blx, c, A, chosen)
            if obj_c > obj_p:
                obj_p, prev_rule, c, A, support_map, cover_map, cov_blx, chosen = obj_c, rule_proposed, c_new, A_new, new_support_map, new_cover_map, new_cov_blx, chosen_new
                # must check again here
                if obj_c > obj_best:
                    obj_best, c_best, A_best, rule_best, chosen_best = obj_c, c_new, A_new, rule_proposed, chosen_new
                    cover_best = new_support_map[len(new_support_map)-1]
            else:
                rand = random.random()
                accept_rate = math.exp((obj_c-obj_p)/temperature)
                if accept_rate > rand:
                    obj_p, prev_rule, c, A, support_map, cover_map, cov_blx, chosen = obj_c, rule_proposed, c_new, A_new, new_support_map, new_cover_map, new_cov_blx, chosen_new
            temperature = init_T/math.log(2+t)
        return obj_best, c_best, A_best, cover_best, chosen_best, rule_best
    
    def fit(self, X, y, n_iteration=5000, init_temperature=0.001, print_progress=False, random_state=42):

        """
        Build a HyRS from the training set (X, y).

        Parameters
        ----------
        X : pd.DataFrame, shape = [n_samples, n_features]
            The training input samples. All features must be binary.

        y : np.array, shape = [n_samples]
            The target values for the training input. Must be binary.
        
        n_iteration : number of iterations of the local search.

        Returns
        -------
        self : obj
        """
        # Set the seed for reproducability
        random.seed(random_state)

        # Store the training data
        self.df = X
        self.Y = y

        self.premined_rules = self.__generate_rulespace()

        self.Y = bitarray(list(y))
        self.Yb = bitarray(list(self.bbox.predict(X)))
        self.Y_c = ~(self.Y^self.Yb) # to speed up
        self.BLX_ACC = sum(self.Y_c)/len(self.Y) # overall black box accuracy
        self.N = len(y)
        self.X = X
        
        
        self.init_rule = random.sample(list(range(len(self.premined_rules))), 3)

        cover_sets = [np.sum(self.df[self.premined_rules[i]], axis=1)==len(self.premined_rules[i]) for i in range(len(self.premined_rules))]

        self.cover_sets = [bitarray(list(cover_sets[i])) for i in range(len(self.premined_rules))]

        #self.cover_sets = [bitarray(np.sum(self.df[self.premined_rules[i]], axis=1)==len(self.premined_rules[i])) for i in range(10)]

        self.obj, self.c, self.A, self.cover, self.chosen, self.rule = self.__simulated_annealing(n_iteration, init_temperature)

        return self

    def predict_with_type(self, X):
        """
        ToDo
        Predict classifications of the input samples X, along with a boolean (one per example)
        indicating whether the example was classified by the interpretable part of the model or not.

        Arguments
        ---------
        X : pd.DataFrame, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the same
            as those of the data used to train the model.

        Returns
        -------
        ToDo
        """

        test_Yb = self.bbox.predict(X)
        # use test data to test the rule list
        output_rules = [self.premined_rules[self.rule[i]] for i in range(len(self.rule))]
        catch_list = np.array([-1]*X.shape[0])
        # to show observation is caught by which rule
        for i in range(X.shape[0]):
            for j in range(len(output_rules)):
                match = True
                for condition in output_rules[j]:
                    if X.iloc[i][condition] == 0:
                        match = False
                        break
                if match == True:
                    catch_list[i] = j
                    break
        
        test_cover_rate = [0]*len(output_rules)
        blx_cover_rate = [0]*len(output_rules)
        test_acc = []
        blx_acc = []
        
        rule_coverd_set = set()
        blx_cover_set = set(range(X.shape[0]))


        Yhat = []
        covered = []
        
        for i in range(len(output_rules)):
            # observation num caught by rule i
            rule_catch = np.where(catch_list==i)[0]
            # the accumulated rules catch by rule list
            rule_coverd_set = rule_coverd_set.union(set(rule_catch))
            # the left part is then caught by blx model
            blx_cover = blx_cover_set.difference(rule_coverd_set)
            # blx cover rate
            blx_cover_rate[i] = len(blx_cover)/(X.shape[0]+0.0001)
            # cover rate and accuracy of rules
            test_cover_rate[i] = len(rule_catch)/(X.shape[0] + 0.0001)
            
        
        
        
        return output_rules, test_cover_rate,test_acc, list(accumulate(test_cover_rate))
        
    def test(self, test_data, test_label):
        test_Yb = self.bbox.predict(test_data)

        # use test data to test the rule list
        output_rules = [self.premined_rules[self.rule[i]] for i in range(len(self.rule))]
        catch_list = np.array([-1]*test_data.shape[0])
        
        # to show observation is caught by which rule
        for i in range(test_data.shape[0]):
            for j in range(len(output_rules)):
                match = True
                for condition in output_rules[j]:
                    if test_data.iloc[i][condition] == 0:
                        match = False
                        break
                if match == True:
                    catch_list[i] = j
                    break
                    
        test_cover_rate = [0]*len(output_rules)
        blx_cover_rate = [0]*len(output_rules)
        test_acc = []
        blx_acc = []
        
        rule_coverd_set = set()
        blx_cover_set = set(range(test_data.shape[0]))
        
        for i in range(len(output_rules)):
            # observation num caught by rule i
            rule_catch = np.where(catch_list==i)[0]
            # the accumulated rules catch by rule list
            rule_coverd_set = rule_coverd_set.union(set(rule_catch))
            # the left part is then caught by blx model
            blx_cover = blx_cover_set.difference(rule_coverd_set)
            # blx cover rate
            blx_cover_rate[i] = len(blx_cover)/(test_data.shape[0]+0.0001)
            # blx accuracy
            blx_acc.append(sum(test_Yb[list(blx_cover)]==test_label[list(blx_cover)])/(len(blx_cover)+0.0001))
            # cover rate and accuracy of rules
            test_cover_rate[i] = len(rule_catch)/(test_data.shape[0] + 0.0001)
            test_acc.append(sum(test_label[rule_catch] == self.chosen[i])/len(rule_catch))
        
        # the overall accuracy of hybrid models
        test_overall_acc = [sum([test_cover_rate[i]*test_acc[i] for i in range(j+1)])+blx_cover_rate[j]*blx_acc[j] for j in range(len(output_rules))]
        
        return test_overall_acc, output_rules, test_cover_rate, test_acc, list(accumulate(test_cover_rate))




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
