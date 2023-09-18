#implement a binary decision tree from scratch

import numpy as np
import matplotlib as mpl
import sklearn as sk

"""class Tree:
    def __init__(self):
        pass
    class Node():
        def __init__(self,dec_index,dec_val,prev = None):
            self.prev = prev
            self.dec_index = dec_index
            self.dec_val = dec_val
            pass
        def set_next(self,next):
            self.next = next
        pass
    def __str__(self):
        pass
    def learn(self,X,y,impurity_measure = 'entropy'):
        pass
    def predict(self,x):
        pass
"""
#classes for construction of tree probably will need to be changed
class Node():
        def __init__(self,dec_index,dec_val,prev = None):
            self.prev = prev
            self.dec_index = dec_index
            self.dec_val = dec_val
            pass
        def set_next(self,next):
            self.next = next
        pass
class Leaf():
    def __init__(self,val):
        self.val = val
        pass

def information_gain(X,y):
    #calculate entropy of X data set
    data_entropy = 0
    for label in y:
        label_prob = y.count(label)/len(y)
        data_entropy -= (label_prob) * np.log2(label_prob)
    #calculate conditional entropy given a split data for each predictor
    num_of_features = len(X[0])
    entropy_list = []
    X = np.swapaxes(X) #may or may not need this?? How is the data matrix formatted?
    for feature in range(0,num_of_features):
        mean = np.mean(X[feature])
        cond_entropy = 0
        for label in y:

    #difference in data set entropy and conditional entropies gives us information gain of each split
    pass
def learn(X,y,impurity_measure = 'entropy'):
    #check if all labels are the same; return leaf of that value if they are
    if len(set(y)) == 1:
        return Leaf(y[0])
    #check if all data values are the same and return most common label if true
    elif len(set(X)==1):
        d1 = {}
        for value in y:
            if value not in d1:
                d1[value] = 1
            else:
                d1[value]+=1
        return Leaf(max(d1,key = lambda x:d1[x]))
    else:
        information_list = information_gain(X,y)



def predict(xs,tree):
    pass