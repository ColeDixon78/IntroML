#implement a binary decision tree from scratch

import numpy as np
import pandas as pd
import matplotlib as mpl
import sklearn as sk

#classes for construction of tree probably will need to be changed
class Node():
        def __init__(self,dec_key,dec_val):
            self.dec_key = dec_key
            self.dec_val = dec_val
        def right(self,next):
            self.right_branch = next
        def left(self,next):
            self.left_branch = next
        pass
class Leaf():
    def __init__(self,val):
        self.val = val


def calculate_data_entropy(y):
    data_entropy = 0
    column = y.keys()[-1]
    for label in y[column].unique():
        label_prob = (y.value_counts()[label])/len(y)
        data_entropy -= (label_prob) * np.log2(label_prob)
    return data_entropy

def calculate_conditional_entropy(X,y):
    entropy_list = []
    for key in X.keys():
        split = np.mean(X[key])
        X_split = X[key] < split
        cond_entropy = 0
        for value in X_split.unique():
            prob_X = X_split.value_counts()[value]/len(X_split)
            column = y.keys()[-1]
            for label in y[column].unique():
                try:
                    prob_y_cond = (((X_split == value)&(y[column] == label)).value_counts()[True]/len(X_split))/prob_X
                except:
                    prob_y_cond = 0
                cond_entropy -= prob_X * prob_y_cond * np.log2(prob_y_cond)
        entropy_list.append(cond_entropy)
    entropy_list = np.array(entropy_list)
    return entropy_list

def information_gain(X,y):
    #calculate entropy of X data set
    data_entropy = calculate_data_entropy(y)
    #calculate conditional entropy given a split data for each predictor
    cond_entropy_array = calculate_conditional_entropy(X,y)
    #difference in data set entropy and conditional entropies gives us information gain of each split
    information_gain = data_entropy - cond_entropy_array
    return np.argmax(information_gain)


def learn(X,y,impurity_measure = 'entropy'):
    #check if all labels are the same; return leaf of that value if they are
    column = y.keys()[-1]
    if len(y[column].unique()) == 1:
        return Leaf(y[column].iloc[0])
    #check if all data values are the same and return most common label if true
    data_is_equal = True
    for key in X.keys():
        if len(X[key].unique()) == 1:
            continue
        else:
            data_is_equal = False
    if data_is_equal:
        return Leaf(y[column].value_counts()[0])
    
    #find split with most information gain and split data to left and right branches
    else:
        split_index = information_gain(X,y)
        split_key = X.keys()[split_index]
        split_value = np.mean(X[split_key])
        branch = Node(split_key,split_value)
        branch.right = learn(X[X[split_key] < split_value],y[X[split_key] < split_value])
        branch.left = learn(X[X[split_key] >= split_value],y[X[split_key] >= split_value])
        return branch


def predict(x,tree):
    pointer = tree
    while not isinstance(pointer,Leaf):
        dec_val = getattr(pointer,'dec_val')
        dec_key = getattr(pointer,'dec_key')
        if x[dec_key].iloc[0] < dec_val:
            pointer = getattr(pointer,'right_branch')
        else:
            pointer = getattr(pointer,'left_branch')
    return getattr(pointer,'val')

#test code

df = pd.read_csv("wine_dataset.csv")
X = df.iloc[:,:-1]
y = df.iloc[:,-1:]

tree = learn(X,y)
x = pd.DataFrame({"citric acid":[0.13],"residual sugar":[1.6],"pH":[3.34],"sulphates":[0.59],"alcohol":[9.2]})
print(predict(x,tree))
