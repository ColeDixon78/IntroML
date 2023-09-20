#implement a binary decision tree from scratch

import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn import model_selection


class Tree():
    def __init__(self):
        head = None
    def __str__(self):
        return f'Head: {self.head}'
    #classes for construction of tree
    class Node():
            def __init__(self,dec_key,dec_val):
                self.dec_key = dec_key
                self.dec_val = dec_val
            def __str__(self):
                return f'Decision key: {self.dec_key}, Desicion value: {self.dec_val}\n \tless than: {self.left_branch}\n \tgreater than: {self.right_branch}'
            def set_right(self,next):
                self.right_branch = next
            def set_left(self,next):
                self.left_branch = next
            def set_prev(self,prev):
                self.prev = prev
            pass
    class Leaf():
        def __init__(self,val):
            self.val = val
            pass
        def __str__(self):
            return f'Leaf value: {self.val}'
        def set_prev(self,prev):
            self.prev = prev
        pass

    def impurity(self,X,y,impurity_measure):
        impurity_list = []
        for key in X.keys():
            split = np.mean(X[key])
            X_split = X[key] < split
            impurity = 0
            for value in X_split.unique():
                prob_X = X_split.value_counts()[value]/len(X_split)
                if impurity_measure == 'gini':
                    impurity += prob_X
                column = y.keys()[-1]
                for label in y[column].unique():
                    try:
                        prob_y_cond = (((X_split == value)&(y[column] == label)).value_counts()[True]/len(X_split))/prob_X
                    except:
                        impurity_list.append(0)
                        continue
                    if impurity_measure == 'entropy':
                        impurity -= prob_X * prob_y_cond * np.log2(prob_y_cond)
                    elif impurity_measure == 'gini':
                        impurity -= prob_X * (prob_y_cond**2)
            impurity_list.append(impurity)
        impurity_list = np.array(impurity_list)
        return impurity_list

    def information_gain(self,X,y,impurity_measure):
        #calculate entropy of X data set - not actually necessary can just minimize conditional entropy
        #calculate conditional entropy given a split data for each predictor
        impurity_array = self.impurity(X,y,impurity_measure)
        if impurity_array.all() == 0:
            return self.Leaf(y[y.keys()[-1]].value_counts().keys()[0])
        #difference in data set entropy and conditional entropies gives us information gain of each split
        return np.argmin(impurity_array)

    def learn(self,X,y,impurity_measure = 'entropy',pruning = False):
        if pruning:
            X_training = X.iloc[:len(X)//2,:]
            X_pruning = X.iloc[len(X)//2:,:]
            y_training = y.iloc[:len(y)//2,:]
            y_pruning = y.iloc[len(y//2):,:]
            self.head = self.learn_recursive(X_training,y_training,impurity_measure)
        else:
            self.head = self.learn_recursive(X,y,impurity_measure)

    def learn_recursive(self,X,y,impurity_measure,prev = None):
        #check if all labels are the same; return leaf of that value if they are
        column = y.keys()[-1]
        if len(y[column].unique()) == 1:
            branch = self.Leaf(y[column].iloc[0])
            branch.set_prev(prev)
            return branch
        #check if all data values are the same and return most common label if true
        data_is_equal = True
        for key in X.keys():
            if len(X[key].unique()) == 1:
                continue
            else:
                data_is_equal = False
        if data_is_equal:
            branch = self.Leaf(y[column].value_counts().keys()[0])
            branch.set_prev(prev)
            return branch
        
        #find split with most information gain and split data to left and right branches
        else:
            split_index = self.information_gain(X,y,impurity_measure)
            if isinstance(split_index,self.Leaf):
                split_index.set_prev(prev)
                return split_index
            split_key = X.keys()[split_index]
            split_value = np.mean(X[split_key])
            branch = self.Node(split_key,split_value)
            branch.set_prev(prev)
            branch.set_left(self.learn_recursive(X[X[split_key] < split_value],y[X[split_key] < split_value],impurity_measure))
            branch.set_right(self.learn_recursive(X[X[split_key] >= split_value],y[X[split_key] >= split_value],impurity_measure))
            return branch

    def predict(self,x):
        pointer = self.head
        while not isinstance(pointer,self.Leaf):
            dec_val = pointer.dec_val
            dec_key = pointer.dec_key
            if x[dec_key].iloc[0] < dec_val:
                pointer = pointer.left_branch
            else:
                pointer = pointer.right_branch
        return pointer.val
    

df = pd.read_csv("wine_dataset.csv")
X = df.iloc[:,:-1]
y = df.iloc[:,-1:]
#split data into training, validation, and test sets
#figure out how to set random seed
np.random.seed()
X_training,X_val_test,y_training,y_val_test = model_selection.train_test_split(X,y,train_size = 0.7,test_size=0.3)
X_validation,X_test,y_validation,y_test = model_selection.train_test_split(X_val_test,y_val_test,train_size = 0.5,test_size=0.5)
#training
model_0 = Tree()
model_0.learn(X_training,y_training)
model_1 = Tree()
model_1.learn(X_training,y_training,impurity_measure = 'gini')
#model_2 = Tree()
#model_2.learn(X_training,y_training,pruning = True)
#model_3 = Tree()
#model_3.learn(X_training,y_training,impurity_measure = 'gini',pruning = True)

