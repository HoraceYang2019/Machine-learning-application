# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:58:31 2023

@author: hao
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

# In[A]: import some data within sklearn for iris classification
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Splitting data into train and testing part
# The 25 % of data is test size of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)
# importing pipes for making the Pipe flow

# In[B]:pipe flow
# PCA(Dimension reduction to two) -> Scaling the data -> DecisionTreeClassification
pipe = Pipeline([ ('pca', PCA(n_components = 2)), 
                 ('std', StandardScaler()),
                 ('decision_tree', DecisionTreeClassifier(random_state=42))
                 ],
                verbose = True)

# fitting the data in the pipe
tree = pipe.fit(X_train, y_train)

# In[C]: scoring data
print(accuracy_score(y_test, tree.predict(X_test)))

# In[]
tree.plot_tree(tree)
plt.show()