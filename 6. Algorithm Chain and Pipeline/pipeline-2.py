# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:58:31 2023

@author: hao
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# In[A]: import some data within sklearn for iris classification
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# In[B]: Construct pipeline
pipe_pca_dtc = Pipeline([('scl', StandardScaler()),
			('pca', PCA(n_components=2)),
			('dtc', DecisionTreeClassifier(random_state=42))],
              verbose = True)

param_range = np.linspace(1, 5, 5)

# Set grid search params
grid_params = [{'criterion': ['gini', 'entropy'],
		'min_samples_leaf': param_range,
		'max_depth': param_range,
		'min_samples_split': param_range[1:]}]

# Construct grid search
gs_pca_dtc  = GridSearchCV(estimator = pipe_pca_dtc,
			param_grid = grid_params,
			scoring = 'accuracy',
			cv = 10)

# Fit using grid search
gs_pca_dtc.fit(X_train, y_train)

# In[C] Best accuracy
print('Best accuracy: %.3f' % gs_pca_dtc.best_score_)

# Best params
print('\nBest params:\n', gs_pca_dtc.best_params_)

print(accuracy_score(y_test, gs_pca_dtc.predict(X_test)))


