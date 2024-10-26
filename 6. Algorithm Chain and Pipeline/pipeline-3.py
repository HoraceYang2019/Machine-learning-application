# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:58:31 2023
advanced:
    How to keep feature names in sklearn Pipeline
https://medium.com/@anderson.riciamorim/how-to-keep-feature-names-in-sklearn-pipeline-e00295359e31
@author: hao
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# In[A]: import some data within sklearn for iris classification
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
seed_no = 42
key_componments = 2
cross_validation = 10

# In[B.1]: Construct pipeline
pipe_pca_dtc = Pipeline([('scl', StandardScaler()),
			('pca', PCA(n_components = key_componments)),
			('clf', DecisionTreeClassifier(random_state = seed_no))],
              verbose = True)

pipe_rfc = Pipeline([('scl', StandardScaler()),
			('clf', RandomForestClassifier(random_state = seed_no))])

pipe_pca_rfc = Pipeline([('scl', StandardScaler()),
			('pca', PCA(n_components = key_componments)),
			('clf', RandomForestClassifier(random_state = seed_no))])

# In[B.2]: parameter ranges for searching
param_range = np.linspace(1, 3, 3).astype(int)

# Set grid search params
params_dtc = [{'clf__criterion': ['gini', 'entropy'],
		'clf__min_samples_leaf': param_range,
		'clf__max_depth': param_range,
		'clf__min_samples_split': param_range[1:]}]

params_rfc = [{'clf__criterion': ['gini', 'entropy'],
		'clf__min_samples_leaf': param_range,
		'clf__max_depth': param_range,
		'clf__min_samples_split': param_range[1:]}]


# In[B.3]: searching method- grid search
jobs = -1
gs_pca_dtc  = GridSearchCV(estimator = pipe_pca_dtc,
			param_grid = params_dtc,
			scoring = 'accuracy',
			cv = cross_validation)

gs_rfc = GridSearchCV(estimator = pipe_rfc,
			param_grid = params_rfc,
			scoring='accuracy',
			cv = cross_validation, 
			n_jobs = jobs)

gs_pca_rfc = GridSearchCV(estimator = pipe_pca_rfc,
			param_grid = params_rfc,
			scoring='accuracy',
			cv = cross_validation, 
			n_jobs = jobs)

# In[B.4] Fit models using the searching method
gs_pca_dtc.fit(X_train, y_train)
gs_rfc.fit(X_train, y_train)
gs_pca_rfc.fit(X_train, y_train)

# In[C] Best accuracy
print(f'pcs-dtc: {accuracy_score(y_test, gs_pca_dtc.predict(X_test))}')
print(f'rfc: {accuracy_score(y_test, gs_rfc.predict(X_test))}')
print(f'pcs-rfc: {accuracy_score(y_test, gs_pca_rfc.predict(X_test))}')
