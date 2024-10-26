#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engine.imputation import AddMissingIndicator, MeanMedianImputer
from feature_engine.transformation import LogTransformer
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

#%%
dataset_dir = '../datasets/titanic/'
jobs = -1
cross_validation = 10

df = pd.read_csv(dataset_dir + 'train.csv')

# Name is not an important feature in our analysis
# Ticket number may indicate being in the same family, but I'll exclude for simplicity
df.drop(['Name', 'Ticket'], axis=1, inplace=True)

FEATURES = ['Cabin', 'Fare', 'Sex', 'Age', 'Parch']

X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], # predictive variables
    df['Survived'],               # target
    test_size = 0.2,                # portion of dataset to allocate to test set
    random_state = 0,               # we are setting the seed here
)

#%%
class MissingBinaryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables

    # Trivial method to be compatible with Sklearn pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')
            X[feature] = np.where(X[feature] == 'Missing', 0, 1)

        return X
#%%
class GenderBinaryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables

    # Trivial method to be compatible with Sklearn pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature] == 'female', 0, 1)

        return X

#%%
class NonZeroTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables

    # Trivial method to be compatible with Sklearn pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna(int(0))
            X[feature] = np.where(X[feature] < 0.1, 0.1, X[feature])

        return X
#%%
def pipe_data_process():   
    dp = Pipeline([
        # -- IMPUTATION --
        ('missing_imputation', MissingBinaryTransformer(variables=['Cabin'])),
        ('missing_indicator', AddMissingIndicator(missing_only=True, variables=['Age'])),
        ('mean_imputation', MeanMedianImputer(imputation_method='median', variables=['Age'])),
    
        # -- TRANSFORMATION --
        ('gender_binarize', GenderBinaryTransformer(variables=['Sex'])),
        ('fare_nonzero', NonZeroTransformer(variables=['Fare'])),
        ('log', LogTransformer(variables=['Fare'])),
        
        # -- SCALING AND PREDICTION -- 
        ('scaler', MinMaxScaler()),        
        ],
        verbose = True)
    
    return dp
#%%
def pipe_classification():   
    LR_pipe = Pipeline([
        ('LogisticRegession', LogisticRegression()),],
        verbose = True)
    
    RFR_pipe = Pipeline([
        ('RandomForestRegressor', RandomForestRegressor()),],
        verbose = True)
    
    return LR_pipe, RFR_pipe

#%%
def pipe_parm():
    # Set grid search params
    params = {
        'LogisticRegession':{
            'LogisticRegession__penalty': ['l1', 'l2'],
            'LogisticRegession__solver': ['liblinear', 'saga'],
            'LogisticRegession__random_state': [0, 1]},
        'RandomForestRegressor':{
            'RandomForestRegressor__max_features': [3, 4, 5], 
            'RandomForestRegressor__n_estimators': [3, 10, 30],}
    }
    return params

#%%
def plt_keys(pipe, method, X, y):    
    print('Best accuracy : %.3f' % pipe.best_score_)
    print('\nBest params:\n',pipe.best_params_)
    
    result = permutation_importance(pipe, X, y, n_repeats=10,
                                    random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    
    fig, ax = plt.subplots()
    ax.barh(FEATURES + ["age_na"], result.importances[sorted_idx].mean(axis=1).T)
    ax.set_title(f"Permutation Importances of {method}")
    fig.tight_layout()
    plt.show()

#%%
def test_data(pipe):
    new_df = pd.read_csv(dataset_dir +'test.csv')
    new_df.drop(['Name', 'Ticket'], axis=1, inplace=True)
    X_test = new_df[FEATURES]
    y_test =  pd.read_csv(dataset_dir +'gender_submission.csv')['Survived']
    X_valid = pipe_data_process().fit_transform(X_test)

    y_pred_test = pipe.predict(X_valid)
    result = pd.DataFrame(y_pred_test, columns = ["Survived"])
    result['PassengerId'] = new_df['PassengerId']
    result.to_csv('result.csv', index=False)
    
    cm = confusion_matrix(y_test, y_pred_test)
    print(f'confusion matrix {cm}')
    plot_confusion_matrix(cm)
    plt.show()
    
# In[ ]:
if __name__ == '__main__':
    X_train_v = pipe_data_process().fit_transform(X_train)
    X_test_v = pipe_data_process().fit_transform(X_test)
    LR_pipe, RFR_pipe = pipe_classification()
    
    params = pipe_parm()
    method = 'LogisticRegession'
    gs_lr_pipe = GridSearchCV(estimator = LR_pipe, 
                           param_grid = params[method],
                           scoring = 'accuracy',
                           cv = cross_validation)
    gs_lr_pipe.fit(X_train_v, y_train)
    
    plt_keys(gs_lr_pipe, method, X_test_v, y_test)
    test_data(gs_lr_pipe)
    
    method = 'RandomForestRegressor'
    gs_rfr_pipe = GridSearchCV(estimator = RFR_pipe, 
                           param_grid = params[method],
                           cv = cross_validation)
    gs_rfr_pipe.fit(X_train_v, y_train)
    plt_keys(gs_rfr_pipe, method, X_test_v, y_test)
    test_data(gs_rfr_pipe)
