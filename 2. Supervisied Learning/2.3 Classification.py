#!/usr/bin/env python
# coding: utf-8

# ## 2.3 Supervised Learning- Classification

# * A. k-Nearest Neighbors: Machine Learning Basics with the K-Nearest Neighbors Algorithm, 
#         https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
# * B. Linear Regression 
# * C. Naive Bayes Classifiers: Naive Bayes Classifier
#         https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c
# * D. Decision trees: Decision Trees Explained — Entropy, Information Gain, Gini Index, CCP Pruning
#         https://towardsdatascience.com/decision-trees-explained-entropy-information-gain-gini-index-ccp-pruning-4d78070db36c
# * E. Ensembles of Decision Trees: Random forests, Understanding Random Forest
#         https://towardsdatascience.com/understanding-random-forest-58381e0602d2
# * Feature visualization
#         https://medium.com/ai-academy-taiwan/%E5%8F%AF%E8%A7%A3%E9%87%8B-ai-xai-%E7%B3%BB%E5%88%97-shap-2c600b4bdc9e

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from warnings import simplefilter
# simplefilter(action='ignore', category=FutureWarning)


# In[2]:


from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print(f"cancer.keys():{cancer.keys()}\n")
print(f"Shape of cancer data: {cancer.data.shape}\n")
print("Sample counts per class:", {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})


# In[3]:


print("Feature names:\n", cancer.feature_names)

# !pip install pandasgui
# In[4]:


from pandasgui import show

cancer_df = pd.DataFrame(cancer.data, columns = cancer.feature_names)
show(cancer_df)


# In[5]:


import plotly.express as px

fig = px.density_contour(data_frame=cancer_df, x='mean smoothness', y=
    'mean radius', z='mean concave points', facet_row=None, facet_col=None,
    )
fig.update_traces(contours_coloring='fill', contours_showlabels=True)
show(fig)


# ### <font color='blue'>  A. k-Nearest Neighbors: k-Neighbors classification </font>

# This visualization help understand how k-Nearest Neighbors work. Given a k value, what will be the prediction?
# 
# In the k=3 circle, green is the majority, new data points will be predicted as green;<br>
# In the k=6 circle, blue is the majority, new data points will be predicted as blue;

# <div>
# <img src="attachment:image.png" width="300"/>
# </div>

# In[6]:


from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=3, metric='minkowski')


# Advantages: The k-Nearest Neighbors algorithm is simple to implement and robust to noisy training data.
# 
# Disadvantages: High cost of computation compared to other algorithms. Storage of data: memory based, so less efficient. Need to define which k value to use.

# ##### Analyzing KNeighborsClassifier
# KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None) <br>
#     * weights{‘uniform’, ‘distance’} <br>
#     * algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}

# In[7]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
   
    clf = KNeighborsClassifier(n_neighbors=n_neighbors) # build the model
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train)) # record training set accuracy
    test_accuracy.append(clf.score(X_test, y_test)) # record generalization accuracy
    
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# * Advantages
#     * The algorithm is simple and easy to implement.
#     * There’s no need to build a model, tune several parameters, or make additional assumptions.
#     * The algorithm is versatile. It can be used for classification, regression, and search.
# 
# * Disadvantages
#     * The algorithm gets significantly slower as the number of examples and/or predictors/independent variables increase.

# ### <font color='blue'> B.Linear Models: Linear models for regression </font>
# Linear Regression cost function
# \begin{alignat}{3}
# J(\theta) = \text{MSE} (\theta) =  \frac{1}{m} \sum_{i=1}^{m} (\theta^T \cdot x^{(i)} - y^{(i)})^2
# \end{alignat}

# #### Linear regression aka ordinary least squares
from sklearn.datasets import fetch_openml
house = fetch_openml(name="house_prices", as_frame=True)
X, y = house.data, house.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# In[8]:


data_url = "http://lib.stat.cmu.edu/datasets/boston"
boston = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([boston.values[::2, :], boston.values[1::2, :2]])
y = boston.values[1::2, 2]


# * Features of Boston housing data
#     * CRIM:     per capita crime rate by town
#     * ZN:       proportion of residential land zoned for lots over 25,000 sq.ft.
#     * INDUS    proportion of non-retail business acres per town
#     * CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#     * NOX      nitric oxides concentration (parts per 10 million)
#     * RM       average number of rooms per dwelling
#     * AGE      proportion of owner-occupied units built prior to 1940
#     * DIS      weighted distances to five Boston employment centres
#     * RAD      index of accessibility to radial highways
#     * TAX      full-value property-tax rate per 10,000
#     * PTRATIO  pupil-teacher ratio by town
#     * B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#     * LSTAT    % lower status of the population
#     * MEDV     Median value of owner-occupied homes in 1000

# In[9]:


features = ['CRIM', 'ZN','INDUS', 'CHAS','NOX','RM','AGE','DIS', 'RAD','TAX','PTRATIO','B','LSTAT', 'MEDV']
boston_df = pd.DataFrame(np.c_[X,y], columns = features)
show(boston_df)


# In[10]:


from sklearn.linear_model import LinearRegression

X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

lr = LinearRegression().fit(X_train, y_train)

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))


# #### Ridge regression
# Ridge Regression cost function
# \begin{alignat}{3}
# J(\theta) = \text{MSE} (\theta) + \alpha \frac{1}{2} \sum_{i=1}^{n} \theta_i^2
# \end{alignat}

# In[11]:


from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))


# In[12]:


ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))


# In[13]:


ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))


# In[14]:


plt.title('Boston Housing Prices')
plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")

plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-25, 25)
plt.legend()


# #### Lasso
# Lasso Regression cost function
# \begin{alignat}{3}
# J(\theta) = \text{MSE} (\theta) + \alpha \sum_{i=1}^{n} \lvert \theta_i \rvert
# \end{alignat}

# In[15]:


from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used:", np.sum(lasso.coef_ != 0))


# In[16]:


# we increase the default setting of "max_iter",
# otherwise the model would warn us that we should increase max_iter.
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used:", np.sum(lasso001.coef_ != 0))


# In[17]:


lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used:", np.sum(lasso00001.coef_ != 0))


# In[18]:


plt.title('Boston Housing Prices')
plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")

plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend() #plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.show()


# ##### Linear models for classification
# * default (penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None) <br>
#     *  penalty{‘l1’, ‘l2’, ‘elasticnet’, None}: 
#         * None: no penalty is added;
#         * 'l2': add a L2 penalty term and it is the default choice;
#         * 'l1': add a L1 penalty term;
#         * 'elasticnet': both L1 and L2 penalty terms are added.
#     * dual: Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.
#     * tol: Tolerance for stopping criteria.
#     * C: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
#     * max_iter, default=100, Maximum number of iterations taken for the solvers to converge.    

# In[19]:


from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)


# In[20]:


logreg = LogisticRegression(max_iter=10000).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))


# In[21]:


logreg100 = LogisticRegression(max_iter=10000, C=100).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))


# In[22]:


logreg001 = LogisticRegression(max_iter=10000,C=0.01).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))


# In[23]:


plt.title('Breast Cancer')
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend()


# In[24]:


for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(max_iter=10000, C=C, solver='liblinear', penalty="l1").fit(X_train, y_train)
    print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
          C, lr_l1.score(X_train, y_train)))
    print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
          C, lr_l1.score(X_test, y_test)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))

plt.title('Breast Cancer')
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")

plt.ylim(-5, 5)
plt.legend(loc=3)


# ##### Linear models for multiclass classification

# ### <font color='blue'> C. Naive Bayes Classifiers </font>
# https://www.geeksforgeeks.org/naive-bayes-classifiers/ <br>
# * Naïve Bayes algorithm is a supervised learning algorithm, which is based on Bayes theorem and used for solving classification problems.
#     * It is mainly used in text classification that includes a high-dimensional training dataset.
#     * Naïve Bayes Classifier is one of the simple and most effective Classification algorithms which helps in building the fast machine learning models that can make quick predictions.
#     * It is a probabilistic classifier, which means it predicts on the basis of the probability of an object.
#     * Some popular examples of Naïve Bayes Algorithm are spam filtration, Sentimental analysis, and classifying articles.

# In[25]:


# training the model on training set
from sklearn.naive_bayes import GaussianNB

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
  
# making predictions on the testing set
y_pred = gnb.predict(X_test)
  
# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics

print(f"Gaussian Naive Bayes model accuracy of cancer data (in %): {metrics.accuracy_score(y_test, y_pred)*100:.3f}")


# ### <font color='blue'> D. Decision trees </font>
# * What Is a Decision Tree? <br>
#     * Simply speaking, the decision tree algorithm breaks the data points into decision nodes resulting in a tree structure.
#     * The decision nodes represent the question based on which the data is split further into two or more child nodes. 
#     * The tree is created until the data points at a specific child node is pure (all data belongs to one class). 
#     * The criteria for creating the most optimal decision questions is the information gain. 

# <div>
# <img src="attachment:image.png" width="500"/>
# </div>

# * Training a machine learning model using a decision tree classification algorithm is about finding the decision tree boundaries.
#     * Decision trees build complex decision boundaries by dividing the feature space into rectangles. 
#     * Here is a sample of how decision boundaries look like after model trained using a decision tree algorithm classifies the Sklearn IRIS data points. 
#     * The feature space consists of two features namely petal length and petal width. 

# In[26]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
X = iris.data[:, 2:]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
clf_tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
clf_tree.fit(X_train, y_train)


# <div>
# <img src="attachment:image-2.png" width="500"/>
# </div>

# In[27]:


from sklearn import tree
fig, ax = plt.subplots(figsize=(10, 10))
tree.plot_tree(clf_tree, fontsize=10)
plt.show()


# ##### Controlling complexity of decision trees

# In[28]:


X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)


# #### correlation and collinear

# In[29]:


X = pd.DataFrame(X_train, columns = cancer.feature_names)
X.head(3)


# In[30]:


y = pd.DataFrame(y_train, columns = ['cancer_or_not'])
y.head(3)


# In[31]:


all_data = pd.concat([X, y], axis = 1)
all_data.head(2)


# In[32]:


corr_matrix = all_data.corr()

for i in range(corr_matrix.shape[0]):
    for j in range(i+1,corr_matrix.shape[1]):
        if abs(corr_matrix.iloc[i, j]) >= 0.8:
            print(corr_matrix.index[i], ' and ', corr_matrix.columns[j], ': ', round(corr_matrix.iloc[i,j], 4))


# #### collinear
# 
# * Dimitris Bertsimas and Michael Lingzhi Li, Scalable holistic linear regression, Operations Research Letters, 48 (2020), pp. 203–208

# In[33]:


from numpy import linalg as LA
epsilon = 10**(-2)
w, v = LA.eigh(X.T.dot(X))
m = np.sum(w < epsilon) # number of the eigenvectors with eigenvalue < epsilon
print('m in small_eigvec ' , m) 


# #### Analyzing Decision Trees

# In[34]:


clf_tree = DecisionTreeClassifier(max_depth=4, random_state=0)
clf_tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(clf_tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(clf_tree.score(X_test, y_test)))


# In[35]:


from sklearn.tree import export_graphviz
import graphviz

export_graphviz(clf_tree, out_file="tree.dot", class_names=["malignant", "benign"],
                feature_names=cancer.feature_names, impurity=False, filled=True)

with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


# #### Feature Importance in trees

# In[36]:


print("Feature importances:")
print(clf_tree.feature_importances_)


# In[37]:


def plot_feature_importances(n_features, feature_names, model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances(cancer.data.shape[1], cancer.feature_names, clf_tree)


# ### <font color='blue'> E. Ensembles of Decision Trees: Random forests </font>
# * Random forest falls under the supervised learning techniques in machine learning and it is certainly one of the most popular algorithms used for both regression and classification purposes. 
#     * The Random Forest algorithm is based on the concept of ensembling learning, which simply means, stacking together a lot of classifiers to improve the performance. 
#     * Simply put, a random forest is a group of decision trees and takes the majority of the outputs of the decision trees to improve the prediction and results

# <div>
# <img src="attachment:image-2.png" width="500"/>
# </div>

# The random forest algorithm can be represented mathematically as:
# 
# \begin{equation}
# \begin{split}
# RF = \text{argmax}_{j\in{(1,2,...,N)}} \sum_{i=1}^{i} \text{Decision Tree-}{i,j}
# \end{split}
# \end{equation} 
# 
# Where class $j$ refers to the classes in the data and $i$ refers to the number of decision trees from 1 up to the ith example. <br>
# Argmax refers to the maximum value of the function, in other words, the majority voting.

# In[38]:


from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))


# #### Ensemble methods
# * The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator.
# * Two families of ensemble methods are usually distinguished:
#     * In averaging methods, the driving principle is to build several estimators independently and then to average their predictions. On average, the combined estimator is usually better than any of the single base estimator because its variance is reduced. Examples: Bagging methods, Forests of randomized trees, …
#     * By contrast, in boosting methods, base estimators are built sequentially and one tries to reduce the bias of the combined estimator. 
#     * The motivation is to combine several weak models to produce a powerful ensemble.

# #### Gradient Boosted Regression Trees (Gradient Boosting Machines)
# * Gradient Tree Boosting or Gradient Boosted Decision Trees (GBDT) is a generalization of boosting to arbitrary differentiable loss functions. 
#     * GBDT is an accurate and effective off-the-shelf procedure that can be used for both regression and classification problems in a variety of areas including Web search ranking and ecology.
#     * The module sklearn.ensemble provides methods for both classification and regression via gradient boosted decision trees.

# In[49]:


from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split( 
    cancer.data, cancer.target, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print(f"Accuracy on training set: {gbrt.score(X_train, y_train):.3f}")
print(f"Accuracy on test set: {gbrt.score(X_test, y_test):.3f}")


# In[40]:


gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

print(f"Accuracy on training set: {gbrt.score(X_train, y_train):.3f}")
print(f"Accuracy on test set: {gbrt.score(X_test, y_test):.3f}")


# In[41]:


gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)

print(f"Accuracy on training set: {gbrt.score(X_train, y_train):.3f}")
print(f"Accuracy on test set: {gbrt.score(X_test, y_test):.3f}")


# In[42]:


gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

plot_feature_importances(cancer.data.shape[1], cancer.feature_names, gbrt)


# ### <font color='blue'> Shap for feature visualization </font>
# 可解釋 AI (XAI) 系列 — SHAP: https://medium.com/ai-academy-taiwan/%E5%8F%AF%E8%A7%A3%E9%87%8B-ai-xai-%E7%B3%BB%E5%88%97-shap-2c600b4bdc9e

# In[48]:


import shap

X, y = pd.DataFrame(cancer.data, columns = cancer.feature_names), cancer.target
model = GradientBoostingClassifier().fit(X, y)
explainer = shap.Explainer(model, X)

shap_values = explainer(X)
shap.waterfall_plot(shap_values[0])

# force plot
shap.force_plot(explainer.expected_value, shap_values.values[0,:], X.iloc[0,:])

# point color: Feature value higher with red, lower with blue
# X axis: range of shape value
shap.summary_plot(shap_values, X)


# In[47]:


import shap
from sklearn.ensemble import RandomForestRegressor

X, y = pd.DataFrame(cancer.data, columns = cancer.feature_names), cancer.target
model = RandomForestRegressor().fit(X, y) # RANDOM FOREST REGRESSOR
explainer = shap.Explainer(model, X)

shap_values = explainer(X)
shap.waterfall_plot(shap_values[0])
# force plot
shap.force_plot(explainer.expected_value, shap_values.values[0,:], X.iloc[0,:])

shap.summary_plot(shap_values, X)


# ### Homework#1
# Please compare the performances of the following models, eg. k-neighbors regression, LogisticRegression, GradientBoostingClassifier, and RandomForestClassifier, using the tool wear data.

# In[ ]:




