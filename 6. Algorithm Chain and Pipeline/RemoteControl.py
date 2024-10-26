# -*- coding: utf-8 -*-

import paho.mqtt.subscribe as subscribe
import paho.mqtt.publish as publish
import time
import json
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

host = 'localhost'
portNo = 	1883
topic_start = 'ml/start'
topic_echo = 'ml/echo'

def LoadData():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y

def DataProcess(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)
    pipe = Pipeline([ ('pca', PCA(n_components = 2)), 
                     ('std', StandardScaler()),
                     ('decision_tree', DecisionTreeClassifier(random_state=42))
                     ],
                    verbose = True)
    tree = pipe.fit(X_train, y_train)
    return accuracy_score(y_test, tree.predict(X_test))

while True:
    msg = subscribe.simple(topic_start, hostname = host)
    print(f'topic: {msg.topic}, value: {msg.payload}')
    time.sleep(1)
    
    X, y = LoadData()
    msg = round(DataProcess(X, y),3)
    jmsg = json.dumps(msg, ensure_ascii=False)
    publish.single(topic_echo, jmsg, hostname= host)