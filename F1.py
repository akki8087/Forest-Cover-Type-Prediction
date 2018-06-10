# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 22:51:16 2018

@author: NP
"""


import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.iloc[:,1:-1]
y = train.iloc[:,-1]
'''
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''
X_train = X
y_train = y
X_test = test.iloc[:,1:]


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
'''
'''
from sklearn.svm import SVC
Classifier = SVC()

'''

# Random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300,criterion = 'entropy',random_state = 0)

'''
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
'''
'''
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
'''

# Predicting the Test set results
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

'''
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#print(classifier.feature_importances_)
print(classifier.score(X_test,y_test))
'''
result = pd.DataFrame()
result['Id'] = test['Id']
result['Cover_Type'] = y_pred
result.to_csv('F_RF.csv',index = False)

