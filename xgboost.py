# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:08:34 2018

@author: shruthi
"""

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas 
dataset=pandas.read_csv('C:/Users/shruthi/Downloads/project_2018_dataset_draft1_csv_binary.csv',encoding="ISO-8859-1",header=None)
dataset.replace({'y':1,'n':0,'M':1,'F':0})
array=dataset.values
X=array[:,1:12]
Y=array[:,0]
seed = 7
test_size = 0.20
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

model = XGBClassifier()
model.fit(X_train, y_train)
#print(model)

y_pred = model.predict(X_test)
predictions = [value for value in y_pred]

#cross validation

scoring='accuracy'
results=[]
kfold=model_selection.KFold(n_splits=10,random_state=seed)
cv_results=model_selection.cross_val_score(model,X_train,y_train,cv=kfold,scoring=scoring)
results.append(cv_results)
msg="XGBOOST cross-validation: %f (%f)"%(cv_results.mean(),cv_results.std())
print(msg)

accuracy = accuracy_score(y_test, predictions)
print("xgboost Accuracy: %.2f%%" % (accuracy * 100.0))

print("confusion matrix:")
print(confusion_matrix(y_test,predictions))

print("classification summary report:")
print(classification_report(y_test,predictions))
