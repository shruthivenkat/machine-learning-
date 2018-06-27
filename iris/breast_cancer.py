# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:23:47 2018

@author: shruthi
"""
from sklearn.datasets import load_breast_cancer
cancer_dataset=load_breast_cancer()

print("key values in the dataset: \n{}".format(cancer_dataset.keys()))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(cancer_dataset['data'],cancer_dataset['target'],random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

prediction=knn.predict(X_test)
print("the prediction test set:\n {}".format(prediction))
print("prediction test score: {:.2f}".format(knn.score(X_test,y_test)))

