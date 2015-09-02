#!/usr/bin/python

"""
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors

    Sara has label 0
    Chris has label 1

"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



#########################################################
### your code goes here ###
#clf = SVC(C=1,kernel="rbf",gamma=1)
clf = SVC(kernel="rbf",C=10000.)

# Shrink the size of the dataset to speed up the training:
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"

#### store your predictions in a list named pred
t0 = time()
pred = clf.predict(features_test)
print "testing time:", round(time()-t0, 3), "s"



acc = accuracy_score(pred, labels_test)
print "Accuracy: %.4f" % acc

print "Prediction for email 10: %i" % pred[10]
print "Prediction for email 26: %i" % pred[26]
print "Prediction for email 50: %i" % pred[50]

count = 0
for email in pred:
    if email == 1:
        count += 1

print "%i emails are predicted to be from Chris" % count

#########################################################
