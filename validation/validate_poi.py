#!/usr/bin/python


"""
    starter code for the validation mini-project
    the first step toward building your POI identifier!

    start by loading/formatting the data

    after that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!

from sklearn import tree
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.cross_validation import train_test_split

clf = tree.DecisionTreeClassifier()#min_samples_split=20)

features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.3,random_state=42)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)

true_positives=0
for poi in zip(labels,pred):
    if poi == (1.0,1):
        true_positives += 1
print "True positives: ",true_positives

# pred        = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
# labels_test = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

acc = accuracy_score(labels_test,pred)
print "Accuracy: %.4f" % acc
prec = precision_score(labels_test,pred)
print "Precision: %.4f" % prec
recall = recall_score(labels_test,pred)
print "Recall: %.4f" % recall
