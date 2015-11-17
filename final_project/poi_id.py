#!/usr/bin/python

import sys
import pickle
import numpy
sys.path.append("../../../Google Drive/Data Science/tools")
from usefultools import indexAll,findOuliers
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### Extract features:
### print data_dict['METTS MARK'].keys()
### 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'email_address', 'from_poi_to_this_person'

### The first feature must be "poi".
features_list = ['poi','shared_receipt_with_poi','fraction_from_poi'] # You will need to use more features
#
# Final feature list:
# features = ['poi','salary','bonus','from_this_person_to_poi','from_poi_to_this_person','fraction_from_poi','fraction_to_poi','log_total_emails_ratio','shared_receipt_with_poi','director_fees','deferral_payments','log_poi_emails_ratio']
# These features do not seam to be very relevant to distinguish POI from non-POI:
# expenses, exercised_stock_options, loan_advances
# 5 Best features:
#  ['salary', 'bonus', 'fraction_from_poi', 'log_total_emails_ratio', 'shared_receipt_with_poi']

# Each element of the array is a datapoint and each element of the datapoint is a feature
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0) # Remove TOTAL row

### Task 3: Create new feature(s)
for person in data_dict:
    n_from   = data_dict[person]['from_this_person_to_poi']
    total_from = data_dict[person]['from_messages']
    # print data_dict[person]['poi'], "from: ", num, "out of", total, "(", float(num)/float(total), ")"
    if total_from != 0:
        data_dict[person]['fraction_from_poi'] = float(n_from)/float(total_from)
    else:
        data_dict[person]['fraction_from_poi'] = float(total_from)

    n_to   = data_dict[person]['from_poi_to_this_person']
    total_to = data_dict[person]['to_messages']
    # print data_dict[person]['poi'],"to: ", person, num, "out of", total, "(", float(num)/float(total), ")"
    if total_to != 0:
        data_dict[person]['fraction_to_poi'] = float(n_to)/float(total_to)
    else:
        data_dict[person]['fraction_to_poi'] = float(total_to)

    if n_from != 0 and n_to != 0:
        data_dict[person]['log_poi_emails_ratio'] = numpy.log(float(n_to)/float(n_from))
    else:
        data_dict[person]['log_poi_emails_ratio'] = numpy.NaN

    if total_from != 0.:
        data_dict[person]['log_total_emails_ratio'] = numpy.log(float(total_to)/float(total_from))
    else:
        data_dict[person]['log_total_emails_ratio'] = numpy.NaN

    total    = data_dict[person]['total_payments']
    deferral = data_dict[person]['deferral_payments']
    #  print data_dict[person]['poi'],"to: ", person, num, "out of", total, "(", float(num)/float(total), ")"
    if total != 0:
        data_dict[person]['relative_deferral'] = float(deferral)/float(total)
    else:
        data_dict[person]['relative_deferral'] = numpy.nan

### Store to my_dataset for easy export below.
import matplotlib.pyplot as plt

my_dataset = featureFormat(data_dict,features_list)
poi, features = targetFeatureSplit( my_dataset )
print "Number of people in the dataset:",len(features)

# If you are testing only 2 selected features, plot their distributions and outliers
if len(features_list) == 3:
    feature1, feature2 = zip(*features)
    feature1 = numpy.reshape( numpy.array(feature1), (len(feature1), 1))
    feature2 = numpy.reshape( numpy.array(feature2), (len(feature2), 1))

    minoroutliers1, majoroutliers1 = findOuliers(feature1)
    if len(majoroutliers1):
        print "Major outliers for",features_list[1],":"
    for outlier in majoroutliers1:
        for person in data_dict:
            if data_dict[person][features_list[1]] == outlier:
                print person,"(",outlier[0],")","POI:",data_dict[person]['poi']
    print ""

    minoroutliers2, majoroutliers2 = findOuliers(feature2)
    if len(majoroutliers2):
        print "Major outliers for",features_list[2],":"
    for outlier in majoroutliers2:
        for person in data_dict:
            if data_dict[person][features_list[2]] == outlier:
                print person,"(",outlier[0],")","POI:",data_dict[person]['poi']
    print ""

    plt.scatter(feature1,feature2)
    for ii, pp in enumerate(feature1):
        if poi[ii]:
            plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")

    plt.xlabel(features_list[1])
    plt.ylabel(features_list[2])
    # plt.xscale('log')
    # plt.yscale('log')

    # Plot POIs and non-POIs in different colors and a legend
    import matplotlib.patches as mpatches
    red_points = mpatches.Patch(color='red', label='POI')
    blue_points = mpatches.Patch(color='blue', label='non POI')
    plt.legend(handles=[blue_points,red_points],loc='best')
    plt.show()


# Fill missing values with the mean of the parameter so they can be used
# within a classifier:
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
features_filled = imp.fit_transform(features)

# Rescaling of features, only needed for SVM, kMeans clustering...
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features_rescaled = scaler.fit_transform(features_filled)


### Extract features and labels from dataset for local testing
# data = featureFormat(data_dict, features_list, sort_keys = True)
# labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features_rescaled, poi, test_size=0.3, random_state=53)

print "Fraction of POIs in the test dataset:",float(sum(labels_test))/float(sum(labels_train)+sum(labels_test))

from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=5)
X_train = selector.fit_transform(features_train, labels_train)
X_test  = selector.transform(features_test)


feat_score = selector.scores_.tolist()
feat_pval = selector.pvalues_.tolist()
templist = features_list[1:]
for feat in templist:
    i = features_list.index(feat) - 1
    print feat,"selector score:",feat_score[i],
    print "(p =",feat_pval[i],")"

feat_support = selector.get_support().tolist()
print "Selected indexes:", [ fl for (fl,fs) in zip(templist,feat_support) if fs ]

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.metrics import accuracy_score,precision_score,recall_score
scores = ['precision', 'recall']

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report


param_grid = [ {},
               { 'criterion': [ 'gini' , 'entropy' ],
                 'max_depth': [ 8 , 6 , 4 ] ,
                 'min_samples_split': [ 10 , 6 , 4 , 2] } ,
               { 'C': [ 1000 , 100 , 10 , 5 , 1 ] } ,
               { 'kernel': [ 'linear', 'rbf', 'sigmoid'] ,
                 'C': [ 1000, 100, 10, 5, 1] ,
                 'gamma': [ 0.05 , 0.1 , 0.2 , 0.5 , 1.0 , 10. ] }
             ]

classifiers = [ GaussianNB() , DecisionTreeClassifier(), LogisticRegression(), SVC() ]
classifier_names = [ 'GaussianNB' , 'DecisionTreeClassifier', 'LogisticRegression', 'SVC' ]

best_performance = { }

for score in scores:
    best_performance[score] = {}
    # for estimator in [ 'GaussianNB' , 'DecisionTreeClassifier' , 'LogisticRegression' , 'SVC' ]:
    for classifier in classifiers:
        index = classifiers.index(classifier)
        print "# %s :: Tuning parameters for %s" % (classifier_names[index],score)
        clf = GridSearchCV(classifier,param_grid[index],
                          scoring='%s_weighted' % score,cv=10)

        clf.fit(X_train,labels_train)

        print "# Best parameters set found on training set:"
        print clf.best_params_
        print "# Grid scores on testing set:"
        for params, mean_score, cv_scores in clf.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r" % (mean_score, cv_scores.std() * 2, params)

        best_performance[score][classifier_names[index]] = "%0.3f with parameters %r" % (clf.best_score_, clf.best_params_)

        print "Detailed classification report on the test set:"
        print classification_report(labels_test,clf.predict(X_test))

for score in scores:
    print "For optimum",score,":"
    for name in classifier_names:
        print "  ",name,":  ",best_performance[score][name]


# acc  = accuracy_score(pred, labels_test)
# print ""
# print "Accuracy:",acc," (Good predictions / All predictions)"
# pre = precision_score(pred, labels_test)
# print "Precision:",pre," (Real POIs / Predicted POIs)"
# rec  = recall_score(pred, labels_test)
# print "Recall:",rec," (Identified POIs / All POIs)"
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# features_train, features_test, labels_train, labels_test = \
    # train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# dump_classifier_and_data(clf, my_dataset, features_list)
dump_classifier_and_data(clf, data_dict, features_list)
