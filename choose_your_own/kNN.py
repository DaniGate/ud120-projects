#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture,output_image

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()

#################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary

n_neighbors = 15
print "Loading %iNN library" % n_neighbors
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors)
print "Training algorithm"
clf.fit(features_train,labels_train)
print "Predicting results"
pred = clf.predict(features_test)

print "Computing algorithm accuracy"
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print "Accuracy: %.4f" % acc
# Accuracy 93.6% for 3NN
# Accuracy 94.0% for 4NN but shouldn't use multiples of 2! Why is better?
# Accuracy 92.0% for 5NN
# Accuracy 93.6% for 7NN
outputfile = "test_%iNN.png" % n_neighbors
print "Saving output plot as %s" % outputfile
prettyPicture(clf, features_test, labels_test,outputfile)
#output_image(outputfile, "png", open("test.png"", "rb").read())
output_image(outputfile, "png", open(outputfile, "rb").read())
