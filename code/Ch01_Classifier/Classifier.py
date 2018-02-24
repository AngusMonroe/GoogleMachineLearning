from sklearn import tree
features = [[140, 1], [130, 1], [150, 0], [170, 0]]  # scikit-learn uses real-valued features
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()  # create a classifier
clf = clf.fit(features, labels)  # discover data native mode
print(clf.predict([[150, 0]]))  # predict result
