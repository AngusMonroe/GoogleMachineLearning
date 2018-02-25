# import a dataset
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

# decision tree
from sklearn import tree
my_classifier_tree = tree.DecisionTreeClassifier()

my_classifier_tree.fit(X_train, y_train)

predictions_tree = my_classifier_tree.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions_tree))

# KNN
from sklearn.neighbors import KNeighborsClassifier
my_classifier_knn = KNeighborsClassifier()

my_classifier_knn.fit(X_train, y_train)

predictions_knn = my_classifier_knn.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions_knn))
