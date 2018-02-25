from sklearn import datasets,tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import dummy_classifier
import myKNeighborClassifier

iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = .5)

#classifier = tree.DecisionTreeClassifier()
#classifier = KNeighborsClassifier()
#classifier = dummy_classifier.dummy_classifier()
classifier = myKNeighborClassifier.myKNeighborClassifier()

classifier.fit(x_train,y_train)

predictions = classifier.predict(x_test)

print accuracy_score(y_test,predictions)
