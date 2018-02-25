import pandas as pd
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("train.csv",sep=';')
y = df['RES']
x = df[['HOME','DRAW','AWAY']]


#classifier = KNeighborsClassifier()
classifier = tree.DecisionTreeClassifier()
classifier.fit(x,y)

predictions = classifier.predict([[1.90,3.50,4.20]])

print predictions
