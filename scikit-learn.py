import matplotlib.pyplot as plt

x =[i for i in range (10)]
print(x)
y = [2*i for i in range (10)]
print (y)
plt.plot(x, y)
plt.xlabel('x-axis')
plt.ylabel('y-axis')


plt.scatter(x, y)

#%%%%%%%%%%%%
#TRAIN TEST SPLIT
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
iris = datasets.load_iris()
X = iris.data
y = iris.target

print (X.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#%%%%%%%%%%
#K NEAREST NEIGHBOUR 

import csv
import numpy as np 
import pandas as pd 
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r'C:\Users\elnaz\Desktop\COURSERA\scikit-learn\car.data.csv')

var = ['buying', 'maint', 'safety']
#print (var)
X= data[var].values
y = data[['class']]

Le=LabelEncoder()
for i in range(len(X[0])):
    X[:, i]=Le.fit_transform(X[:, i])


label_mapping= {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}

y['class'] = y['class'].map(label_mapping)
y = np.array(y)

KNN=neighbors.KNeighborsClassifier(n_neighbors=25, weights= 'uniform')

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2)

KNN.fit(X_train, y_train)

prediction = KNN.predict(X_test)
accuracy= metrics.accuracy_score(y_test, prediction)
print('prediction: ', prediction)
print ('accuracy', accuracy)
a = 100
print ('actual value: ', y[a] )
print ('predicted value: ', KNN.predict(X)[a])

#%%%%%%%%%%%%%%%%%%%%%%%%%%

#SVM, support vector machine 

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn import neighbors, metrics

iris = datasets.load_iris()
X = iris.data
y = iris.target

classes = ['Iris setosa', 'Iris Versicolour', 'Iris Virginica']

print (X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)


model = svm.SVC()
model.fit(X_train, y_train)

print(model)

prediction = model.predict(X_test)
accuracy= metrics.accuracy_score(y_test, prediction)
print('prediction: ', prediction)
print ('accuracy', accuracy)
a = 100
print ('actual value: ', y[a] )
print ('predicted value: ', model.predict(X)[a])

for i in range (len(prediction)):
    print (classes[prediction[i]])
    
    
#%%%%%%%%%%%%%%%%
    
#LINEAR REGRESSION 
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

boston=datasets.load_boston()
X = boston.data
y = boston.target

print (X.shape, y.shape)

l_reg=linear_model.LinearRegression()

plt.scatter(X.T[5], y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

model= l_reg.fit(X_train, y_train)

prediction = model.predict(X_test)
print('prediction: ', prediction)
print('R^2 value: ', l_reg.score(X, y))
print('coefficient value: ', l_reg.coef_)
print('intercept: ', l_reg.intercept_)

#%%%%%%%%%%%%%%%%%%%%%%

#Logostic regression 
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import numpy as pd

bc = load_breast_cancer()
X =scale(bc.data)

y=bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

model = KMeans(n_clusters=2, random_state=0)

model.fit(X_train)

prediction = model.predict(X_test)

labels = model.labels_
accuracy= metrics.accuracy_score(y_test, prediction)

print('labels: ', labels)
print('prediction: ', prediction)
print('accuracy: ', accuracy)
print('actual: ', y_test)