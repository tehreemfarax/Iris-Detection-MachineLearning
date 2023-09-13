from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris() #Loading the dataset
print(iris.keys())

iris = pd.DataFrame(
    data= np.c_[iris['data'], iris['target']],
    columns= iris['feature_names'] + ['target']
    )
print(iris)
print(iris.head(6))
species = []
for i in range (len(iris['target'])):
    if iris['target'][i] == 0:
        species.append('setosa')
    elif iris['target'][i] == 1:
        species.append('versicolor')
    else:
        species.append('virginica')

iris['species'] = species

iris.groupby('species').size()

print(iris.describe())

setosa = iris[iris.species == 'setosa']
versicolor = iris[iris.species == 'versicolor']
virginica = iris[iris.species == 'virginica']

fig, ax = plt.subplots()
ax.set_xlabel('sepal length (cm)')
ax.set_ylabel('sepal width (cm)')
ax.grid()
ax.set_title('Iris Sepal')
ax.legend()
fig.set_size_inches(13,7)
ax.scatter(setosa['sepal length (cm)'], setosa['sepal width (cm)'], label = 'Setosa', facecolor = 'blue')
ax.scatter(versicolor['sepal length (cm)'], versicolor['sepal width (cm)'], label = 'Versicolor', facecolor = 'red')
ax.scatter(virginica['sepal length (cm)'], virginica['sepal width (cm)'], label = 'Virginica', facecolor = 'green')


x = iris.drop(['target','species'], axis=1)

x=x.to_numpy()[:, (0,1)]
y = iris['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 42)

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

training_prediction = log_reg.predict(x_train)
print(training_prediction)

test_prediction = log_reg.predict(x_test)
print(test_prediction)

print('Precision, Recall, Confusion Matrix, in training \n ')

print(metrics.classification_report(y_train, training_prediction, digits=3))
print(metrics.classification_report(y_train, training_prediction))

print('Precision, Recall, Confusion Matrix, in training \n ')

print(metrics.classification_report(y_test,test_prediction, digits=3))
print(metrics.classification_report(y_test,test_prediction))

plt.show()