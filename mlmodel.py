import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

import pickle


# Simple Iris Flower Prediction App
# This app predicts the **Iris flower** type!


# Load the dataset
iris = datasets.load_iris()
X = iris.data
target = iris.target


# label encode the y values

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder.fit(target)

y =label_encoder.transform(target)
# create a random foreset classifier

randomforest = RandomForestClassifier()
# fit the model
randomforest.fit(X, y)

#Save the model to the current directory
# Pickle serlializes objects so they can be saved to a file, and loaded in a program again later on.

pickle.dump(randomforest, open ('model.pkl', 'wb'))



#Load model to compare results
predict_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

model = pickle.load(open('model.pkl', 'rb'))
# predict = model.predict([[5.1, 3.5, 1.4, 0.2]])
# predict = model.predict([[5.7, 2.8, 4.1, 1.3]])
predict = model.predict([[6.7, 3.0, 5.2, 2.5]])


print(f'Prediction is: {predict_labels[predict[0]]}')
# if (predict == 0):
#     print(f'prediction: Iris-setosa')
# elif (predict == 1):
#     print(f'prediction: Iris-versicolor')
# elif(predict == 2):
#     print(f'prediction: Iris-virginica')
# print(model.predict([[5.1, 3.5, 1.4, 0.2]]))
# print(model.predict([[5.7, 2.8, 4.1, 1.3]]))
# print(model.predict([[6.7, 3.0, 5.2, 2.5]]))

