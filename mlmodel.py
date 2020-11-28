import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


# Simple Iris Flower Prediction App
# This app predicts the **Iris flower** type!


# Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target


# Split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# label encode the y values

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder.fit(y_train)

y_train_encoded =label_encoder.transform(y_train)
# create a random foreset classifier

randomforest = RandomForestClassifier()
# fit the model
randomforest.fit(X_train, y_train_encoded)

# Save the model to the current directory
import joblib
joblib.dump(randomforest, 'model.sav')

predict_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Load the model
model = joblib.load("model.sav")


# Make a prediction usign a test case (input_row)
input_row = [6.7, 3.0, 5.2, 2.3]
predict = model.predict([input_row])
print(f'Prediction is: {predict_labels[predict[0]]}')
