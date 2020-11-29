import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from pickle import dump, load


# Simple Iris Flower Prediction App
# This app predicts the **Iris flower** type!


# Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target


# Split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

for i in range(X_test.shape[1]):
    print('>%d train: min=%.3f, max%.3f, test: min=%.3f, max%.3f' %
        (i, X_train[:,i].min(), X_train[:,i].max(),
        X_test[:, i].min(), X_test[:, i].max()))


# define scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# label encode the y values

# fit scaler on the training set
scaler.fit(X_train)

# transform the X_train datasets
X_train_scaled = scaler.transform(X_train)


# Integer encode the labeled data (y)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder.fit(y_train)

y_train_encoded =label_encoder.transform(y_train)
# create a random foreset classifier

randomforest = RandomForestClassifier()
# fit the model
randomforest.fit(X_train_scaled, y_train_encoded)

# Save the model to the current directory
import joblib
joblib.dump(randomforest, 'model.sav')

# save the scaler using pickle
dump(scaler, open('scaler.pkl', 'wb'))

predict_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Load the model
model = joblib.load("model.sav")

#load the scaler
scaler = load(open('scaler.pkl', 'rb'))

# Make a prediction
# 1. input 3 values in a 3 list
input_row1 = [[6.7, 3.0, 5.2, 2.3]]
input_row2 = [[5.1,3.5,1.4, 0.2]]
input_row3 = [[5.7,2.8,4.1,1.3]]

# transform the inputs using the scaler
input_row1_scaled = scaler.transform(input_row1)
input_row2_scaled = scaler.transform(input_row2)
input_row3_scaled = scaler.transform(input_row3)

# make 3 predictions
predict = model.predict(input_row1_scaled)
print(f'Prediction is: {predict_labels[predict[0]]}')

predict = model.predict(input_row2_scaled)
print(f'Prediction is: {predict_labels[predict[0]]}')

predict = model.predict(input_row3_scaled)
print(f'Prediction is: {predict_labels[predict[0]]}')

