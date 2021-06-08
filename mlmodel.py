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

# Show a simple data summary
for i in range(X_test.shape[1]):
    print('>%d train: min=%.3f, max%.3f, test: min=%.3f, max%.3f' %
        (i, X_train[:,i].min(), X_train[:,i].max(),
        X_test[:, i].min(), X_test[:, i].max()))


# Define the scaling function 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit the scaling function to the training set
scaler.fit(X_train)

# Transform the training set
X_train_scaled = scaler.transform(X_train)

# Define the label encoder and fit it to the training set
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(y_train)

# Transform the labels of the training set
y_train_encoded = label_encoder.transform(y_train)

# Define the model
randomforest = RandomForestClassifier()

# Fit the model to the scaled / encoded training set
randomforest.fit(X_train_scaled, y_train_encoded)

# Save the model to a pickle file (i.e., "pickle it")
# so we can use it from the Flask server. 
dump(randomforest, open('randomforest.pkl', 'wb'))

# Save the scaling function to a pickle file (i.e., "pickle it")
# so we can use it from the Flask server. 
dump(scaler, open('scaler.pkl', 'wb'))


# The code that follows will typically occur in the Flask server ...

# Define prediction labels.
predict_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Load the model.
randomforest = load(open('randomforest.pkl', 'rb'))

# Load the scaler.
scaler = load(open('scaler.pkl', 'rb'))

# Now, scale the data and make a prediction ...

# 1. Create three diffferent sets of inputs (i.e., three
# different irises). Note that each set is constructed 
# as a list inside of another list (or an array inside of
# another array). This is how scikit-learn needs it. 
input_row1 = [[6.7, 3.0, 5.2, 2.3]]
input_row2 = [[5.1, 3.5, 1.4, 0.2]]
input_row3 = [[5.7, 2.8, 4.1, 1.3]]

# 2. Transform each input using the scaler function.
input_row1_scaled = scaler.transform(input_row1)
input_row2_scaled = scaler.transform(input_row2)
input_row3_scaled = scaler.transform(input_row3)

# 3. Make a prediction for each input.
predict = randomforest.predict(input_row1_scaled)
print(f'Prediction 1 is: {predict_labels[predict[0]]}')

predict =randomforest.predict(input_row2_scaled)
print(f'Prediction 2 is: {predict_labels[predict[0]]}')

predict =randomforest.predict(input_row3_scaled)
print(f'Prediction 3 is: {predict_labels[predict[0]]}')

