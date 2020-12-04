#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from pickle import load

#Initialize the flask App
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] =0

# Load the model
model = joblib.load('model.sav')

#load the scaler 
scaler = load(open('scaler.pkl','rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    
    # create a list for output labels

    prediction_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    
    # get the list of entered values from the website.
    features = [x for x in request.form.values()]

    # get the list of enterd values from the website and convert them to float
    int_features = [float(x) for x in request.form.values()]

    # Put the list into another list
    final_features = [np.array(int_features)]

     
    # **** preprocess the input using the scaler ****"  
    final_features_scaled = scaler.transform(final_features)

    # make a prediction
    prediction_encoded = model.predict(final_features_scaled)
    prediction = prediction_labels[prediction_encoded[0]]
   

    prediction_text = f'Iris flower type is predicted to be :  {prediction}'
    return render_template('index.html', prediction_text = prediction_text, features = features)

if __name__ == "__main__":
    app.run(debug=True)