#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

#Initialize the flask App
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] =0

# Load the model
model = joblib.load('model.sav')

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    
    # create a list for output labels

    prediction_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    
    # get the list from the website
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    # **** Additional pre-processing would occur here ****"  
    

    # make a prediction
    prediction_encoded = model.predict(final_features)
    prediction = prediction_labels[prediction_encoded[0]]
   

    prediction_text = f'Iris flower type is predicted to be :  {prediction}'
    return render_template('index.html', prediction_text = prediction_text)

if __name__ == "__main__":
    app.run(debug=True)