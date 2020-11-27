#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

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

    prediction_encoded = model.predict(final_features)
    prediction = prediction_labels[prediction_encoded[0]]
   


    return render_template('index.html', prediction_text='Iris Flower type is  :  {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)