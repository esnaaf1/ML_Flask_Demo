# ML_flask_demo

## How to Integrate Machine Learning with Flask 


This Demo is intended to show necessary steps required to integrate a machine learning model with Flask server and jinja to make predictions

### Dataset
Iris dataset is one of the datasets that is already supplied by scikit-learn

### ML Model
I used RandomForest classifier model for this example; however any classifier can be used

### Steps

1. Create a python file for the RandomForest model that includes data pre-processing,fit, predfict, and saves the model using pickle (mlmodel.py)

2. Create a an html file to capture users inputs for predicting Iris flower types, and passes the data to Flask server using jinja (index.html)

3. Create a sub-folder called 'templates' where index.html is saved

4. Create a python file (app.py) to start a FLask server, open the save Machnine Learning model and make a prediction based on the user input 

