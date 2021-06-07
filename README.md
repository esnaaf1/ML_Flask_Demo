# ML_flask_demo

## How to Integrate Machine Learning with Flask 

This demonstration intended to show how to integrate a Machine Learning Model with Flask and HTML. 


### Please clone this repository to your computer and then do the following:

1. Navigate to the folder that contains ``app.py`` and launch a GitBash (Windows) or  Terminal (Mac).
1. Type ``source activate PythonData`` and then hit ENTER.
1. Type ``python app.py`` and then hit ENTER.
1. Notice that Flask server starts and tells you which port it is running on.  Don't close this window
1. Enter the following addrress in your Chrome browser:   http://127.0.0.1:5000/
1. Enter the data for Sepal LEngth, Sepal Width, Petal Length, Petal Width ( note that all the fields are required) and click the Predict button
Note:  You can enter the following test cases:
    test case 1: 6.7,3.0, 5.2, 2.3
    test case 2: 5.7.2.8, 4.1, 1.3
    test case 3: 5.1, 3.5, 1.4, 0.2


### Additional notes

### Dataset
Iris dataset is one of the datasets that is already supplied by scikit-learn

### ML Model
I used RandomForest classifier model for this example; however any classifier can be used

* Please be sure to follow the same directory structure
* You can replace RandomForests with your own model in the ``mlmodel.py``
* In the Iris example, I don't have any pre-processing of the input data, but you might need to do that for your dataset
* There are several options to save and load the model.  In this example, I used the joblib library.  Note that joblib library was used in the Machine Learning homework.


