import os
import sys
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logger

def save_object(file_path, obj):
    """
    Saves a Python object to a file using pickle.

    Args:
        file_path (str): The path where the object will be saved.
        obj (any): The Python object to save (e.g., a preprocessor or model).
    """
    try:
        # Get the directory name from the full file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it doesn't already exist
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory '{dir_path}' created or already exists.")

        # Open the file in write-binary mode ("wb") and save the object
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        print(f"Object saved successfully at: {file_path}")

    except Exception as e:
        # If an error occurs, raise a custom exception for detailed logging
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Trains, tunes, and evaluates multiple models.

    This function iterates through a dictionary of models, uses GridSearchCV to find the
    best hyperparameters, trains the model with these parameters, and evaluates its
    performance on both the training and testing sets.

    Args:
        X_train: Training feature data.
        y_train: Training target data.
        X_test: Testing feature data.
        y_test: Testing target data.
        models (dict): A dictionary of model names and their instances.
        param (dict): A dictionary of hyperparameters for each model for GridSearchCV.

    Returns:
        A dictionary containing the R2 score for each model on the test data.
    """
    try:
        # Create an empty dictionary to store the report (model name: test_score)
        report = {}

        # Iterate through each model provided in the dictionary
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            para = param[model_name]

            print(f"--- Evaluating Model: {model_name} ---")

            # Use GridSearchCV to find the best hyperparameters for the current model
            # cv=3 means 3-fold cross-validation will be used
            print(f"Performing GridSearchCV for {model_name}...")
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            print(f"Best parameters found: {gs.best_params_}")

            # Set the model's parameters to the best ones found by GridSearchCV
            model.set_params(**gs.best_params_)
            # Train the model on the full training data with the best parameters
            model.fit(X_train, y_train)
            print(f"{model_name} trained with best parameters.")

            # Make predictions on both the training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate the R2 score for both sets
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            print(f"Train R2 Score: {train_model_score:.4f}")
            print(f"Test R2 Score: {test_model_score:.4f}")
            print("--------------------------------" + "-" * len(model_name))


            # Store the model's test score in the report dictionary
            report[model_name] = test_model_score

        return report

    except Exception as e:
        # If an error occurs, raise a custom exception
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads a Python object from a file using pickle.

    Args:
        file_path (str): The path of the file to load.

    Returns:
        The loaded Python object.
    """
    try:
        # Open the file in read-binary mode ("rb") and load the object
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        
        print(f"Object loaded successfully from: {file_path}")
        return obj

    except Exception as e:
        # If an error occurs, raise a custom exception
        raise CustomException(e, sys)
