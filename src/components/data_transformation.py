import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Import necessary tools from scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import custom modules for logging and exception handling
from src.exception import CustomException
from src.logger import logger
# You would need a utility function like this to save your .pkl file
from src.utils import save_object 

@dataclass
class DataTransformationConfig:
    """
    Configuration class for Data Transformation.
    This dataclass holds the file path for the preprocessing object that will be saved.
    """
    preprocessor_obj_file_path = os.path.join("artifact", "preprocessor.pkl")

class DataTransformation:
    """
    This class is responsible for the data preprocessing tasks.
    It creates and applies a preprocessing pipeline.
    """
    def __init__(self):
        """Initializes the data transformation configuration."""
        self.data_transformation_config = DataTransformationConfig()

    def get_data_tranformer_object(self, X):
        """
        This method defines the data transformation pipeline (preprocessor).
        It creates separate pipelines for numerical and categorical features and
        combines them using ColumnTransformer.
        
        Args:
            X (pd.DataFrame): The input feature DataFrame.

        Returns:
            A scikit-learn ColumnTransformer object.
        """
        try:
            print("[+] Creating data transformer object...")
            # Automatically identify numerical and categorical feature column names
            num_features = list(X.select_dtypes(exclude="object").columns)
            cat_features = list(X.select_dtypes(include="object").columns)
            
            # Using logger, but you can also use print
            logger.info(f"Identified numerical features: {num_features}")
            logger.info(f"Identified categorical features: {cat_features}")
            print(f"Identified numerical features: {num_features}")
            print(f"Identified categorical features: {cat_features}")

            # Define the preprocessing pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    # Step 1: Handle missing values by replacing them with the column's mean
                    ("imputer", SimpleImputer(strategy="mean")),
                    # Step 2: Scale features to have zero mean and unit variance
                    ("scaler", StandardScaler())
                ]
            )

            # Define the preprocessing pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    # Step 1: Handle missing values by replacing them with the most frequent value
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    # Step 2: Convert categorical text data into a numerical format (one-hot encoding)
                    ("ohe", OneHotEncoder(handle_unknown='ignore')),
                    # Step 3: Scale the one-hot encoded data. `with_mean=False` is important for sparse data.
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            # Combine the numerical and categorical pipelines into a single preprocessor object
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_features),
                    ("cat_pipeline", cat_pipeline, cat_features)
                ]
            )

            print("[+] ColumnTransformer object created successfully.")
            return preprocessor

        except Exception as e:
            # Log the error and raise a custom exception if anything goes wrong
            logger.error("[-] Error occurred during data transformation setup")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        """
        This method orchestrates the full data transformation process.
        It reads data, applies the preprocessor, and saves the preprocessor object.
        """
        try:
            print("\n--- Starting Data Transformation ---")
            # Read the raw train and test datasets
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            print(f"Data loaded. Train Shape: {train_df.shape}, Test Shape: {test_df.shape}")

            # Define the target column and separate features (X) from the target (y)
            target_column = 'math_score'
            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]
            print("Separated features and target variables for train and test sets.")

            # Get the preprocessor object (ColumnTransformer)
            preprocessing_obj = self.get_data_tranformer_object(X_train)
            
            # Apply the preprocessor to the datasets
            # fit_transform on training data to learn the transformations
            X_train_processed = preprocessing_obj.fit_transform(X_train)
            # transform on testing data using the transformations learned from the training data
            X_test_processed = preprocessing_obj.transform(X_test)
            print("Preprocessing applied to train and test data.")

            # Combine the processed features and the target variable back into arrays
            train_arr = np.c_[X_train_processed, np.array(y_train)]
            test_arr = np.c_[X_test_processed, np.array(y_test)]
            print("Combined processed features and target into final arrays.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            print(f"Preprocessor object would be saved to: {self.data_transformation_config.preprocessor_obj_file_path}")
            
            print("--- Data Transformation Complete ---")
            
            # Return the processed arrays and the path to the saved preprocessor
            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            # If an error occurs, log it and raise the custom exception
            logger.error(f"[-] Error in initiate_data_transformation: {e}")
            raise CustomException(e, sys)

# # This block executes only when the script is run directly for testing.
# if __name__ == "__main__":
#     # This block demonstrates how the entire pipeline would run.
#     # To execute this, you first need to run the `data_ingestion.py` script
#     # to create the 'artifact/train.csv' and 'artifact/test.csv' files.

#     print("--- Running Standalone Data Transformation Test ---")

#     # Step 1: Get the data paths (these would come from data_ingestion)
#     train_path = 'artifact/train.csv'
#     test_path = 'artifact/test.csv'

#     # Check if data files exist before running
#     if not os.path.exists(train_path) or not os.path.exists(test_path):
#         print(f"Error: Make sure '{train_path}' and '{test_path}' exist.")
#         print("Please run the data_ingestion.py script first.")
#     else:
#         # Step 2: Create an instance of the DataTransformation class
#         data_transformation = DataTransformation()
        
#         # Step 3: Run the transformation process
#         train_array, test_array, preprocessor_file_path = data_transformation.initiate_data_transformation(train_path, test_path)

#         # Print the results
#         print("\n--- Test Run Summary ---")
#         print(f"Processed training array shape: {train_array.shape}")
#         print(f"Processed testing array shape: {test_array.shape}")
#         print(f"Preprocessor file path: {preprocessor_file_path}")
#         print("--------------------------")