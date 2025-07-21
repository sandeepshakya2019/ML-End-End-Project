import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_Iobject
import os

class PredictPipeline:
    """
    This class is responsible for making predictions on new data.
    It loads the preprocessor and the trained model to generate output.
    """
    def __init__(self):
        """Initializes the prediction pipeline."""
        pass

    def predict(self, features):
        """
        Loads the necessary objects and makes a prediction.
        
        Args:
            features (pd.DataFrame): The input features for prediction.

        Returns:
            The model's prediction.
        """
        try:
            # Define the paths to the preprocessor and model files
            model_path = os.path.join("artifact", "model.pkl")
            preprocessor_path = os.path.join('artifact', 'preprocessor.pkl')
            
            print("Before Loading")
            # Load the preprocessor and model objects from their files
            model = load_Iobject(file_path=model_path)
            preprocessor = load_Iobject(file_path=preprocessor_path)
            print("After Loading")

            # Transform the input features using the loaded preprocessor
            data_scaled = preprocessor.transform(features)
            # Make a prediction using the loaded model
            preds = model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            # If an error occurs, raise a custom exception
            raise CustomException(e, sys)

class CustomData:
    """
    This class is responsible for mapping input data from a web form
    or other source to a format that the prediction pipeline can use.
    """
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        
        # Assign all the input features to instance attributes
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Converts the instance attributes into a pandas DataFrame.
        This DataFrame will have the correct column names and structure
        to be used by the prediction pipeline.

        Returns:
            pd.DataFrame: A DataFrame containing the input data.
        """
        try:
            # Create a dictionary from the instance attributes
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert the dictionary to a pandas DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # If an error occurs, raise a custom exception
            raise CustomException(e, sys)
