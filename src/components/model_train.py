import os
import sys
from dataclasses import dataclass

# Import various regression models from scikit-learn and other libraries
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Import custom modules for logging, exception handling, and utilities
from src.exception import CustomException
from src.logger import logger
from src.utils import save_object, evaluate_models

# Import other components to run the full pipeline
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for the Model Trainer.
    This holds the file path where the trained model will be saved.
    """
    trained_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    """
    This class is responsible for training, evaluating, and saving the best model.
    """
    def __init__(self):
        """Initializes the model trainer configuration."""
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        This is the main method that orchestrates the model training process.
        
        Args:
            train_array (np.array): The processed training data array (features + target).
            test_array (np.array): The processed testing data array (features + target).
        """
        try:
            logger.info("Splitting training and test input data")
            print("\n--- Starting Model Training ---")
            
            # Split the arrays into features (X) and target (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # All rows, all columns except the last one
                train_array[:, -1],   # All rows, only the last column
                test_array[:, :-1],
                test_array[:, -1]
            )
            print("Data split into features (X) and target (y) complete.")

            # Define a dictionary of models to be evaluated
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            # Define a dictionary of hyperparameters for GridSearchCV
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate all models using the utility function
            print("Evaluating all models...")
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)
            
            print("\n--- Model Evaluation Report ---")
            print(model_report)
            print("-----------------------------")

            # Get the best model's score from the report
            best_model_score = max(sorted(model_report.values()))

            # Find the name of the best model corresponding to the best score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with R2 score > 0.6")
            
            
            print(f"Best Model Found: {best_model_name} with R2 Score: {best_model_score}")

            # If the best model's score is below a certain threshold, raise an exception
            logger.info(f"Best model found on both training and testing dataset: {best_model_name}")

            # Save the best performing model to a file
            print("Saving the best model...")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Make predictions on the test set with the best model
            predicted = best_model.predict(X_test)

            # Calculate the R2 score for the final model
            r2_square = r2_score(y_test, predicted)
            
            print(f"Final R2 score of the best model on the test data is: {r2_square}")
            print("--- Model Training Complete ---")
            
            return r2_square

        except Exception as e:
            # If any error occurs, log it and raise the custom exception
            logger.error(f"Error in initiate_model_trainer: {e}")
            raise CustomException(e, sys)

# # This block executes only when the script is run directly.
# # It runs the full pipeline from data ingestion to model training.
# if __name__ == "__main__":
#     print("--- Running Full Training Pipeline ---")
    
#     # Step 1: Data Ingestion
#     data_ingestion = DataIngestion()
#     train_data_path, test_data_path, _ = data_ingestion.initiate_data_ingestion()
#     print(f"Data ingestion complete. Train path: {train_data_path}, Test path: {test_data_path}")

#     # Step 2: Data Transformation
#     data_transformation = DataTransformation()
#     train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
#     print("Data transformation complete.")

#     # Step 3: Model Training
#     model_trainer = ModelTrainer()
#     final_r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    
#     print("\n--- Pipeline Finished ---")
#     print(f"Final R2 Score of the best model: {final_r2_score}")
#     print("-------------------------")
