import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
# from src.logger import logger
from src.logger import logger


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact', "train.csv")
    test_data_path: str = os.path.join('artifact', "test.csv")
    raw_data_path: str = os.path.join('artifact', "raw.csv")

class DataIngestion:
    def __init__(self):
        # Initialize configuration with data paths
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("[+] Entered Data Ingestion Method")

        try:
            # Read the input dataset
            csv_path = 'notebook/datasets/stud.csv'  # Change if needed
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Dataset file not found at path: {csv_path}")
            df = pd.read_csv(csv_path)
            logger.info(f"[+] Dataset successfully read with shape: {df.shape}")

            # Create artifact directory if not present
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info(f"[+] Raw dataset saved at {self.ingestion_config.raw_data_path}")

            # Train-Test Split
            logger.info("[+] Performing train-test split (80/20)")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logger.info(f"[+] Train dataset saved at {self.ingestion_config.train_data_path}")
            logger.info(f"[+] Test dataset saved at {self.ingestion_config.test_data_path}")

            # Return file paths for further processing
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path,
            )

        except Exception as e:
            logger.error("[-] Error occurred during data ingestion")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()