import os
import sys
from dataclasses import dataclass

# Get current file directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))

# Add PROJECT_ROOT to Python path (NOT src)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation

import pandas as pd

from sklearn.model_selection import train_test_split


# Dataset path
DATA_SOURCE_PATH = os.path.join(
    PROJECT_ROOT,
    "src",
    "notebook",
    "data",
    "stud.csv"
)


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(PROJECT_ROOT, "artifacts", "train.csv")
    test_data_path: str = os.path.join(PROJECT_ROOT, "artifacts", "test.csv")
    raw_data_path: str = os.path.join(PROJECT_ROOT, "artifacts", "raw.csv")
    source_data_path: str = DATA_SOURCE_PATH


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered the Data Ingestion component")

        try:
            # Read dataset
            df = pd.read_csv(self.ingestion_config.source_data_path)
            logging.info("Dataset loaded successfully")

            # Create artifacts directory
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            # Save raw data
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info("Train-Test split started")

            # Split dataset
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Save train and test datasets
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info("Data Ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_path=train_path, test_path=test_path, target_column_name="math_score")

    print("Train File :", train_path)
    print("Test File  :", test_path)    # ...existing code...
   