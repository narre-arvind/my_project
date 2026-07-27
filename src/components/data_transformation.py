import os
import sys
from dataclasses import dataclass

# Get current file directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root is two levels above src/components
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
# Ensure the project root is on the import path so src imports work
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        PROJECT_ROOT,
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            numerical_features = ["writing_score", "reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            logging.info(f"Numerical columns: {numerical_features}")
            logging.info(f"Categorical columns: {categorical_features}")

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path: str,
        test_path: str,
        target_column_name: str = "math_score"
    ):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded training and test data for transformation.")
            logging.info("Obtaining preprocessing object.")

            preprocessor = self.get_data_transformer_object()

            input_features_train = train_df.drop(columns=[target_column_name])
            target_feature_train = train_df[target_column_name]
            input_features_test = test_df.drop(columns=[target_column_name])
            target_feature_test = test_df[target_column_name]

            logging.info("Applying transformation pipeline to train data.")
            input_feature_train_array = preprocessor.fit_transform(input_features_train)

            logging.info("Applying transformation pipeline to test data.")
            input_feature_test_array = preprocessor.transform(input_features_test)

            save_object(self.config.preprocessor_obj_file_path, preprocessor)

            train_arr = np.c_[input_feature_train_array, np.array(target_feature_train)]
            test_arr = np.c_[input_feature_test_array, np.array(target_feature_test)]

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print("Train File :", train_path)
    print("Test File  :", test_path)

    # Data Transformation
    print("\n--- Starting Data Transformation ---")
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)
    print("Data Transformation completed!")
    print("Preprocessor saved at:", preprocessor_path)

    # Model Training
    print("\n--- Starting Model Training ---")
    model_trainer = ModelTrainer()
    r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path)
    print(f"Model R² Score: {r2_score}")
    print("Model Training completed!")