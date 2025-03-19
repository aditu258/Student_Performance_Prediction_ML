import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """Reads a CSV file and returns a Pandas DataFrame"""
        try:
            logging.info(f"📥 Reading CSV file from: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(f"❌ Error reading CSV: {str(e)}", sys)
        
    def get_data_transformer_object(self) -> ColumnTransformer:
        """Creates a ColumnTransformer with pipelines for numerical and categorical features"""
        try:
            # Exclude the target column from numerical columns
            numerical_columns = ["reading_score", "writing_score"]
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),  # Handle missing values
                ("scaler", StandardScaler())  # Normalize numerical features
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing categorical values
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),  # One-hot encode categorical data
                ("scaler", StandardScaler(with_mean=False))  # Scale categorical features
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            return preprocessor
        except Exception as e:
            raise NetworkSecurityException(f"❌ Error in data transformer object: {str(e)}", sys)
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """Applies data transformation to training and testing datasets"""
        try:
            logging.info("🚀 Starting Data Transformation...")

            # Load train and test data
            train_file_path = self.data_validation_artifact.valid_train_file_path
            test_file_path = self.data_validation_artifact.valid_test_file_path

            if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
                raise NetworkSecurityException("❌ Train/Test file not found!", sys)

            train_df = self.read_data(train_file_path)
            test_df = self.read_data(test_file_path)

            # Strip leading/trailing spaces from column names
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            # ✅ Debugging: Print dataset info
            logging.info(f"📝 Train DataFrame Columns: {train_df.columns.tolist()}")
            logging.info(f"📝 Test DataFrame Columns: {test_df.columns.tolist()}")
            logging.info(f"📝 Train DataFrame Head:\n{train_df.head()}")
            logging.info(f"📝 Test DataFrame Head:\n{test_df.head()}")

            # 🛑 Check if required columns exist
            required_columns = ["math_score", "reading_score", "writing_score", "gender", "race_ethnicity", 
                                "parental_level_of_education", "lunch", "test_preparation_course", TARGET_COLUMN]
            missing_cols_train = [col for col in required_columns if col not in train_df.columns]
            missing_cols_test = [col for col in required_columns if col not in test_df.columns]

            if missing_cols_train or missing_cols_test:
                raise NetworkSecurityException(
                    f"❌ Missing columns in Train DataFrame: {missing_cols_train}, Test DataFrame: {missing_cols_test}", 
                    sys
                )

            # Drop target column from features (if it's not required for transformation)
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # ✅ Debugging: Print input feature DataFrame info
            logging.info(f"🛠️ Input Feature Train DataFrame Columns: {input_feature_train_df.columns.tolist()}")
            logging.info(f"🛠️ Input Feature Train DataFrame Head:\n{input_feature_train_df.head()}")
            logging.info(f"🛠️ Input Feature Test DataFrame Columns: {input_feature_test_df.columns.tolist()}")
            logging.info(f"🛠️ Input Feature Test DataFrame Head:\n{input_feature_test_df.head()}")

            # Get preprocessor object
            preprocessor = self.get_data_transformer_object()
            
            # ✅ Debugging: Print preprocessor details
            logging.info(f"🛠️ Preprocessor Object: {preprocessor}")
            logging.info(f"🛠️ Numerical Columns: {preprocessor.transformers[0][2]}")
            logging.info(f"🛠️ Categorical Columns: {preprocessor.transformers[1][2]}")

            # Apply transformation
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # ✅ Debugging: Print transformed data shape
            logging.info(f"🛠️ Transformed Train Data Shape: {input_feature_train_arr.shape}")
            logging.info(f"🛠️ Transformed Test Data Shape: {input_feature_test_arr.shape}")

            # Combine transformed features with target values
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save processed data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            logging.info("✅ Data Transformation completed successfully!")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise NetworkSecurityException(f"❌ Data Transformation Failed: {str(e)}", sys)