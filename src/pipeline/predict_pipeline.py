"""Load persisted artifacts and generate predictions for new inputs."""
import os
import sys
import threading
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "artifacts", "preprocessor.pkl")
ARTIFACT_LOCK = threading.Lock()


def allow_artifact_rebuild():
    raw_value = os.getenv("ALLOW_ARTIFACT_REBUILD", "1").strip().lower()
    return raw_value not in {"0", "false", "no"}


def ensure_prediction_artifacts():
    """Build the saved model artifacts if they are missing in the runtime environment."""
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        return

    if not allow_artifact_rebuild():
        missing_files = [
            path
            for path in (MODEL_PATH, PREPROCESSOR_PATH)
            if not os.path.exists(path)
        ]
        raise RuntimeError(
            "Prediction artifacts are missing and automatic rebuild is disabled. "
            f"Missing files: {missing_files}"
        )

    with ARTIFACT_LOCK:
        if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
            return

        logging.info("Prediction artifacts not found. Regenerating them in the active environment.")

        from src.components.data_ingestion import DataIngestion
        from src.components.data_transformation import DataTransformation
        from src.components.model_trainer import ModelTrainer

        train_path, test_path = DataIngestion().initiate_data_ingestion()
        train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(train_path, test_path)
        score = ModelTrainer().initiate_model_trainer(train_arr, test_arr)
        logging.info("Prediction artifacts regenerated successfully. Test R2 score: %.4f", score)


class PredictPipeline:
    """Apply the saved preprocessor and model to incoming features."""

    def predict(self, features):
        try:
            ensure_prediction_artifacts()
            logging.info("Loading prediction artifacts")
            model = load_object(file_path=MODEL_PATH)
            preprocessor = load_object(file_path=PREPROCESSOR_PATH)

            transformed_features = preprocessor.transform(features)
            return model.predict(transformed_features)
        except Exception as e:
            if "_fill_dtype" in str(e):
                raise CustomException(
                    RuntimeError(
                        "Saved artifacts are incompatible with the current scikit-learn version. "
                        "Regenerate them by running 'python src\\components\\data_ingestion.py' "
                        "in the active virtual environment."
                    ),
                    sys,
                )
            raise CustomException(e, sys)


@dataclass
class CustomData:
    """Represent one prediction request in the model's expected feature schema."""

    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int

    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame(
                {
                    "gender": [self.gender],
                    "race_ethnicity": [self.race_ethnicity],
                    "parental_level_of_education": [self.parental_level_of_education],
                    "lunch": [self.lunch],
                    "test_preparation_course": [self.test_preparation_course],
                    "reading_score": [self.reading_score],
                    "writing_score": [self.writing_score],
                }
            )
        except Exception as e:
            raise CustomException(e, sys)

