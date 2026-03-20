"""Load persisted artifacts and generate predictions for new inputs."""
import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "artifacts", "preprocessor.pkl")


class PredictPipeline:
    """Apply the saved preprocessor and model to incoming features."""

    def predict(self, features):
        try:
            logging.info("Loading prediction artifacts")
            model = load_object(file_path=MODEL_PATH)
            preprocessor = load_object(file_path=PREPROCESSOR_PATH)

            transformed_features = preprocessor.transform(features)
            return model.predict(transformed_features)
        except Exception as e:
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

