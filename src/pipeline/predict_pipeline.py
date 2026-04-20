"""Load persisted artifacts and generate predictions for new inputs."""
from collections.abc import Mapping
import os
import sys
import threading
from dataclasses import dataclass

import numpy as np

from src.exception import CustomException
from src.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERICAL_COLUMNS
from src.logger import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ONNX_MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model.onnx")
PICKLE_MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")
MODEL_PATH = PICKLE_MODEL_PATH
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "artifacts", "preprocessor.pkl")
ARTIFACT_LOCK = threading.Lock()
SESSION_LOCK = threading.Lock()
ONNX_SESSION = None


def get_model_runtime():
    runtime = os.getenv("MODEL_RUNTIME", "onnx").strip().lower()
    if runtime not in {"onnx", "pickle"}:
        raise ValueError("MODEL_RUNTIME must be either 'onnx' or 'pickle'.")
    return runtime


def allow_artifact_rebuild():
    raw_value = os.getenv("ALLOW_ARTIFACT_REBUILD", "1").strip().lower()
    return raw_value not in {"0", "false", "no"}


def required_artifact_paths(runtime):
    if runtime == "onnx":
        return [ONNX_MODEL_PATH]
    return [PICKLE_MODEL_PATH, PREPROCESSOR_PATH]


def missing_artifact_paths(runtime):
    return [path for path in required_artifact_paths(runtime) if not os.path.exists(path)]


def ensure_prediction_artifacts(runtime=None):
    """Build saved prediction artifacts locally if they are missing."""
    runtime = runtime or get_model_runtime()

    if not missing_artifact_paths(runtime):
        return

    if not allow_artifact_rebuild():
        missing_files = missing_artifact_paths(runtime)
        if runtime == "onnx":
            raise RuntimeError(
                "ONNX artifact missing and automatic rebuild is disabled. "
                f"Missing files: {missing_files}"
            )
        raise RuntimeError(
            "Pickle prediction artifacts are missing and automatic rebuild is disabled. "
            f"Missing files: {missing_files}"
        )

    with ARTIFACT_LOCK:
        if not missing_artifact_paths(runtime):
            return

        logging.info("Prediction artifacts not found. Regenerating them in the active environment.")

        from src.components.data_ingestion import DataIngestion
        from src.components.data_transformation import DataTransformation
        from src.components.model_trainer import ModelTrainer

        train_path, test_path = DataIngestion().initiate_data_ingestion()
        train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(train_path, test_path)
        score = ModelTrainer().initiate_model_trainer(train_arr, test_arr)
        logging.info("Prediction artifacts regenerated successfully. Test R2 score: %.4f", score)


def get_onnx_session():
    global ONNX_SESSION

    if ONNX_SESSION is not None:
        return ONNX_SESSION

    with SESSION_LOCK:
        if ONNX_SESSION is not None:
            return ONNX_SESSION

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime is not installed. Install requirements.txt for serving, "
                "or set MODEL_RUNTIME=pickle in an environment that has training dependencies."
            ) from exc

        logging.info("Loading ONNX model from %s", ONNX_MODEL_PATH)
        ONNX_SESSION = ort.InferenceSession(
            ONNX_MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )
        return ONNX_SESSION


def extract_column(features, column):
    if isinstance(features, Mapping):
        if column not in features:
            raise KeyError(f"Missing required feature column: {column}")
        values = features[column]
    else:
        values = features[column]

    if hasattr(values, "to_numpy"):
        values = values.to_numpy()

    values_array = np.asarray(values)
    if values_array.ndim == 0:
        values_array = values_array.reshape(1)

    return values_array.reshape(-1)


def validate_batch_size(batch_size, column, values):
    if batch_size is None:
        return len(values)
    if len(values) != batch_size:
        raise ValueError(
            f"Feature column '{column}' has {len(values)} rows, expected {batch_size}."
        )
    return batch_size


def features_to_onnx_inputs(features):
    onnx_inputs = {}
    batch_size = None

    for column in NUMERICAL_COLUMNS:
        values = extract_column(features, column)
        batch_size = validate_batch_size(batch_size, column, values)
        try:
            onnx_inputs[column] = values.astype(np.float32).reshape(-1, 1)
        except ValueError as exc:
            raise ValueError(f"Feature column '{column}' must be numeric.") from exc

    for column in CATEGORICAL_COLUMNS:
        values = extract_column(features, column)
        batch_size = validate_batch_size(batch_size, column, values)
        onnx_inputs[column] = values.astype(str).reshape(-1, 1)

    return onnx_inputs


def features_to_dataframe(features):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Pickle runtime requires pandas and scikit-learn dependencies. "
            "Install requirements-training.txt or use the default MODEL_RUNTIME=onnx path."
        ) from exc

    if isinstance(features, pd.DataFrame):
        return features

    normalized_features = {
        column: extract_column(features, column)
        for column in FEATURE_COLUMNS
    }
    return pd.DataFrame(normalized_features)


class PredictPipeline:
    """Apply the active prediction runtime to incoming features."""

    def predict(self, features):
        try:
            runtime = get_model_runtime()
            if runtime == "pickle":
                return self.predict_with_pickle(features)
            return self.predict_with_onnx(features)
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

    def predict_with_onnx(self, features):
        ensure_prediction_artifacts(runtime="onnx")
        session = get_onnx_session()
        outputs = session.run(None, features_to_onnx_inputs(features))
        return np.asarray(outputs[0]).reshape(-1)

    def predict_with_pickle(self, features):
        from src.utils import load_object

        ensure_prediction_artifacts(runtime="pickle")
        logging.info("Loading pickle prediction artifacts")
        model = load_object(file_path=PICKLE_MODEL_PATH)
        preprocessor = load_object(file_path=PREPROCESSOR_PATH)
        transformed_features = preprocessor.transform(features_to_dataframe(features))
        return model.predict(transformed_features)


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
            return {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
        except Exception as e:
            raise CustomException(e, sys)

