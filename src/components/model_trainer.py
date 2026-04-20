import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import sklearn
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_TYPES,
    NUMERICAL_COLUMNS,
    TARGET_COLUMN,
)
from src.logger import logging
from src.utils import evaluate_models, load_object, save_object

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MIN_MODEL_SCORE = 0.6
MODEL_PARAMS = {
    "Decision Tree": {
        "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
    },
    "Random Forest": {
        "n_estimators": [8, 16, 32, 64, 128, 256],
    },
    "Gradient Boosting": {
        "learning_rate": [0.1, 0.01, 0.05, 0.001],
        "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        "n_estimators": [8, 16, 32, 64, 128, 256],
    },
    "Linear Regression": {},
    "AdaBoost Regressor": {
        "learning_rate": [0.1, 0.01, 0.5, 0.001],
        "n_estimators": [8, 16, 32, 64, 128, 256],
    },
}


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")
    preprocessor_file_path = os.path.join(PROJECT_ROOT, "artifacts", "preprocessor.pkl")
    onnx_model_file_path = os.path.join(PROJECT_ROOT, "artifacts", "model.onnx")
    metadata_file_path = os.path.join(PROJECT_ROOT, "artifacts", "model_metadata.json")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def get_models(self):
        """Return the candidate regressors used during model selection."""
        return {
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "AdaBoost Regressor": AdaBoostRegressor(),
        }

    def export_onnx_pipeline(self, *, best_model, best_model_name, best_model_score):
        """Export the fitted preprocessor and selected model as one ONNX graph."""
        try:
            from onnxmltools.utils import save_model
            from skl2onnx import convert_sklearn, get_latest_tested_opset_version
            from skl2onnx.common.data_types import FloatTensorType, StringTensorType
        except ImportError as exc:
            raise RuntimeError(
                "ONNX export dependencies are missing. Install requirements-training.txt "
                "before running the training pipeline."
            ) from exc

        preprocessor = load_object(self.model_trainer_config.preprocessor_file_path)
        serving_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", best_model),
            ]
        )

        initial_types = [
            *[(column, FloatTensorType([None, 1])) for column in NUMERICAL_COLUMNS],
            *[(column, StringTensorType([None, 1])) for column in CATEGORICAL_COLUMNS],
        ]
        target_opset = get_latest_tested_opset_version()

        logging.info("Exporting ONNX model to %s", self.model_trainer_config.onnx_model_file_path)
        onnx_model = convert_sklearn(
            serving_pipeline,
            "student_score_pipeline",
            initial_types=initial_types,
            target_opset={"": target_opset, "ai.onnx.ml": 1},
        )
        save_model(onnx_model, self.model_trainer_config.onnx_model_file_path)

        metadata = {
            "artifact_format": {
                "serving": "onnx",
                "fallback": "pickle",
            },
            "feature_names": FEATURE_COLUMNS,
            "feature_types": FEATURE_TYPES,
            "onnx_inputs": [
                {"name": column, "type": "float", "shape": [None, 1]}
                for column in NUMERICAL_COLUMNS
            ]
            + [
                {"name": column, "type": "string", "shape": [None, 1]}
                for column in CATEGORICAL_COLUMNS
            ],
            "target_column": TARGET_COLUMN,
            "model_name": best_model_name,
            "r2_score": float(best_model_score),
            "sklearn_version": sklearn.__version__,
            "onnx_opset": target_opset,
            "onnx_ml_opset": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.model_trainer_config.metadata_file_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, indent=2)
            file_obj.write("\n")

    def initiate_model_trainer(self, train_array, test_array):
        """Train, select, and persist the best-performing regression model."""
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = self.get_models()

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=MODEL_PARAMS,
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < MIN_MODEL_SCORE:
                raise ValueError("No best model found")
            logging.info("Best model found: %s with score %.4f", best_model_name, best_model_score)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predictions = best_model.predict(X_test)
            final_score = r2_score(y_test, predictions)
            self.export_onnx_pipeline(
                best_model=best_model,
                best_model_name=best_model_name,
                best_model_score=final_score,
            )

            return final_score
        except Exception as e:
            raise CustomException(e, sys)
