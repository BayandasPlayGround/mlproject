import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object

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
            return r2_score(y_test, predictions)
        except Exception as e:
            raise CustomException(e, sys)
