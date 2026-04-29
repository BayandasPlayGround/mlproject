import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor

from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_TYPES,
    NUMERICAL_COLUMNS,
    TARGET_COLUMN,
)
from src.logger import logging
from src.utils import save_object

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MIN_MODEL_SCORE = 0.6
RANDOM_STATE = 42
CV_FOLDS = 5
RANDOM_SEARCH_ITERATIONS = 30
MAX_SHORTLIST_SIZE = 5
MIN_DIVERSE_MODEL_SCORE = 0.80
ENSEMBLE_R2_IMPROVEMENT = 0.002
ENSEMBLE_R2_DROP_TOLERANCE = 0.001
PREVIOUS_TEST_R2_REFERENCE = 0.8804
MEANINGFUL_REGRESSION_THRESHOLD = 0.01
TREE_OR_BOOSTING_MODELS = {
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "AdaBoost",
}


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")
    preprocessor_file_path = os.path.join(PROJECT_ROOT, "artifacts", "preprocessor.pkl")
    onnx_model_file_path = os.path.join(PROJECT_ROOT, "artifacts", "model.onnx")
    metadata_file_path = os.path.join(PROJECT_ROOT, "artifacts", "model_metadata.json")
    model_selection_report_file_path = os.path.join(
        PROJECT_ROOT,
        "artifacts",
        "model_selection_report.json",
    )


def identity_transformer():
    """Return a cloneable no-op transformer that keeps feature names for reporting."""
    return FunctionTransformer(feature_names_out="one-to-one")


def json_safe(value):
    """Convert sklearn/numpy/pandas objects into JSON-safe report values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (pd.Interval, pd.Timestamp)):
        return str(value)

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    if hasattr(value, "get_params"):
        return {
            "class": value.__class__.__name__,
            "params": json_safe(value.get_params(deep=False)),
        }

    return str(value)


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        self.scoring = {
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        }
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

    def get_models(self):
        """Return broad ONNX-safe sklearn candidates for model screening."""
        return {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=RANDOM_STATE),
            "ElasticNet": ElasticNet(random_state=RANDOM_STATE, max_iter=10000),
            "Linear SVR": LinearSVR(random_state=RANDOM_STATE, max_iter=20000),
            "SVR": SVR(),
            "KNN": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "Random Forest": RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_estimators=100,
                n_jobs=1,
            ),
            "Extra Trees": ExtraTreesRegressor(
                random_state=RANDOM_STATE,
                n_estimators=100,
                n_jobs=1,
            ),
            "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
            "AdaBoost": AdaBoostRegressor(random_state=RANDOM_STATE),
            "MLP": MLPRegressor(
                random_state=RANDOM_STATE,
                hidden_layer_sizes=(32,),
                max_iter=1000,
            ),
        }

    def get_preprocessor(self):
        """Create a fresh preprocessing pipeline for CV-safe model pipelines."""
        return DataTransformation().get_data_transformer_object()

    def build_pipeline(self, model):
        return Pipeline(
            steps=[
                ("preprocessor", self.get_preprocessor()),
                ("model", clone(model)),
            ]
        )

    def get_preprocessing_search_space(self):
        return {
            "preprocessor__num_pipeline__imputer__strategy": ["mean", "median"],
            "preprocessor__num_pipeline__scaler": [
                StandardScaler(),
                MinMaxScaler(),
                RobustScaler(),
                identity_transformer(),
            ],
            "preprocessor__cat_pipelines__one_hot_encoder": [
                OneHotEncoder(handle_unknown="ignore"),
                OneHotEncoder(handle_unknown="ignore", drop="first"),
            ],
            "preprocessor__cat_pipelines__scaler": [
                StandardScaler(with_mean=False),
                identity_transformer(),
            ],
        }

    def get_model_search_space(self, model_name):
        params = self.get_preprocessing_search_space()

        if model_name == "Ridge":
            params.update({"model__alpha": [0.01, 0.1, 1, 3, 10, 30, 100]})
        elif model_name == "Linear SVR":
            params.update(
                {
                    "model__C": [0.1, 0.3, 1, 3, 10],
                    "model__epsilon": [0, 0.1, 0.5, 1],
                    "model__loss": [
                        "epsilon_insensitive",
                        "squared_epsilon_insensitive",
                    ],
                }
            )
        elif model_name == "Gradient Boosting":
            params.update(
                {
                    "model__n_estimators": [50, 100, 150, 200],
                    "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
                    "model__max_depth": [2, 3, 4],
                    "model__subsample": [0.7, 0.85, 1.0],
                }
            )
        elif model_name in {"Random Forest", "Extra Trees"}:
            params.update(
                {
                    "model__n_estimators": [100, 200, 300],
                    "model__max_depth": [None, 4, 6, 8, 12],
                    "model__min_samples_leaf": [1, 2, 4],
                    "model__max_features": [0.5, "sqrt", 1.0],
                }
            )
        elif model_name == "MLP":
            params.update(
                {
                    "model__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)],
                    "model__alpha": [0.0001, 0.001, 0.01],
                    "model__learning_rate_init": [0.001, 0.003, 0.01],
                }
            )

        return params

    def read_training_data(self, train_path, test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        missing_train_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(train_df.columns)
        missing_test_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(test_df.columns)

        if missing_train_columns:
            raise ValueError(f"Train data is missing required columns: {sorted(missing_train_columns)}")
        if missing_test_columns:
            raise ValueError(f"Test data is missing required columns: {sorted(missing_test_columns)}")

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df[TARGET_COLUMN]

        return X_train, y_train, X_test, y_test

    def summarize_cv_scores(self, scores):
        return {
            "r2_mean": float(np.nanmean(scores["test_r2"])),
            "r2_std": float(np.nanstd(scores["test_r2"])),
            "mae_mean": float(-np.nanmean(scores["test_mae"])),
            "mae_std": float(np.nanstd(-scores["test_mae"])),
            "rmse_mean": float(-np.nanmean(scores["test_rmse"])),
            "rmse_std": float(np.nanstd(-scores["test_rmse"])),
            "fit_time_mean": float(np.nanmean(scores.get("fit_time", [np.nan]))),
            "score_time_mean": float(np.nanmean(scores.get("score_time", [np.nan]))),
        }

    def summarize_search_result(self, search):
        best_index = search.best_index_
        results = search.cv_results_
        return {
            "r2_mean": float(results["mean_test_r2"][best_index]),
            "r2_std": float(results["std_test_r2"][best_index]),
            "mae_mean": float(-results["mean_test_mae"][best_index]),
            "mae_std": float(results["std_test_mae"][best_index]),
            "rmse_mean": float(-results["mean_test_rmse"][best_index]),
            "rmse_std": float(results["std_test_rmse"][best_index]),
        }

    def run_screening(self, X_train, y_train, models):
        screening_report = []

        logging.info("Starting stage 1 screening for %d candidate models", len(models))
        for model_name, model in models.items():
            logging.info("Screening model: %s", model_name)
            pipeline = self.build_pipeline(model)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    scores = cross_validate(
                        pipeline,
                        X_train,
                        y_train,
                        cv=self.cv,
                        scoring=self.scoring,
                        n_jobs=1,
                        error_score="raise",
                    )

                metrics = self.summarize_cv_scores(scores)
                screening_report.append(
                    {
                        "model_name": model_name,
                        "status": "ok",
                        **metrics,
                    }
                )
                logging.info(
                    "Screened %s: CV R2 %.4f +/- %.4f",
                    model_name,
                    metrics["r2_mean"],
                    metrics["r2_std"],
                )
            except Exception as exc:
                logging.warning("Screening failed for %s: %s", model_name, exc)
                screening_report.append(
                    {
                        "model_name": model_name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

        return screening_report

    def select_shortlist(self, screening_report):
        successful = sorted(
            [result for result in screening_report if result.get("status") == "ok"],
            key=lambda result: result["r2_mean"],
            reverse=True,
        )

        if not successful:
            raise ValueError("No model completed stage 1 screening successfully.")

        shortlist = successful[:MAX_SHORTLIST_SIZE]
        has_tree_or_boosting = any(
            result["model_name"] in TREE_OR_BOOSTING_MODELS for result in shortlist
        )

        if not has_tree_or_boosting:
            diverse_candidate = next(
                (
                    result
                    for result in successful
                    if result["model_name"] in TREE_OR_BOOSTING_MODELS
                    and result["r2_mean"] >= MIN_DIVERSE_MODEL_SCORE
                ),
                None,
            )
            if diverse_candidate is not None:
                if len(shortlist) >= MAX_SHORTLIST_SIZE:
                    shortlist[-1] = diverse_candidate
                else:
                    shortlist.append(diverse_candidate)

        deduplicated = {}
        for result in shortlist:
            deduplicated[result["model_name"]] = result

        return sorted(deduplicated.values(), key=lambda result: result["r2_mean"], reverse=True)

    def tune_shortlist(self, X_train, y_train, models, shortlist):
        tuned_models = []
        tuning_report = []

        logging.info("Starting stage 2 tuning for %d shortlisted models", len(shortlist))
        for shortlist_result in shortlist:
            model_name = shortlist_result["model_name"]
            logging.info("Tuning model: %s", model_name)

            search = RandomizedSearchCV(
                estimator=self.build_pipeline(models[model_name]),
                param_distributions=self.get_model_search_space(model_name),
                n_iter=RANDOM_SEARCH_ITERATIONS,
                scoring=self.scoring,
                refit="r2",
                cv=self.cv,
                random_state=RANDOM_STATE,
                n_jobs=1,
                error_score=np.nan,
                return_train_score=False,
            )

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    search.fit(X_train, y_train)

                metrics = self.summarize_search_result(search)
                if not math.isfinite(metrics["r2_mean"]):
                    raise ValueError("RandomizedSearchCV returned a non-finite best R2 score.")

                report_item = {
                    "model_name": model_name,
                    "status": "ok",
                    "cv_metrics": metrics,
                    "best_params": json_safe(search.best_params_),
                    "screening_r2_mean": shortlist_result["r2_mean"],
                }
                tuning_report.append(report_item)
                tuned_models.append(
                    {
                        "model_name": model_name,
                        "estimator": search.best_estimator_,
                        "cv_metrics": metrics,
                        "best_params": json_safe(search.best_params_),
                    }
                )
                logging.info("Tuned %s: CV R2 %.4f", model_name, metrics["r2_mean"])
            except Exception as exc:
                logging.warning("Tuning failed for %s: %s", model_name, exc)
                tuning_report.append(
                    {
                        "model_name": model_name,
                        "status": "failed",
                        "error": str(exc),
                        "screening_r2_mean": shortlist_result["r2_mean"],
                    }
                )

        if not tuned_models:
            raise ValueError("No model completed randomized search successfully.")

        return (
            sorted(tuned_models, key=lambda result: result["cv_metrics"]["r2_mean"], reverse=True),
            tuning_report,
        )

    def get_voting_weight_sets(self, model_count):
        if model_count == 2:
            return [None, [2, 1], [1, 2]]

        return [
            None,
            [2, 1, 1],
            [1, 2, 1],
            [1, 1, 2],
            [3, 2, 1],
        ]

    def make_voting_pipeline(self, tuned_models, weights):
        estimators = []
        for index, tuned_model in enumerate(tuned_models):
            estimator_name = f"model_{index}_{tuned_model['model_name'].lower().replace(' ', '_')}"
            estimators.append(
                (
                    estimator_name,
                    clone(tuned_model["estimator"].named_steps["model"]),
                )
            )

        shared_preprocessor = clone(tuned_models[0]["estimator"].named_steps["preprocessor"])
        voting_regressor = VotingRegressor(estimators=estimators, weights=weights, n_jobs=1)

        return Pipeline(
            steps=[
                ("preprocessor", shared_preprocessor),
                ("model", voting_regressor),
            ]
        )

    def try_voting_ensemble(self, X_train, y_train, tuned_models):
        top_models = tuned_models[:3]
        attempts = []

        if len(top_models) < 2:
            return None, {"attempts": attempts, "selected": None}

        best_individual = tuned_models[0]
        best_individual_r2 = best_individual["cv_metrics"]["r2_mean"]
        best_individual_mae = best_individual["cv_metrics"]["mae_mean"]
        best_candidate = None

        logging.info("Evaluating VotingRegressor ensembles from the top tuned models")
        for weights in self.get_voting_weight_sets(len(top_models)):
            weight_label = "equal" if weights is None else weights
            pipeline = self.make_voting_pipeline(top_models, weights)

            try:
                scores = cross_validate(
                    pipeline,
                    X_train,
                    y_train,
                    cv=self.cv,
                    scoring=self.scoring,
                    n_jobs=1,
                    error_score="raise",
                )
                metrics = self.summarize_cv_scores(scores)
                attempt = {
                    "model_name": "Voting Regressor",
                    "base_models": [model["model_name"] for model in top_models],
                    "weights": weight_label,
                    "status": "ok",
                    "cv_metrics": metrics,
                }
                attempts.append(attempt)

                improves_r2 = metrics["r2_mean"] >= best_individual_r2 + ENSEMBLE_R2_IMPROVEMENT
                improves_mae_without_r2_drop = (
                    metrics["mae_mean"] <= best_individual_mae
                    and metrics["r2_mean"] >= best_individual_r2 - ENSEMBLE_R2_DROP_TOLERANCE
                )

                if improves_r2 or improves_mae_without_r2_drop:
                    if best_candidate is None or metrics["r2_mean"] > best_candidate["cv_metrics"]["r2_mean"]:
                        best_candidate = {
                            "model_name": "Voting Regressor",
                            "estimator": pipeline,
                            "cv_metrics": metrics,
                            "best_params": {
                                "base_models": [model["model_name"] for model in top_models],
                                "weights": weight_label,
                                "shared_preprocessor_from": top_models[0]["model_name"],
                            },
                        }
            except Exception as exc:
                logging.warning("VotingRegressor failed with weights %s: %s", weight_label, exc)
                attempts.append(
                    {
                        "model_name": "Voting Regressor",
                        "base_models": [model["model_name"] for model in top_models],
                        "weights": weight_label,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

        if best_candidate is not None:
            best_candidate["estimator"].fit(X_train, y_train)
            return best_candidate, {
                "attempts": attempts,
                "selected": {
                    "model_name": best_candidate["model_name"],
                    "cv_metrics": best_candidate["cv_metrics"],
                    "best_params": best_candidate["best_params"],
                },
            }

        return None, {"attempts": attempts, "selected": None}

    def select_final_estimator(self, X_train, y_train, tuned_models):
        best_tuned_model = tuned_models[0]
        ensemble_candidate, ensemble_report = self.try_voting_ensemble(
            X_train,
            y_train,
            tuned_models,
        )

        if ensemble_candidate is not None:
            logging.info(
                "Selected VotingRegressor with CV R2 %.4f",
                ensemble_candidate["cv_metrics"]["r2_mean"],
            )
            return ensemble_candidate, ensemble_report

        logging.info(
            "Selected %s with CV R2 %.4f",
            best_tuned_model["model_name"],
            best_tuned_model["cv_metrics"]["r2_mean"],
        )
        return best_tuned_model, ensemble_report

    def calculate_test_metrics(self, estimator, X_test, y_test):
        predictions = estimator.predict(X_test)
        return {
            "r2": float(r2_score(y_test, predictions)),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        }

    def get_transformed_feature_names(self, preprocessor):
        try:
            return [str(name) for name in preprocessor.get_feature_names_out()]
        except Exception as exc:
            logging.warning("Could not get transformed feature names: %s", exc)
            return []

    def native_model_importance(self, model, transformed_feature_names):
        native_report = {}

        if hasattr(model, "coef_"):
            coefficients = np.ravel(model.coef_)
            native_report["coefficients"] = self.rank_named_values(
                transformed_feature_names,
                coefficients,
                sort_by_absolute_value=True,
            )

        if hasattr(model, "feature_importances_"):
            native_report["feature_importances"] = self.rank_named_values(
                transformed_feature_names,
                model.feature_importances_,
                sort_by_absolute_value=False,
            )

        return native_report

    def rank_named_values(self, names, values, sort_by_absolute_value):
        ranked = []
        for index, value in enumerate(np.ravel(values)):
            name = names[index] if index < len(names) else f"feature_{index}"
            ranked.append({"feature": name, "value": float(value)})

        sort_key = (
            (lambda item: abs(item["value"]))
            if sort_by_absolute_value
            else (lambda item: item["value"])
        )
        return sorted(ranked, key=sort_key, reverse=True)

    def compute_permutation_importance(self, estimator, X_train, y_train):
        try:
            result = permutation_importance(
                estimator,
                X_train,
                y_train,
                scoring="r2",
                n_repeats=5,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        except Exception as exc:
            return {"status": "failed", "error": str(exc)}

        importances = []
        for index, feature in enumerate(X_train.columns):
            importances.append(
                {
                    "feature": feature,
                    "importance_mean": float(result.importances_mean[index]),
                    "importance_std": float(result.importances_std[index]),
                }
            )

        return {
            "status": "ok",
            "importances": sorted(
                importances,
                key=lambda item: item["importance_mean"],
                reverse=True,
            ),
        }

    def build_feature_importance_report(self, final_result, tuned_models, X_train, y_train):
        final_estimator = final_result["estimator"]
        preprocessor = final_estimator.named_steps["preprocessor"]
        model = final_estimator.named_steps["model"]
        transformed_feature_names = self.get_transformed_feature_names(preprocessor)

        permutation_by_model = []
        native_importance_by_model = []
        for tuned_model in tuned_models:
            tuned_estimator = tuned_model["estimator"]
            tuned_feature_names = self.get_transformed_feature_names(
                tuned_estimator.named_steps["preprocessor"]
            )
            native_importance_by_model.append(
                {
                    "model_name": tuned_model["model_name"],
                    "transformed_feature_names": tuned_feature_names,
                    "native_importance": self.native_model_importance(
                        tuned_estimator.named_steps["model"],
                        tuned_feature_names,
                    ),
                }
            )
            permutation_by_model.append(
                {
                    "model_name": tuned_model["model_name"],
                    "permutation_importance": self.compute_permutation_importance(
                        tuned_estimator,
                        X_train,
                        y_train,
                    ),
                }
            )

        return {
            "transformed_feature_names": transformed_feature_names,
            "native_final_model_importance": self.native_model_importance(
                model,
                transformed_feature_names,
            ),
            "final_model_permutation_importance": self.compute_permutation_importance(
                final_estimator,
                X_train,
                y_train,
            ),
            "shortlisted_model_native_importance": native_importance_by_model,
            "shortlisted_model_permutation_importance": permutation_by_model,
        }

    def build_residual_diagnostics(self, estimator, X_train, y_train):
        predictions = cross_val_predict(
            estimator,
            X_train,
            y_train,
            cv=self.cv,
            n_jobs=1,
        )
        residuals = np.asarray(y_train) - np.asarray(predictions)
        diagnostics_df = X_train.copy()
        diagnostics_df["_actual"] = np.asarray(y_train)
        diagnostics_df["_prediction"] = np.asarray(predictions)
        diagnostics_df["_residual"] = residuals
        diagnostics_df["_abs_error"] = np.abs(residuals)

        largest_errors = []
        for index, row in diagnostics_df.nlargest(10, "_abs_error").iterrows():
            item = {
                "row_index": int(index),
                "actual": float(row["_actual"]),
                "prediction": float(row["_prediction"]),
                "residual": float(row["_residual"]),
                "absolute_error": float(row["_abs_error"]),
            }
            for column in FEATURE_COLUMNS:
                item[column] = json_safe(row[column])
            largest_errors.append(item)

        group_summary = {}
        for column in CATEGORICAL_COLUMNS:
            grouped = diagnostics_df.groupby(column)["_residual"].agg(["count", "mean", "std"])
            group_summary[column] = json_safe(grouped.reset_index().to_dict(orient="records"))

        score_band_summary = {}
        bins = [-math.inf, 59, 69, 79, 89, math.inf]
        labels = ["0-59", "60-69", "70-79", "80-89", "90-100"]
        for column in NUMERICAL_COLUMNS:
            band_column = f"{column}_band"
            diagnostics_df[band_column] = pd.cut(diagnostics_df[column], bins=bins, labels=labels)
            grouped = diagnostics_df.groupby(band_column, observed=False)["_residual"].agg(
                ["count", "mean", "std"]
            )
            score_band_summary[column] = json_safe(grouped.reset_index().to_dict(orient="records"))

        return {
            "out_of_fold_metrics": {
                "r2": float(r2_score(y_train, predictions)),
                "mae": float(mean_absolute_error(y_train, predictions)),
                "rmse": float(np.sqrt(mean_squared_error(y_train, predictions))),
                "residual_mean": float(np.mean(residuals)),
                "residual_std": float(np.std(residuals)),
            },
            "largest_absolute_errors": largest_errors,
            "residual_mean_by_categorical_group": group_summary,
            "residual_mean_by_score_band": score_band_summary,
        }

    def build_residual_correlation_report(self, tuned_models, X_train, y_train):
        residual_frame = pd.DataFrame(index=X_train.index)

        for tuned_model in tuned_models:
            try:
                predictions = cross_val_predict(
                    tuned_model["estimator"],
                    X_train,
                    y_train,
                    cv=self.cv,
                    n_jobs=1,
                )
                residual_frame[tuned_model["model_name"]] = np.asarray(y_train) - np.asarray(predictions)
            except Exception as exc:
                logging.warning(
                    "Residual correlation skipped for %s: %s",
                    tuned_model["model_name"],
                    exc,
                )

        if residual_frame.shape[1] < 2:
            return {"status": "skipped", "reason": "Fewer than two residual series were available."}

        return {
            "status": "ok",
            "models": list(residual_frame.columns),
            "correlation": json_safe(residual_frame.corr().round(4).to_dict()),
        }

    def export_onnx_pipeline(self, fitted_pipeline):
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

        initial_types = [
            *[(column, FloatTensorType([None, 1])) for column in NUMERICAL_COLUMNS],
            *[(column, StringTensorType([None, 1])) for column in CATEGORICAL_COLUMNS],
        ]
        target_opset = get_latest_tested_opset_version()

        logging.info("Exporting ONNX model to %s", self.model_trainer_config.onnx_model_file_path)
        onnx_model = convert_sklearn(
            fitted_pipeline,
            "student_score_pipeline",
            initial_types=initial_types,
            target_opset={"": target_opset, "ai.onnx.ml": 1},
        )
        save_model(onnx_model, self.model_trainer_config.onnx_model_file_path)
        return target_opset

    def build_metadata(self, final_result, test_metrics, target_opset):
        cv_metrics = final_result["cv_metrics"]
        test_r2_delta = test_metrics["r2"] - PREVIOUS_TEST_R2_REFERENCE

        return {
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
            "model_name": final_result["model_name"],
            "r2_score": float(test_metrics["r2"]),
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
            "selection": {
                "method": "screening_plus_randomized_search",
                "cv_folds": CV_FOLDS,
                "random_search_iterations": RANDOM_SEARCH_ITERATIONS,
                "random_state": RANDOM_STATE,
                "best_params": final_result.get("best_params", {}),
            },
            "baseline_comparison": {
                "previous_test_r2_reference": PREVIOUS_TEST_R2_REFERENCE,
                "test_r2_delta": float(test_r2_delta),
                "meaningful_regression_threshold": MEANINGFUL_REGRESSION_THRESHOLD,
                "possible_regression": bool(test_r2_delta < -MEANINGFUL_REGRESSION_THRESHOLD),
            },
            "sklearn_version": sklearn.__version__,
            "onnx_opset": target_opset,
            "onnx_ml_opset": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def write_json(self, file_path, payload):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file_obj:
            json.dump(json_safe(payload), file_obj, indent=2)
            file_obj.write("\n")

    def build_model_selection_report(
        self,
        screening_report,
        shortlist,
        tuning_report,
        tuned_models,
        ensemble_report,
        final_result,
        test_metrics,
        feature_importance_report,
        residual_diagnostics,
        residual_correlation_report,
    ):
        return {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": {
                "cv_folds": CV_FOLDS,
                "random_state": RANDOM_STATE,
                "random_search_iterations": RANDOM_SEARCH_ITERATIONS,
                "max_shortlist_size": MAX_SHORTLIST_SIZE,
                "minimum_model_score": MIN_MODEL_SCORE,
                "minimum_diverse_model_score": MIN_DIVERSE_MODEL_SCORE,
            },
            "screening": screening_report,
            "shortlist": [
                {
                    "model_name": result["model_name"],
                    "r2_mean": result["r2_mean"],
                    "r2_std": result["r2_std"],
                    "mae_mean": result["mae_mean"],
                    "rmse_mean": result["rmse_mean"],
                }
                for result in shortlist
            ],
            "tuning": tuning_report,
            "tuned_model_ranking": [
                {
                    "model_name": tuned_model["model_name"],
                    "cv_metrics": tuned_model["cv_metrics"],
                    "best_params": tuned_model["best_params"],
                }
                for tuned_model in tuned_models
            ],
            "ensemble": ensemble_report,
            "final_model": {
                "model_name": final_result["model_name"],
                "cv_metrics": final_result["cv_metrics"],
                "test_metrics": test_metrics,
                "best_params": final_result.get("best_params", {}),
            },
            "feature_importance": feature_importance_report,
            "residual_diagnostics": residual_diagnostics,
            "residual_correlation": residual_correlation_report,
        }

    def persist_artifacts(self, fitted_pipeline):
        save_object(
            file_path=self.model_trainer_config.preprocessor_file_path,
            obj=fitted_pipeline.named_steps["preprocessor"],
        )
        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=fitted_pipeline.named_steps["model"],
        )

    def initiate_model_trainer(self, train_path, test_path):
        """Train, select, and persist the best-performing regression pipeline."""
        try:
            logging.info("Reading raw train and test data for model training")
            X_train, y_train, X_test, y_test = self.read_training_data(train_path, test_path)
            models = self.get_models()

            screening_report = self.run_screening(X_train, y_train, models)
            shortlist = self.select_shortlist(screening_report)
            logging.info(
                "Shortlisted models: %s",
                ", ".join(result["model_name"] for result in shortlist),
            )

            tuned_models, tuning_report = self.tune_shortlist(
                X_train,
                y_train,
                models,
                shortlist,
            )
            final_result, ensemble_report = self.select_final_estimator(
                X_train,
                y_train,
                tuned_models,
            )
            fitted_pipeline = final_result["estimator"]

            test_metrics = self.calculate_test_metrics(fitted_pipeline, X_test, y_test)
            if test_metrics["r2"] < MIN_MODEL_SCORE:
                raise ValueError(
                    f"No acceptable model found. Best test R2 was {test_metrics['r2']:.4f}."
                )

            logging.info(
                "Best model found: %s with test R2 %.4f",
                final_result["model_name"],
                test_metrics["r2"],
            )

            self.persist_artifacts(fitted_pipeline)
            target_opset = self.export_onnx_pipeline(fitted_pipeline)

            metadata = self.build_metadata(final_result, test_metrics, target_opset)
            self.write_json(self.model_trainer_config.metadata_file_path, metadata)

            feature_importance_report = self.build_feature_importance_report(
                final_result,
                tuned_models,
                X_train,
                y_train,
            )
            residual_diagnostics = self.build_residual_diagnostics(
                fitted_pipeline,
                X_train,
                y_train,
            )
            residual_correlation_report = self.build_residual_correlation_report(
                tuned_models,
                X_train,
                y_train,
            )
            model_selection_report = self.build_model_selection_report(
                screening_report,
                shortlist,
                tuning_report,
                tuned_models,
                ensemble_report,
                final_result,
                test_metrics,
                feature_importance_report,
                residual_diagnostics,
                residual_correlation_report,
            )
            self.write_json(
                self.model_trainer_config.model_selection_report_file_path,
                model_selection_report,
            )

            return test_metrics["r2"]
        except Exception as e:
            raise CustomException(e, sys)
