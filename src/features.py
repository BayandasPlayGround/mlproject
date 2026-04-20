"""Shared feature schema for training, ONNX export, and inference."""

NUMERICAL_COLUMNS = ["writing_score", "reading_score"]
CATEGORICAL_COLUMNS = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
TARGET_COLUMN = "math_score"

FEATURE_TYPES = {
    "writing_score": "float",
    "reading_score": "float",
    "gender": "string",
    "race_ethnicity": "string",
    "parental_level_of_education": "string",
    "lunch": "string",
    "test_preparation_course": "string",
}
