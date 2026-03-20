# Student Exam Performance Predictor

This project is an end-to-end machine learning application that predicts a student's `math_score` from the rest of the student profile.

It includes:

- data ingestion
- preprocessing and feature transformation
- model training and model selection
- saved training artifacts
- a Flask frontend for interactive predictions

## Problem Summary

The model predicts `math_score` using these input features:

- `gender`
- `race_ethnicity`
- `parental_level_of_education`
- `lunch`
- `test_preparation_course`
- `reading_score`
- `writing_score`

The source dataset is stored in:

- `notebook/data/stud.csv`

## Project Structure

```text
mlproject/
|-- app.py
|-- requirements.txt
|-- README.md
|-- notebook/
|   |-- data/stud.csv
|   |-- EDA STUDENT PERFORMANCE.ipynb
|   `-- MODEL TRAINING.ipynb
|-- artifacts/
|   |-- data.csv
|   |-- train.csv
|   |-- test.csv
|   |-- preprocessor.pkl
|   `-- model.pkl
|-- logs/
|-- src/
|   |-- exception.py
|   |-- logger.py
|   |-- utils.py
|   |-- components/
|   |   |-- data_ingestion.py
|   |   |-- data_transformation.py
|   |   `-- model_trainer.py
|   `-- pipeline/
|       `-- predict_pipeline.py
`-- templates/
    |-- base.html
    |-- index.html
    `-- home.html
```

## Pipeline Overview

### 1. Data Ingestion

File:

- `src/components/data_ingestion.py`

Responsibilities:

- load `notebook/data/stud.csv`
- create the `artifacts/` directory if needed
- save a raw copy of the dataset to `artifacts/data.csv`
- split the dataset into train and test sets
- save:
  - `artifacts/train.csv`
  - `artifacts/test.csv`

### 2. Data Transformation

File:

- `src/components/data_transformation.py`

Responsibilities:

- define numeric and categorical feature groups
- build a `ColumnTransformer`
- apply:
  - median imputation + scaling for numeric features
  - most-frequent imputation + one-hot encoding + scaling for categorical features
- fit on training data only
- transform both train and test data
- save the fitted preprocessor to `artifacts/preprocessor.pkl`

### 3. Model Training

File:

- `src/components/model_trainer.py`

Responsibilities:

- receive transformed train/test arrays
- train multiple candidate regressors
- tune them with `GridSearchCV`
- compare test `R-squared` scores
- save the best model to `artifacts/model.pkl`

### 4. Prediction Pipeline

File:

- `src/pipeline/predict_pipeline.py`

Responsibilities:

- load `artifacts/preprocessor.pkl`
- load `artifacts/model.pkl`
- transform incoming user data
- return the predicted maths score

### 5. Flask Frontend

File:

- `app.py`

Routes:

- `/` -> landing page
- `/predictdata` -> prediction form and prediction result page

Templates:

- `templates/index.html`
- `templates/home.html`
- `templates/base.html`

## Setup

### 1. Clone the project

```powershell
git clone <your-repo-url>
cd mlproject
```

### 2. Create and activate a virtual environment

PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## How To Run The Project

### Option 1. Run the full training pipeline

This command:

- ingests the raw CSV
- creates train/test artifacts
- fits the preprocessing object
- trains the model
- saves the final artifacts
- prints the final test `R-squared` score

```powershell
python src\components\data_ingestion.py
```

Expected output will look similar to:

```text
Training completed. Test R2 score: 0.8804
```

### Option 2. Run the Flask application

Start the app:

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## How To Test The Project

There are five useful testing levels.

### 1. End-to-end training test

Run:

```powershell
python src\components\data_ingestion.py
```

What this confirms:

- the raw CSV can be loaded
- train/test split works
- transformation works
- model training works
- artifacts are saved correctly

After it completes, check that these files exist:

- `artifacts/data.csv`
- `artifacts/train.csv`
- `artifacts/test.csv`
- `artifacts/preprocessor.pkl`
- `artifacts/model.pkl`

### 2. Backend prediction smoke test

Run this from the project root:

```powershell
@'
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

sample = CustomData(
    gender="female",
    race_ethnicity="group B",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="none",
    reading_score=72,
    writing_score=74,
)

features = sample.get_data_as_data_frame()
prediction = PredictPipeline().predict(features)
print("Prediction:", prediction)
'@ | python -
```

What this confirms:

- saved artifacts can be loaded
- the preprocessor still matches the model
- the prediction pipeline works without the web app

### 3. Flask route smoke test

Run this from the project root:

```powershell
@'
from app import app

client = app.test_client()

resp_index = client.get("/")
print("GET /:", resp_index.status_code)

resp_form = client.get("/predictdata")
print("GET /predictdata:", resp_form.status_code)

resp_post = client.post("/predictdata", data={
    "gender": "female",
    "race_ethnicity": "group B",
    "parental_level_of_education": "bachelor's degree",
    "lunch": "standard",
    "test_preparation_course": "none",
    "reading_score": "72",
    "writing_score": "74",
})
print("POST /predictdata:", resp_post.status_code)
'@ | python -
```

Expected result:

- `GET /` returns `200`
- `GET /predictdata` returns `200`
- `POST /predictdata` returns `200`

### 4. Manual browser test

1. Run:

```powershell
python app.py
```

2. Open:

```text
http://127.0.0.1:5000
```

3. Click `Open Predictor`

4. Fill in the form with valid values, for example:

- Gender: `female`
- Race or Ethnicity: `group B`
- Parental Level of Education: `bachelor's degree`
- Lunch Type: `standard`
- Test Preparation Course: `none`
- Reading Score: `72`
- Writing Score: `74`

5. Submit the form

6. Confirm that:

- a maths score is displayed
- no template error appears
- the form preserves values after submission

### 5. Validation test

Test the form with bad inputs to confirm server-side validation works.

Examples:

- leave a dropdown empty
- set `reading_score` to `120`
- set `writing_score` to a non-numeric value through a custom request

Expected behavior:

- the page reloads
- validation messages are shown
- the app does not crash

## Current Entry Points

### Training

```powershell
python src\components\data_ingestion.py
```

### Web app

```powershell
python app.py
```

### Prediction logic only

Use:

- `src/pipeline/predict_pipeline.py`

Key classes:

- `PredictPipeline`
- `CustomData`

## Artifacts

The project saves these files in `artifacts/`:

- `data.csv` -> raw copied dataset
- `train.csv` -> training split
- `test.csv` -> test split
- `preprocessor.pkl` -> fitted preprocessing object
- `model.pkl` -> best trained model

If you change preprocessing logic or model logic, retrain the project so these artifacts are regenerated.

## Logs

Log files are written to the `logs/` directory.

They help you trace:

- ingestion progress
- transformation progress
- training progress
- prediction artifact loading

If something fails, check the newest log file first.

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run commands from the project root:

```powershell
cd mlproject
python src\components\data_ingestion.py
```

### `Prediction failed` in the frontend

Most likely causes:

- `artifacts/model.pkl` does not exist
- `artifacts/preprocessor.pkl` does not exist
- preprocessing code changed but artifacts were not retrained

Fix:

```powershell
python src\components\data_ingestion.py
```

### Scikit-learn version mismatch warning

If pickled artifacts were created with a different scikit-learn version, retrain the pipeline:

```powershell
python src\components\data_ingestion.py
```

### Form submits but prediction looks wrong

Check:

- field names in `app.py`
- field names in `templates/home.html`
- feature order in `src/pipeline/predict_pipeline.py`
- expected training schema in `src/components/data_transformation.py`

## Dependencies

Current dependencies from `requirements.txt`:

- pandas
- scikit-learn
- numpy
- flask
- seaborn
- matplotlib
- catboost
- xgboost

## Next Improvements

Possible next steps:

- move inline CSS from templates into `static/`
- add automated unit tests with `pytest`
- add model and preprocessor version metadata
- add a production WSGI server configuration
- deploy the Flask app

## Author

Author:

- Bayanda
