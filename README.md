# Student Exam Performance Predictor

This project is an end-to-end machine learning application that predicts a student's `math_score` from the rest of the student profile.

It includes:

- data ingestion from the raw CSV dataset
- preprocessing with a scikit-learn `ColumnTransformer`
- model training and model selection
- saved model and preprocessor artifacts
- a Flask frontend for interactive predictions
- optional containerization with Docker
- optional deployment packaging for Azure

## Project Overview

The model predicts `math_score` using these input features:

- `gender`
- `race_ethnicity`
- `parental_level_of_education`
- `lunch`
- `test_preparation_course`
- `reading_score`
- `writing_score`

The source dataset is stored at:

- `notebook/data/stud.csv`

## Project Structure

```text
mlproject/
|-- app.py
|-- Dockerfile
|-- .dockerignore
|-- README.md
|-- requirements.txt
|-- requirements-extras.txt
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
|   |   |-- __init__.py
|   |   |-- data_ingestion.py
|   |   |-- data_transformation.py
|   |   `-- model_trainer.py
|   `-- pipeline/
|       `-- predict_pipeline.py
`-- templates/
    |-- base.html
    |-- home.html
    |-- index.html
    `-- shutdown.html
```

## Pipeline Overview

### 1. Data ingestion

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

### 2. Data transformation

File:

- `src/components/data_transformation.py`

Responsibilities:

- define numeric and categorical feature groups
- build a scikit-learn preprocessing pipeline
- apply median imputation and scaling to numeric features
- apply most-frequent imputation, one-hot encoding, and scaling to categorical features
- fit preprocessing on the training split only
- save the fitted preprocessor to `artifacts/preprocessor.pkl`

### 3. Model training

File:

- `src/components/model_trainer.py`

Responsibilities:

- receive transformed train and test arrays
- train multiple regression models
- evaluate them with grid search where configured
- choose the best model by `R-squared`
- save the best model to `artifacts/model.pkl`

### 4. Prediction pipeline

File:

- `src/pipeline/predict_pipeline.py`

Responsibilities:

- load `artifacts/preprocessor.pkl`
- load `artifacts/model.pkl`
- transform incoming prediction data
- return the predicted maths score
- rebuild artifacts automatically in local usage when they are missing

### 5. Flask frontend

File:

- `app.py`

Routes:

- `/` -> landing page
- `/predictdata` -> prediction form and prediction results
- `/shutdown` -> local-only app shutdown route

Templates:

- `templates/base.html`
- `templates/index.html`
- `templates/home.html`
- `templates/shutdown.html`

## Local Setup

### 1. Clone the repository

```powershell
git clone <your-repository-url>
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

Optional notebook and experimentation dependencies:

```powershell
pip install -r requirements-extras.txt
```

## Running The Project

### Option 1. Train the full pipeline

This command:

- loads the raw dataset
- creates the train and test splits
- fits the preprocessing object
- trains the model
- writes the saved artifacts

```powershell
python src\components\data_ingestion.py
```

Expected output will look similar to:

```text
Training completed. Test R2 score: 0.8804
```

### Option 2. Run the Flask app

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

### Important artifact note

If `artifacts/model.pkl` and `artifacts/preprocessor.pkl` do not exist, the prediction pipeline can regenerate them automatically in local usage. In the container image, automatic rebuild is disabled by default, so the image should already include the artifacts.

For the best local and container experience, generate artifacts ahead of time with:

```powershell
python src\components\data_ingestion.py
```

## Testing

### 1. End-to-end training smoke test

```powershell
python src\components\data_ingestion.py
```

Confirm these files exist afterward:

- `artifacts/data.csv`
- `artifacts/train.csv`
- `artifacts/test.csv`
- `artifacts/preprocessor.pkl`
- `artifacts/model.pkl`

### 2. Backend prediction smoke test

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

### 3. Flask route smoke test

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

3. Open the predictor page.

4. Submit a sample request such as:

- Gender: `female`
- Race or Ethnicity: `group B`
- Parental Level of Education: `bachelor's degree`
- Lunch Type: `standard`
- Test Preparation Course: `none`
- Reading Score: `72`
- Writing Score: `74`

5. Confirm that:

- a maths score is returned
- the page does not crash
- validation messages appear for bad input

## Docker

The project includes a production-oriented `Dockerfile` for a Python 3.13 container.

The container runtime now:

- runs the app with `gunicorn`
- binds to `0.0.0.0`
- respects `PORT` with a fallback to `5000`
- runs as a non-root user
- exposes `/health` for smoke checks and platform monitoring
- expects the trained artifacts to already exist in the image

### 1. Build the image

```powershell
docker build -t student-score-predictor:latest .
```

### 2. Run the container

```powershell
docker run --rm -p 5000:5000 student-score-predictor:latest
```

Then open:

```text
http://127.0.0.1:5000
```

### 3. Recommended container build flow

Generate artifacts first, then build:

```powershell
python src\components\data_ingestion.py
docker build -t student-score-predictor:latest .
```

This bakes `model.pkl` and `preprocessor.pkl` into the image and avoids runtime retraining inside the container.

### 4. Verify the container health endpoint

```powershell
docker run --rm -p 5000:5000 student-score-predictor:latest
```

Then open:

```text
http://127.0.0.1:5000/health
```

Expected response:

```json
{"status":"ok"}
```

### 5. Optional named container example

```powershell
docker run --name student-score-app -p 5000:5000 student-score-predictor:latest
```

Stop it later with:

```powershell
docker stop student-score-app
```

## Deployment Learnings

The changes that mattered most for reliable cloud deployment were:

- keep the runtime image focused on prediction, not notebook-only experimentation dependencies
- generate `model.pkl` and `preprocessor.pkl` before building the image
- run the app with `gunicorn` instead of the Flask development server
- expose a simple `/health` endpoint and test it locally and in CI
- run the container as a non-root user
- stream logs to stdout so platform log viewers can see startup and request output
- use one deployment driver at a time, rather than mixing multiple automatic deployment mechanisms

## Azure Container Registry

Azure Container Registry, or ACR, is Azure's private Docker image registry. A common flow is:

1. build the Docker image locally
2. tag the image for your ACR login server
3. push the image to ACR
4. configure Azure Web App for Containers to pull that image

### 1. Sign in to Azure

```powershell
az login
```

### 2. Create a resource group if needed

```powershell
az group create --name <resource-group-name> --location <azure-region>
```

### 3. Create an Azure Container Registry

```powershell
az acr create --resource-group <resource-group-name> --name <acr-name> --sku Basic
```

### 4. Log in to ACR from Docker

```powershell
az acr login --name <acr-name>
```

### 5. Build the image locally

```powershell
docker build -t student-score-predictor:latest .
```

### 6. Tag the image for ACR

```powershell
docker tag student-score-predictor:latest <acr-name>.azurecr.io/<repository>:latest
```

### 7. Push the image to ACR

```powershell
docker push <acr-name>.azurecr.io/<repository>:latest
```

### 8. Verify the image in ACR

```powershell
az acr repository list --name <acr-name> --output table
```

To inspect tags:

```powershell
az acr repository show-tags --name <acr-name> --repository <repository> --output table
```

### Notes for Azure container deployments

- If your deployment target pulls from ACR, it will use the image exactly as pushed.
- If you want the model artifacts baked into the image, run training before `docker build`.
- Prefer a standard Linux custom-container Web App for a single-container Flask application.
- The app should be configured to listen on the same port the container exposes. This project defaults to `5000`.
- If the Web App is already pointing at the correct ACR image and tag, deployment can be as simple as pushing a new image and restarting the app.

## GitHub Actions

The deployment workflow is manual-only and is designed for Azure container deployments.

Workflow file:

- `.github/workflows/azure-webapp-deploy.yml`

Trigger:

- `workflow_dispatch`

What the workflow does:

1. checks out the repository
2. installs Python dependencies
3. runs `python src/components/data_ingestion.py` so model artifacts are generated before image build
4. logs into Azure with `azure/login`
5. logs into Azure Container Registry
6. builds the container image
7. smoke-tests the image by calling `/health`
8. pushes the image to ACR
9. restarts the Azure Web App

Why the workflow only restarts the app:

- the Web App is already configured in Azure to use the container image from ACR
- the workflow does not need to reconfigure the Web App on every run
- once the new image is pushed, restarting the Web App is enough to make Azure pull the updated image

Required GitHub configuration:

- secret: `AZURE_CREDENTIALS`
- variable: `AZURE_WEBAPP_NAME`
- variable: `AZURE_RESOURCE_GROUP`

Create `AZURE_CREDENTIALS` with Azure CLI:

```powershell
az login
az ad sp create-for-rbac --role contributor --scopes /subscriptions/<subscription-id>/resourceGroups/<resource-group-name> --json-auth
```

Paste the full JSON output into the GitHub secret named `AZURE_CREDENTIALS`.

That means GitHub will not deploy automatically on every push. To run it:

1. push your code to GitHub
2. open the repository in GitHub
3. go to `Actions`
4. select `Build And Deploy Container To Azure Web App`
5. click `Run workflow`

Recommended one-time Azure checks:

- confirm the Web App still points to the correct ACR image
- confirm the app is able to pull from ACR
- confirm the container port setting is correct for your app
- if you enabled ACR continuous deployment in Azure Deployment Center, disable it so GitHub Actions remains the single deployment driver

## User Behavior And Sessions

This application does not implement authentication or user accounts.

That means:

- there is no login
- there is no logout
- users simply close the browser tab when they are done

The shutdown route and shutdown button are only available for local usage and are hidden in deployed environments.

## Logs

Log files are written to `logs/`.

Use them to inspect:

- ingestion progress
- preprocessing progress
- model training progress
- artifact regeneration
- prediction failures

If something breaks, check the newest log file first.

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run commands from the repository root:

```powershell
cd mlproject
python src\components\data_ingestion.py
```

### `Prediction failed` in the frontend

Most common causes:

- `artifacts/model.pkl` is missing
- `artifacts/preprocessor.pkl` is missing
- preprocessing changed but the saved artifacts were not retrained
- the saved artifacts were created with a different scikit-learn version

Fix:

```powershell
python src\components\data_ingestion.py
```

### Scikit-learn version mismatch warning

If pickled artifacts were created under a different scikit-learn version, retrain them in the active environment:

```powershell
python src\components\data_ingestion.py
```

### Azure deployment works but prediction fails

Check:

- the container image includes `artifacts/model.pkl`
- the container image includes `artifacts/preprocessor.pkl`
- the Web App is still configured to use the correct image and tag
- the running container can start correctly on the configured port

If needed, regenerate artifacts before building the image and redeploy the container.

### Docker container exits immediately

Check container logs:

```powershell
docker logs <container-name>
```

Then verify:

- `gunicorn` is installed
- the image can import `app:app`
- the runtime user can read the application files
- the image includes the saved prediction artifacts

### Health check fails

Run:

```powershell
docker run --rm -p 5000:5000 student-score-predictor:latest
```

Then open:

```text
http://127.0.0.1:5000/health
```

## Dependencies

Current dependencies from `requirements.txt`:

- pandas
- scikit-learn==1.8.0
- numpy
- flask
- gunicorn

Optional extras from `requirements-extras.txt`:

- seaborn
- matplotlib
- catboost
- xgboost

## Entry Points

Training:

```powershell
python src\components\data_ingestion.py
```

Web app:

```powershell
python app.py
```

Docker:

```powershell
docker build -t student-score-predictor:latest .
docker run --rm -p 5000:5000 student-score-predictor:latest
```

## Author

- Bayanda
