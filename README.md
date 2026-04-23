# Student Exam Performance Predictor

This demo project is an end-to-end machine learning application that predicts a student's `math_score` from the rest of the student profile.

It includes:

- data ingestion from the raw CSV dataset
- preprocessing with a scikit-learn `ColumnTransformer`
- model training and model selection
- an exported ONNX inference artifact for serving
- temporary pickle artifacts for parity checks and rollback
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
|-- requirements-container.txt
|-- requirements-training.txt
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
|   |-- model.pkl
|   |-- model.onnx
|   `-- model_metadata.json
|-- logs/
|-- src/
|   |-- exception.py
|   |-- features.py
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
- apply one-hot encoding and scaling to categorical features
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
- export the fitted preprocessing-plus-model pipeline to `artifacts/model.onnx`
- save deployment metadata to `artifacts/model_metadata.json`

### 4. Prediction pipeline

File:

- `src/pipeline/predict_pipeline.py`

Responsibilities:

- load `artifacts/model.onnx` through ONNX Runtime by default
- convert incoming prediction data into the ONNX input schema
- return the predicted maths score
- optionally use `MODEL_RUNTIME=pickle` to load `artifacts/preprocessor.pkl` and `artifacts/model.pkl`
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

For serving the Flask app with the exported ONNX model:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For training and ONNX export:

```powershell
pip install -r requirements-training.txt
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
- exports the ONNX serving artifact
- writes the temporary pickle fallback artifacts

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

`artifacts/model.onnx` is the default serving artifact. `artifacts/model.pkl` and `artifacts/preprocessor.pkl` are still generated as temporary fallback artifacts for parity checks and rollback.

If `artifacts/model.onnx` does not exist, the prediction pipeline can regenerate artifacts automatically in local usage when training dependencies are installed. In the container image, automatic rebuild is disabled by default, so the image must already include `artifacts/model.onnx`.

For the best local and container experience, generate artifacts ahead of time with:

```powershell
python src\components\data_ingestion.py
```

To force the old pickle path during the transition:

```powershell
$env:MODEL_RUNTIME = "pickle"
python app.py
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
- `artifacts/model.onnx`
- `artifacts/model_metadata.json`
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

Build the image directly; the Dockerfile now generates the ONNX artifact in a builder stage:

```powershell
docker build -t student-score-predictor:latest .
```

If you want to validate the training/export path independently, run `python src\components\data_ingestion.py` first. The runtime image stays lean, while the build still bakes `model.onnx` into the image.

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
- generate `model.onnx` during the image build or in the deployment workflow before packaging
- run the app with `gunicorn` instead of the Flask development server
- expose a simple `/health` endpoint and test it locally and in CI
- run the container as a non-root user
- stream logs to stdout so platform log viewers can see startup and request output
- use one deployment driver at a time, rather than mixing multiple automatic deployment mechanisms

## Practical Azure Implementation

In practice, this project would usually run as an online prediction service behind a school, tutoring, analytics, or admin-facing web application. The deployable prediction bundle is the exported ONNX graph plus lightweight metadata:

- `artifacts/model.onnx`
- `artifacts/model_metadata.json`

A common Azure shape is:

- train and export the artifacts from CI, Azure ML, or a scheduled batch job
- store the exported artifacts as versioned build output, in Azure Blob Storage, or in an Azure ML registry
- bake the current artifacts into the container image, as this repository's deployment flow does, or load the active version from durable storage at startup
- push the container image to Azure Container Registry
- deploy the Flask and `gunicorn` service to Azure App Service for Containers, Azure Container Apps, or AKS
- expose `/health` for platform monitoring and `/predictdata` for the current form-based prediction workflow
- optionally add a JSON endpoint such as `/api/predict` if a separate website, mobile app, or backend service needs to call the model directly

The request flow is typically:

1. A user opens the predictor page, a school dashboard, or another product screen that needs a maths score estimate.
2. The frontend or backend collects the required student profile fields: gender, race or ethnicity group, parental education, lunch type, test preparation status, reading score, and writing score.
3. The client sends those fields to the prediction service. In the current app, that happens through a form POST to `/predictdata`.
4. The Flask route validates the request and converts the submitted values into the model's expected feature schema.
5. `PredictPipeline` sends the inputs to ONNX Runtime, which executes the exported preprocessing and model graph and returns the predicted `math_score`.
6. The client renders the prediction on the page or uses it inside a larger workflow, such as academic support triage, reporting, or intervention planning.
7. The serving layer logs prediction failures, startup output, and request output so Azure logs can be used for troubleshooting and operational monitoring.

On Azure, the clean separation is usually:

- website or dashboard: owns page rendering, authentication, and user session context
- application backend: decides when predictions are needed and applies any business rules around who can request or view them
- prediction service: owns validation, ONNX Runtime loading, and score generation
- storage and registry: keep trained artifacts, container images, metrics, and deployment lineage
- monitoring jobs: compare live input distributions and prediction quality against the training reference data when later outcome data is available

That means this repo can be used either as:

- a direct web app where users submit the form and receive a prediction synchronously
- an internal prediction microservice called by another backend before the final page payload is assembled
- a batch or scheduled scorer that generates predictions for records stored elsewhere, then writes the results back to a database or analytics table

The simplest production version is usually the current containerized app on Azure App Service for Containers: GitHub Actions trains the artifacts, builds the image, smoke-tests `/health`, pushes the image to Azure Container Registry, and deploys that exact image to the Web App. For a larger product, keep this service focused on prediction and let the surrounding application handle identity, permissions, student records, and presentation.

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
2. installs training and ONNX export dependencies
3. runs `python src/components/data_ingestion.py` as a preflight check for the training and export path
4. logs into Azure Container Registry with registry credentials
5. builds the container image
6. smoke-tests the image by calling `/health` and posting a sample prediction request
7. pushes the image to ACR
8. deploys the pushed image to Azure Web App by using the app publish profile

Required GitHub configuration:

- secret: `AZURE_WEBAPP_PUBLISH_PROFILE`
- secret: `ACR_USERNAME`
- secret: `ACR_PASSWORD`
- variable: `AZURE_WEBAPP_NAME`

How to get those values from the Azure portal:

1. Open the target Azure Web App.
2. On `Overview`, select `Get publish profile` and download the file.
3. Create the GitHub secret `AZURE_WEBAPP_PUBLISH_PROFILE` using the full contents of that downloaded file.
4. Open the Azure Container Registry.
5. Go to `Access keys`.
6. Enable `Admin user` if it is disabled.
7. Copy the username into GitHub secret `ACR_USERNAME`.
8. Copy one of the passwords into GitHub secret `ACR_PASSWORD`.
9. Create the GitHub variable `AZURE_WEBAPP_NAME` with your Web App name.

Important for Linux Web Apps:

- if `Get publish profile` fails, add the app setting `WEBSITE_WEBDEPLOY_USE_SCM=true` in the Azure portal and try again
- if you do not have access to Microsoft Entra ID, this publish-profile path is usually the simplest way to let GitHub deploy the app

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
- confirm the publish profile you downloaded belongs to the same Web App named in `AZURE_WEBAPP_NAME`

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

- `artifacts/model.onnx` is missing
- `onnxruntime` is not installed in the serving environment
- preprocessing changed but the saved artifacts were not retrained
- `MODEL_RUNTIME=pickle` is set but `artifacts/model.pkl` or `artifacts/preprocessor.pkl` is missing

Fix:

```powershell
pip install -r requirements-training.txt
python src\components\data_ingestion.py
```

### Scikit-learn version mismatch warning

This should only affect the temporary pickle fallback path. If pickled artifacts were created under a different scikit-learn version, retrain them in the active environment:

```powershell
python src\components\data_ingestion.py
```

### Azure deployment works but prediction fails

Check:

- the container image includes `artifacts/model.onnx`
- the container installed `onnxruntime`
- the Web App is still configured to use the correct image and tag
- the running container can start correctly on the configured port

If needed, regenerate artifacts before building the image and redeploy the container.
If you are building from a clean checkout, rerun the Docker build or regenerate artifacts with `python src\components\data_ingestion.py` before rebuilding.

### Docker container exits immediately

Check container logs:

```powershell
docker logs <container-name>
```

Then verify:

- `gunicorn` is installed
- the image can import `app:app`
- the runtime user can read the application files
- the image includes `artifacts/model.onnx`

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

- numpy
- flask
- onnxruntime

Container-only dependencies from `requirements-container.txt`:

- gunicorn

Training and ONNX export dependencies from `requirements-training.txt`:

- numpy
- pandas
- scikit-learn==1.8.0
- onnxruntime
- onnx
- skl2onnx
- onnxmltools

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
