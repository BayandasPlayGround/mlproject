"""Flask entrypoint for the student exam performance predictor."""
import os
import threading

from flask import Flask, abort, render_template, request

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", None}

SELECT_FIELDS = [
    {
        "name": "gender",
        "label": "Gender",
        "placeholder": "Select gender",
        "options": [
            {"value": "male", "label": "Male"},
            {"value": "female", "label": "Female"},
        ],
    },
    {
        "name": "race_ethnicity",
        "label": "Race or Ethnicity",
        "placeholder": "Select ethnicity",
        "options": [
            {"value": "group A", "label": "Group A"},
            {"value": "group B", "label": "Group B"},
            {"value": "group C", "label": "Group C"},
            {"value": "group D", "label": "Group D"},
            {"value": "group E", "label": "Group E"},
        ],
    },
    {
        "name": "parental_level_of_education",
        "label": "Parental Level of Education",
        "placeholder": "Select parent education",
        "options": [
            {"value": "associate's degree", "label": "Associate's degree"},
            {"value": "bachelor's degree", "label": "Bachelor's degree"},
            {"value": "high school", "label": "High school"},
            {"value": "master's degree", "label": "Master's degree"},
            {"value": "some college", "label": "Some college"},
            {"value": "some high school", "label": "Some high school"},
        ],
        "full_width": True,
    },
    {
        "name": "lunch",
        "label": "Lunch Type",
        "placeholder": "Select lunch type",
        "options": [
            {"value": "free/reduced", "label": "Free/reduced"},
            {"value": "standard", "label": "Standard"},
        ],
    },
    {
        "name": "test_preparation_course",
        "label": "Test Preparation Course",
        "placeholder": "Select course status",
        "options": [
            {"value": "none", "label": "None"},
            {"value": "completed", "label": "Completed"},
        ],
    },
]

SCORE_FIELDS = [
    {
        "name": "reading_score",
        "label": "Reading Score out of 100",
        "placeholder": "Enter reading score",
        "min": 0,
        "max": 100,
    },
    {
        "name": "writing_score",
        "label": "Writing Score out of 100",
        "placeholder": "Enter writing score",
        "min": 0,
        "max": 100,
    },
]


def render_prediction_page(
    *,
    results=None,
    error_message=None,
    field_errors=None,
    form_data=None,
    status_code=200,
):
    response = render_template(
        "home.html",
        results=results,
        error_message=error_message,
        field_errors=field_errors or {},
        form_data=form_data or {},
        select_fields=SELECT_FIELDS,
        score_fields=SCORE_FIELDS,
    )
    return response, status_code


def validate_form_data(form_data):
    cleaned_data = {}
    field_errors = {}

    for field in SELECT_FIELDS:
        value = form_data.get(field["name"], "").strip()
        allowed_values = {option["value"] for option in field["options"]}

        if not value:
            field_errors[field["name"]] = f"{field['label']} is required."
        elif value not in allowed_values:
            field_errors[field["name"]] = f"Select a valid value for {field['label'].lower()}."
        else:
            cleaned_data[field["name"]] = value

    for field in SCORE_FIELDS:
        raw_value = form_data.get(field["name"], "").strip()

        if not raw_value:
            field_errors[field["name"]] = f"{field['label']} is required."
            continue

        try:
            value = int(raw_value)
        except ValueError:
            field_errors[field["name"]] = f"{field['label']} must be a whole number."
            continue

        if not field["min"] <= value <= field["max"]:
            field_errors[field["name"]] = (
                f"{field['label']} must be between {field['min']} and {field['max']}."
            )
            continue

        cleaned_data[field["name"]] = value

    return cleaned_data, field_errors


def schedule_shutdown():
    shutdown_func = request.environ.get("werkzeug.server.shutdown")

    if shutdown_func is not None:
        threading.Timer(0.5, shutdown_func).start()
    else:
        threading.Timer(0.5, lambda: os._exit(0)).start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_prediction_page()

    form_data = request.form.to_dict()
    cleaned_data, field_errors = validate_form_data(form_data)

    if field_errors:
        return render_prediction_page(
            results=None,
            error_message="Please correct the highlighted fields and try again.",
            field_errors=field_errors,
            form_data=form_data,
            status_code=400,
        )

    try:
        custom_data = CustomData(
            gender=cleaned_data["gender"],
            race_ethnicity=cleaned_data["race_ethnicity"],
            parental_level_of_education=cleaned_data["parental_level_of_education"],
            lunch=cleaned_data["lunch"],
            test_preparation_course=cleaned_data["test_preparation_course"],
            reading_score=cleaned_data["reading_score"],
            writing_score=cleaned_data["writing_score"],
        )

        prediction_df = custom_data.get_data_as_data_frame()
        prediction = PredictPipeline().predict(prediction_df)[0]

        return render_prediction_page(
            results=round(float(prediction), 2),
            error_message=None,
            field_errors={},
            form_data=form_data,
        )
    except Exception as exc:
        return render_prediction_page(
            results=None,
            error_message=f"Prediction failed: {exc}",
            field_errors={},
            form_data=form_data,
            status_code=500,
        )


@app.route("/shutdown", methods=["POST"])
def shutdown():
    if request.remote_addr not in LOCAL_HOSTS:
        abort(403)

    if not app.testing:
        schedule_shutdown()

    return render_template("shutdown.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")

