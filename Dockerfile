
FROM python:3.13-slim-bookworm AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    MODEL_RUNTIME=onnx \
    ALLOW_ARTIFACT_REBUILD=1

COPY requirements.txt requirements-training.txt ./

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt -r requirements-training.txt

COPY . .

RUN python src/components/data_ingestion.py \
    && test -f artifacts/model.onnx \
    && test -f artifacts/model_metadata.json \
    && test -f artifacts/model.pkl \
    && test -f artifacts/preprocessor.pkl

FROM python:3.13-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    MODEL_RUNTIME=onnx \
    ALLOW_ARTIFACT_REBUILD=0

RUN useradd --create-home appuser

COPY requirements.txt requirements-container.txt ./

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-container.txt

COPY --from=builder --chown=appuser:appuser /app/app.py /app/app.py
COPY --from=builder --chown=appuser:appuser /app/src /app/src
COPY --from=builder --chown=appuser:appuser /app/templates /app/templates
COPY --from=builder --chown=appuser:appuser /app/artifacts /app/artifacts

EXPOSE 5000

USER appuser

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 120 --access-logfile - --error-logfile - app:app"]
