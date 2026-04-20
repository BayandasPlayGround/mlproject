
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

COPY --chown=appuser:appuser . .

RUN test -f artifacts/model.onnx \
    && chown -R appuser:appuser /app

EXPOSE 5000

USER appuser

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 120 --access-logfile - --error-logfile - app:app"]
