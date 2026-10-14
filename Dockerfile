FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface TORCH_HOME=/app/.cache/torch
WORKDIR /app

# WITH_ML=0 — лёгкий образ без моделей: правила и голос браузера.
ARG WITH_ML=1

COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt && \
    if [ "$WITH_ML" = "1" ]; then \
      apt-get update && apt-get install -y --no-install-recommends build-essential cmake && \
      pip install -r requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu && \
      apt-get purge -y build-essential cmake && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*; \
    fi

COPY app ./app
COPY static ./static
COPY data/voice_assistant_events.csv ./data/voice_assistant_events.csv

RUN useradd --create-home appuser && mkdir -p /app/.cache && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
