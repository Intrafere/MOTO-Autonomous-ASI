FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MOTO_GENERIC_MODE=true \
    HOST=0.0.0.0 \
    PORT=8000 \
    MOTO_DATA_ROOT=/app/backend/data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-generic.txt ./

RUN pip install --upgrade pip \
    && pip install -r requirements-generic.txt

COPY package.json moto-update-manifest.json ./
COPY backend ./backend
COPY docker ./docker

RUN sed -i 's/\r$//' /app/docker/entrypoint.sh \
    && chmod +x /app/docker/entrypoint.sh \
    && mkdir -p /app/backend/data \
    && python - <<'PY'
from fastembed import TextEmbedding

from backend.shared.fastembed_provider import FASTEMBED_MODEL_NAME

TextEmbedding(model_name=FASTEMBED_MODEL_NAME)
PY

EXPOSE 8000

ENTRYPOINT ["/app/docker/entrypoint.sh"]
