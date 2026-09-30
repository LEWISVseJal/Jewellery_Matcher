FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    BLIS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/model_cache/huggingface \
    HF_HUB_CACHE=/app/model_cache/huggingface/hub \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir \
        torch==2.6.0 \
        torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        -r /app/backend/requirements.txt

COPY backend /app/backend
COPY frontend /app/frontend

RUN mkdir -p \
    /app/backend/uploads \
    /app/backend/database \
    /app/backend/catalogue/gold \
    /app/backend/catalogue/prototype \
    /app/model_cache/huggingface

EXPOSE 10000

CMD ["sh", "-c", "gunicorn --workers 1 --threads 2 --timeout 180 --bind 0.0.0.0:${PORT:-10000} backend.app:app"]