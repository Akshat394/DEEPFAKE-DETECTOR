FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    HOST=0.0.0.0 \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deepfake_detection/requirements.txt /app/deepfake_detection/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/deepfake_detection/requirements.txt && \
    pip install fastapi uvicorn python-multipart transformers

COPY src/backend /app/src/backend
COPY deepfake_detection/src /app/deepfake_detection/src
COPY deepfake_detection/configs /app/deepfake_detection/configs

RUN mkdir -p /app/deepfake_detection/checkpoints /tmp/hf_cache && \
    chmod -R 777 /tmp/hf_cache

EXPOSE 7860

CMD ["python", "src/backend/main.py"]
