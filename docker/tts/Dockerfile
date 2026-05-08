FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/fishaudio/fish-speech.git /app/fish-speech

RUN pip install --no-cache-dir \
    fastapi uvicorn python-multipart pydantic \
    torch torchaudio soundfile huggingface_hub

RUN pip install --no-cache-dir -e /app/fish-speech

RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='openfishai/fish-speech-1.5', \
    local_dir='/app/checkpoints/fish-speech-1.5')"

COPY server.py /app/server.py

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
