FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone fish-speech
RUN git clone --depth 1 https://github.com/fishaudio/fish-speech.git /app/fish-speech

# Install PyTorch CPU first (separate step with its own index)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies from PyPI
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    pydantic \
    soundfile \
    huggingface_hub

# Install fish-speech — skip pyaudio if it fails (not needed for API use)
RUN pip install --no-cache-dir -e /app/fish-speech || \
    ( pip install --no-cache-dir --no-deps -e /app/fish-speech && \
      grep -v -i pyaudio /app/fish-speech/requirements.txt > /tmp/req_filtered.txt && \
      pip install --no-cache-dir -r /tmp/req_filtered.txt || true )

# Download model checkpoint at build time
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
import pathlib; \
dest = pathlib.Path('/app/checkpoints/fish-speech-1.5'); \
dest.mkdir(parents=True, exist_ok=True); \
snapshot_download(repo_id='openfishai/fish-speech-1.5', local_dir=str(dest), ignore_patterns=['*.bin']); \
print('Checkpoint downloaded.')"

# Copy application
COPY server.py .

# Voice sample directory (populated via /upload-voice endpoint)
RUN mkdir -p /app/voice

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]