import io
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)

CHECKPOINT = "/app/checkpoints/fish-speech-1.5"
VOICE_PATH = Path("/app/voice/voice_sample.wav")

app = FastAPI(title="Lecteur FR TTS")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        log.info("Loading TTSInferenceEngine…")
        sys.path.insert(0, "/app/fish-speech")
        from fish_speech.inference_engine import TTSInferenceEngine
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _engine = TTSInferenceEngine(
            checkpoint_path=CHECKPOINT,
            device=device,
            precision="float16" if device == "cuda" else "float32",
            compile=False,
        )
        log.info("Engine ready on %s", device)
    return _engine


class SpeakRequest(BaseModel):
    text: str
    speed: float = 1.0


@app.get("/health")
def health():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "ok",
        "device": device,
        "voice_loaded": VOICE_PATH.exists(),
        "model_ready": _engine is not None,
    }


@app.post("/speak")
def speak(req: SpeakRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is empty")

    engine = get_engine()
    ref_audio = str(VOICE_PATH) if VOICE_PATH.exists() else None
    ref_text = None

    log.info("Synthesising %d chars, speed=%.2f, ref=%s", len(req.text), req.speed, ref_audio)

    try:
        result = engine.inference(
            text=req.text,
            reference_audio=ref_audio,
            reference_text=ref_text,
            max_new_tokens=0,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            speed=req.speed,
        )
        audio_data, sample_rate = result
    except Exception as exc:
        log.exception("Inference failed")
        raise HTTPException(500, str(exc))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        sf.write(tmp_wav.name, audio_data, sample_rate)
        wav_path = tmp_wav.name

    try:
        mp3_buf = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-q:a", "4", "-f", "mp3", "pipe:1"],
            capture_output=True, check=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise HTTPException(500, "ffmpeg error: " + exc.stderr.decode())
    finally:
        os.unlink(wav_path)

    return StreamingResponse(io.BytesIO(mp3_buf), media_type="audio/mpeg")


@app.post("/upload-voice")
async def upload_voice(file: UploadFile = File(...)):
    if file.content_type not in ("audio/wav", "audio/wave", "audio/mpeg", "audio/mp3", "audio/x-wav"):
        raise HTTPException(400, "Upload a WAV or MP3 file")

    VOICE_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw = await file.read()
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
        tmp.write(raw)
        src_path = tmp.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", str(VOICE_PATH)],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise HTTPException(500, "ffmpeg conversion failed: " + exc.stderr.decode())
    finally:
        os.unlink(src_path)

    log.info("Voice sample saved to %s", VOICE_PATH)
    return {"status": "ok", "path": str(VOICE_PATH)}
