import os
import tempfile
import threading

import ctranslate2
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles

# Models the frontend can pick between, smallest/fastest first.
# "large" is multiple GB and slow on CPU; included for completeness, not the default.
MODEL_CHOICES = ["tiny", "base", "small", "medium", "turbo", "large"]
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "base")

# No NVIDIA GPU here (AMD card, no CUDA support in CTranslate2), so this
# will be 0 and we fall back to CPU. Kept generic in case this ever runs
# on NVIDIA hardware.
_gpu_count = ctranslate2.get_cuda_device_count()
DEVICE = "cuda" if _gpu_count > 0 else "cpu"
# int8 quantization is the fast, low-memory choice for CPU; float16 for GPU.
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
DEVICE_LABEL = f"GPU x{_gpu_count}" if DEVICE == "cuda" else "CPU"

# Loaded lazily per model name and cached, so startup only pays for the
# default model instead of downloading every size up front.
_models: dict[str, WhisperModel] = {}
_models_lock = threading.Lock()


def get_model(name: str) -> WhisperModel:
    if name not in MODEL_CHOICES:
        raise HTTPException(status_code=400, detail=f"Unknown model '{name}'")
    with _models_lock:
        if name not in _models:
            print(f"[model] loading '{name}' on {DEVICE} ({COMPUTE_TYPE})")
            _models[name] = WhisperModel(name, device=DEVICE, compute_type=COMPUTE_TYPE)
            print(f"[model] '{name}' ready")
        return _models[name]


print(f"[startup] device: {DEVICE} ({DEVICE_LABEL})")
get_model(DEFAULT_MODEL)  # warm the default so the first request isn't slow

app = FastAPI()


@app.get("/api/health")
def health():
    return {
        "device": DEVICE,
        "device_label": DEVICE_LABEL,
        "models": MODEL_CHOICES,
        "default_model": DEFAULT_MODEL,
        "loaded_models": list(_models.keys()),
    }


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), model: str = Form(DEFAULT_MODEL)):
    whisper_model = get_model(model)

    suffix = os.path.splitext(file.filename or "")[1] or ".m4a"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments, _info = whisper_model.transcribe(tmp_path)
        text = "".join(segment.text for segment in segments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        os.remove(tmp_path)

    return {"text": text, "model": model, "device": DEVICE}


# Serve the frontend last so /api/* routes above take priority.
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
