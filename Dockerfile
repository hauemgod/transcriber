FROM python:3.11-slim

# faster-whisper decodes audio via PyAV (bundled codec libs), so no
# system ffmpeg package is needed here.
WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Which Whisper model to load at startup. Override at `docker run` time
# with -e WHISPER_MODEL=small (bigger = slower + more accurate).
ENV WHISPER_MODEL=base
# Caches downloaded model weights in a mounted volume instead of re-pulling
# them every container start.
ENV XDG_CACHE_HOME=/srv/.cache

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
