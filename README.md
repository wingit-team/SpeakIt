# SpeakIt

Headless Text-to-Speech microservice with two modes:
- Voice cloning via GPT-SoVITS submodule.
- Procedural NPC voices via XTTS v2 with SQLite-backed speaker latents.

## Backend (FastAPI)

### Requirements
- Python 3.10+
- FFmpeg (recommended for audio decoding)

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run
```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Endpoints
- `POST /clone_voice` (multipart/form-data)
  - `target_text`, `reference_text`, `reference_audio`, `temperature`, `top_p`, `top_k`
- `POST /generate_npc_speech` (JSON or form)
  - `text`, `npc_id` (optional)
  - Returns `audio/wav` with `X-NPC-ID` response header
- `GET /npc_list`

### Example Requests
```bash
curl -X POST "http://localhost:7860/clone_voice" \
  -F "target_text=Hello there" \
  -F "reference_text=Hello world" \
  -F "reference_audio=@/path/to/ref.wav" \
  --output cloned_voice.wav
```

```bash
curl -X POST "http://localhost:7860/generate_npc_speech" \
  -H "Content-Type: application/json" \
  -d '{"text":"Greetings traveler","npc_id":"npc_guard_01"}' \
  --output npc_speech.wav
```

## Frontend (React)

The UI lives in `frontend/` and targets the backend via `VITE_BACKEND_URL`.

```bash
cd frontend
npm install
npm run dev
```

Set the backend URL for production builds:
```bash
VITE_BACKEND_URL=https://your-hf-space-url
```

## Configuration

Set these environment variables when running the backend:

- `GPT_SOVITS_ROOT`: Path to the GPT-SoVITS submodule (default `GPT-SoVITS`).
- `GPT_SOVITS_GPT_PATH`: Path to the GPT model checkpoint.
- `GPT_SOVITS_SOVITS_PATH`: Path to the SoVITS model checkpoint.
- `GPT_SOVITS_REF_LANG`: Reference text language (default `en`).
- `GPT_SOVITS_TGT_LANG`: Target text language (default `en`).
- `XTTS_LANGUAGE`: Language code for XTTS (default `en`).

Example:
```bash
export GPT_SOVITS_GPT_PATH=/models/gpt.ckpt
export GPT_SOVITS_SOVITS_PATH=/models/sovits.pth
export GPT_SOVITS_REF_LANG=en
export GPT_SOVITS_TGT_LANG=en
export XTTS_LANGUAGE=en
```

## Notes
- GPT-SoVITS inference uses the submodule CLI and requires model checkpoints via env vars.
- XTTS v2 uses procedural latents generated at runtime and stored in SQLite.
- SQLite database `npc_voices.db` is created at runtime (ignored by git).
