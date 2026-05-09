# SpeakIt

Headless Text-to-Speech microservice with two modes:
- Voice cloning via GPT-SoVITS (submodule placeholder wiring).
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

## Notes
- GPT-SoVITS and XTTS logic is wired as placeholders and intended to be replaced with real model calls.
- SQLite database `npc_voices.db` is created at runtime (ignored by git).

