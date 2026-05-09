## Plan: SpeakIt FastAPI TTS Microservice

Create a modular, engine-agnostic TTS microservice handling zero-shot cloning via GPT-SoVITS and procedural NPC voices via XTTS v2 with SQLite persistence, served via FastAPI.

### Steps
1. Create `requirements.txt` adding `fastapi`, `uvicorn`, `TTS`, `torch`, `torchaudio`, and `sqlite3` dependencies, then `git add/commit/push`.
2. Add GPT-SoVITS repository as a git submodule.
3. Create `models/replicator.py` defining `GPTSoVITSReplicator` to handle target text, reference audio, and transcript inputs for zero-shot synthesis, using the submodule, then `commit/push`.
4. Create `models/generator.py` defining `XTTSGenerator` to integrate SQLite, serialize/deserialize NPC latent tensors in JSON, and implement `speak(text, npc_id)`. If `npc_id` (seed) is missing, implicitly create a new NPC. Then `commit/push`.
5. Update `main.py` to instantiate the generator and replicator classes, adding `/clone_voice` and `/generate_npc_speech` endpoints returning LINEAR16 PCM `.wav`, then `commit/push`.
6. Prepare backend for HuggingFace Spaces deployment (e.g., configuring host/port for HF Spaces).
7. Create a separate Frontend project (for Cloudflare Pages) to interact with the backend API.

### Further Considerations
1. Do we need an endpoint to explicitly register/create new NPCs, or should `/generate_npc_speech` implicitly create one if `npc_id` is missing?
