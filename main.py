from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import os
import numpy as np
import soundfile as sf
from models import GPTSoVITSReplicator, XTTSGenerator
from pydantic import BaseModel
import uuid
from typing import Optional

# Initialize FastAPI app with CORS enabled
app = FastAPI(
    title="SpeakIt TTS Microservice",
    description="Headless TTS engine for voice cloning and NPC voice synthesis",
    version="1.0.0"
)

# Enable CORS for Cloudflare Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
replicator = GPTSoVITSReplicator()
generator = XTTSGenerator(db_path="npc_voices.db")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "SpeakIt TTS Microservice",
        "status": "running",
        "replicator_ready": replicator.is_initialized(),
        "generator_ready": generator.is_initialized()
    }


@app.post("/clone_voice")
async def clone_voice(
    target_text: str = Form(...),
    reference_text: str = Form(...),
    reference_audio: UploadFile = File(...),
    temperature: float = Form(0.3),
    top_p: float = Form(0.7),
    top_k: int = Form(20)
):
    """
    Clone a voice from reference audio and synthesize target text.
    
    Args:
        target_text: Text to synthesize with cloned voice
        reference_text: Transcript of the reference audio
        reference_audio: Audio file to use as voice reference (.wav, .mp3, etc.)
        temperature: Sampling temperature for semantic tokens (0.0-1.0)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
    
    Returns:
        audio/wav stream (LINEAR16 PCM)
    """
    try:
        # Save uploaded file temporarily
        temp_audio_path = f"/tmp/ref_{reference_audio.filename}"
        os.makedirs("/tmp", exist_ok=True)
        
        with open(temp_audio_path, "wb") as f:
            f.write(await reference_audio.read())
        
        # Synthesize audio using replicator
        audio_array = replicator.synthesize(
            target_text=target_text,
            reference_audio_path=temp_audio_path,
            reference_text=reference_text,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        
        # Clean up temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        # Convert to WAV bytes
        sample_rate = 24000
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_array, sample_rate, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        
        return StreamingResponse(
            iter([wav_buffer.getvalue()]),
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=cloned_voice.wav"}
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice synthesis failed: {str(e)}")


class NPCSpeechRequest(BaseModel):
    text: str
    npc_id: Optional[str] = None


@app.post("/generate_npc_speech")
async def generate_npc_speech(
    payload: Optional[NPCSpeechRequest] = Body(None),
    text: Optional[str] = Form(None),
    npc_id: Optional[str] = Form(None)
):
    """
    Generate speech for an NPC with a consistent synthesized voice.
    If NPC doesn't exist, it is implicitly created with a random seed.

    Args:
        text: Text to synthesize
        npc_id: Unique identifier for the NPC (seed)

    Returns:
        audio/wav stream (LINEAR16 PCM)
    """
    try:
        if payload is not None:
            text = payload.text
            npc_id = payload.npc_id

        if text is None or not text.strip():
            raise HTTPException(status_code=422, detail="Text cannot be empty")

        if npc_id is None or not npc_id.strip():
            npc_id = f"npc_{uuid.uuid4().hex[:8]}"

        # Generate audio using generator
        audio_array = generator.speak(text, npc_id)

        # Convert to WAV bytes
        sample_rate = 24000
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_array, sample_rate, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)

        return StreamingResponse(
            iter([wav_buffer.getvalue()]),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=npc_speech.wav",
                "X-NPC-ID": npc_id
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")


@app.get("/npc_list")
async def list_npcs():
    """List all registered NPCs."""
    try:
        npcs = generator.list_npcs()
        return {"npcs": npcs, "count": len(npcs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing NPCs: {str(e)}")
