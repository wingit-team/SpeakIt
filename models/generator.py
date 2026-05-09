"""
XTTS v2 Procedural Voice Generator Module
Handles NPC voice generation with SQLite-based latent tensor persistence.
"""

import sqlite3
import json
import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from TTS.api import TTS


class XTTSGenerator:
    """
    Procedural Voice Generator for NPC voice synthesis using XTTS v2.
    
    Manages NPC voice latents, serializes/deserializes them to/from SQLite,
    and generates speech with consistent voices per NPC.
    """
    
    def __init__(self, db_path: str = "npc_voices.db", device: Optional[str] = None):
        """
        Initialize the XTTS Generator.
        
        Args:
            db_path: Path to SQLite database file for storing NPC voice latents
            device: Device to run inference on ('cuda' or 'cpu').
                    If None, automatically selects CUDA if available, else CPU.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.db_path = db_path
        self.model = None
        self.initialized = False
        self.language = os.getenv("XTTS_LANGUAGE", "en")
        self.sample_rate = 24000
        self._init_db()
        self._init_model()
    
    def _init_db(self):
        """Initialize SQLite database for NPC voice storage."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS npc_voices (
                    npc_id TEXT PRIMARY KEY,
                    voice_latent JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON
                )
            """)
            
            conn.commit()
            conn.close()
            print(f"[XTTS] Database initialized at {self.db_path}")
        except Exception as e:
            print(f"[XTTS] Database initialization error: {e}")
    
    def _init_model(self):
        """
        Initialize the XTTS v2 model.
        """
        try:
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            if hasattr(self.model, "synthesizer") and self.model.synthesizer:
                self.sample_rate = getattr(self.model.synthesizer, "output_sample_rate", 24000)
            self.initialized = True
        except Exception as e:
            print(f"[XTTS] Model initialization warning: {e}")
    
    def _generate_speaker_latent(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a unique speaker latent (conditioning tensor/embedding).
        """
        if not self.initialized or self.model is None:
            raise RuntimeError("XTTS model not initialized")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        duration_seconds = 3.0
        num_samples = int(self.sample_rate * duration_seconds)
        noise_audio = torch.randn(1, num_samples)

        try:
            xtts_model = self.model.synthesizer.tts_model
            gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
                noise_audio,
                self.sample_rate
            )
        except Exception as e:
            raise RuntimeError(f"XTTS conditioning latent generation failed: {e}")

        return {
            "gpt_cond_latent": self._serialize_latent(gpt_cond_latent.squeeze(0).cpu().numpy()),
            "speaker_embedding": self._serialize_latent(speaker_embedding.squeeze(0).cpu().numpy())
        }
    
    def _serialize_latent(self, latent: np.ndarray) -> str:
        """
        Serialize a latent tensor to JSON string.
        """
        return json.dumps({
            "shape": list(latent.shape),
            "data": latent.tolist(),
            "dtype": "float32"
        })
    
    def _deserialize_latent(self, latent_json: str) -> np.ndarray:
        """
        Deserialize a JSON string to latent tensor.
        """
        data = json.loads(latent_json)
        latent = np.array(data["data"], dtype=np.float32)
        return latent.reshape(data["shape"])
    
    def register_npc(self, npc_id: str, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Register a new NPC with a unique voice latent.
        Implicitly creates the NPC if it doesn't exist.
        
        Args:
            npc_id: Unique identifier for the NPC
            seed: Optional seed for reproducible voice generation
        
        Returns:
            Dictionary with NPC registration info
        """
        if not npc_id.strip():
            raise ValueError("npc_id cannot be empty")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if NPC already exists
            cursor.execute("SELECT voice_latent FROM npc_voices WHERE npc_id = ?", (npc_id,))
            existing = cursor.fetchone()
            
            if existing:
                conn.close()
                return {
                    "status": "existing",
                    "npc_id": npc_id,
                    "message": f"NPC {npc_id} already registered"
                }
            
            # Generate new latent
            latent_data = self._generate_speaker_latent(seed=seed)
            latent_json = json.dumps(latent_data)
            
            # Insert into database
            cursor.execute("""
                INSERT INTO npc_voices (npc_id, voice_latent, metadata)
                VALUES (?, ?, ?)
            """, (npc_id, latent_json, json.dumps({"seed": seed, "created_by": "auto"})))
            
            conn.commit()
            conn.close()
            
            print(f"[XTTS] Registered NPC: {npc_id}")
            return {
                "status": "created",
                "npc_id": npc_id,
                "message": f"NPC {npc_id} registered successfully"
            }
        
        except Exception as e:
            raise RuntimeError(f"Failed to register NPC {npc_id}: {str(e)}")
    
    def speak(self, text: str, npc_id: str) -> np.ndarray:
        """
        Generate speech for an NPC.
        Implicitly creates the NPC if it doesn't exist with a random seed.
        
        Args:
            text: Text to synthesize
            npc_id: Unique identifier for the NPC
        
        Returns:
            Synthesized audio as numpy array (Linear16 PCM)
        
        Raises:
            ValueError: If text is empty
            RuntimeError: If synthesis fails
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        if not npc_id.strip():
            raise ValueError("npc_id cannot be empty")
        
        try:
            # Register NPC if it doesn't exist
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT voice_latent FROM npc_voices WHERE npc_id = ?", (npc_id,))
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                # Implicitly create the NPC with a random seed
                self.register_npc(npc_id, seed=None)
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT voice_latent FROM npc_voices WHERE npc_id = ?", (npc_id,))
                result = cursor.fetchone()
            
            # Retrieve and deserialize latent
            latent_json = result[0]
            latent_payload = json.loads(latent_json)
            gpt_cond_latent = torch.tensor(
                self._deserialize_latent(latent_payload["gpt_cond_latent"])
            ).unsqueeze(0).to(self.device)
            speaker_embedding = torch.tensor(
                self._deserialize_latent(latent_payload["speaker_embedding"])
            ).unsqueeze(0).to(self.device)

            conn.close()

            xtts_model = self.model.synthesizer.tts_model
            inference_result = xtts_model.inference(
                text,
                self.language,
                speaker_embedding,
                gpt_cond_latent
            )

            audio = inference_result["wav"] if isinstance(inference_result, dict) else inference_result
            audio = audio.detach().cpu().numpy()

            if audio.ndim > 1:
                audio = audio.squeeze()

            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            return audio_int16
        
        except Exception as e:
            raise RuntimeError(f"Speech synthesis failed for NPC {npc_id}: {str(e)}")
    
    def get_npc_latent(self, npc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the latent for a specific NPC.
        
        Args:
            npc_id: Unique identifier for the NPC
        
        Returns:
            Dictionary with latent data, or None if NPC not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT voice_latent FROM npc_voices WHERE npc_id = ?", (npc_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return json.loads(result[0])
            return None
        
        except Exception as e:
            print(f"[XTTS] Error retrieving NPC latent: {e}")
            return None
    
    def list_npcs(self) -> list:
        """
        List all registered NPCs.
        
        Returns:
            List of NPC IDs
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT npc_id FROM npc_voices ORDER BY created_at")
            npcs = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return npcs
        
        except Exception as e:
            print(f"[XTTS] Error listing NPCs: {e}")
            return []
    
    def get_device(self) -> str:
        """Get the device currently being used."""
        return self.device
    
    def is_initialized(self) -> bool:
        """Check if the model is initialized."""
        return self.initialized

