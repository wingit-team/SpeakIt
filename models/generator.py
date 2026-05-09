"""
XTTS v2 Procedural Voice Generator Module
Handles NPC voice generation with SQLite-based latent tensor persistence.
"""

import sqlite3
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


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
        For now, this is a placeholder for the actual model loading logic.
        In production, this would load from the TTS library.
        """
        try:
            # Placeholder for XTTS v2 model initialization
            # In production, this would use: from TTS.models import load_model
            # self.model = load_model("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            print(f"[XTTS] Initializing model on device: {self.device}")
            self.initialized = True
        except Exception as e:
            print(f"[XTTS] Model initialization warning: {e}")
    
    def _generate_speaker_latent(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a unique speaker latent (conditioning tensor/embedding).
        
        Args:
            seed: Optional seed for reproducibility. If None, random latent is generated.
        
        Returns:
            Dictionary containing serialized latent tensor
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random latent vector (placeholder)
        # In production, this would be extracted from reference audio or procedurally generated
        latent = np.random.randn(512, 768).astype(np.float32)  # Placeholder dimensions
        
        return {
            "shape": list(latent.shape),
            "data": latent.tolist(),
            "dtype": "float32"
        }
    
    def _serialize_latent(self, latent: np.ndarray) -> str:
        """
        Serialize a latent tensor to JSON string.
        
        Args:
            latent: Numpy array representing the latent tensor
        
        Returns:
            JSON string representation of the latent
        """
        return json.dumps({
            "shape": list(latent.shape),
            "data": latent.tolist(),
            "dtype": "float32"
        })
    
    def _deserialize_latent(self, latent_json: str) -> np.ndarray:
        """
        Deserialize a JSON string to latent tensor.
        
        Args:
            latent_json: JSON string representation of the latent
        
        Returns:
            Numpy array representing the latent tensor
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
            latent = self._deserialize_latent(latent_json)
            
            conn.close()
            
            # Placeholder: Generate audio using XTTS v2 with the latent
            # In production, this would use: self.model.synthesize(text, latent)
            print(f"[XTTS] Synthesizing for NPC {npc_id}: {len(text)} chars")
            
            # Generate placeholder audio (5 seconds at 24kHz)
            sample_rate = 24000
            duration = (len(text) / 100) * 1.5  # Rough estimate: ~100 chars per 1.5 seconds
            num_samples = int(sample_rate * duration)
            
            synthesized_audio = np.random.randn(num_samples).astype(np.float32) * 0.1
            
            # Normalize audio to [-1, 1] range
            max_val = np.max(np.abs(synthesized_audio))
            if max_val > 0:
                synthesized_audio = synthesized_audio / max_val
            
            # Convert to Linear16 PCM (int16)
            audio_int16 = np.clip(synthesized_audio * 32767, -32768, 32767).astype(np.int16)
            
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

