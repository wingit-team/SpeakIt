"""
GPT-SoVITS Voice Replicator Module
Handles zero-shot voice cloning using GPT-SoVITS architecture.
"""

import sys
import os
import torch
import torchaudio
import numpy as np
import subprocess
import tempfile
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple


class GPTSoVITSReplicator:
    """
    Voice Replicator for zero-shot voice cloning using GPT-SoVITS.
    
    Accepts target text, reference audio file, and reference text transcript to
    synthesize audio with the cloned voice.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the Voice Replicator.

        Args:
            device: Device to run inference on ('cuda' or 'cpu').
                    If None, automatically selects CUDA if available, else CPU.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.initialized = False
        self.gpt_sovits_root = Path(os.getenv("GPT_SOVITS_ROOT", "GPT-SoVITS"))
        self.gpt_path = os.getenv("GPT_SOVITS_GPT_PATH")
        self.sovits_path = os.getenv("GPT_SOVITS_SOVITS_PATH")
        self.ref_lang = os.getenv("GPT_SOVITS_REF_LANG", "en")
        self.tgt_lang = os.getenv("GPT_SOVITS_TGT_LANG", "en")
        self._init_model()
    
    def _init_model(self):
        """
        Initialize the GPT-SoVITS model.
        Uses the GPT-SoVITS CLI entrypoint when model paths are configured.
        """
        try:
            if self.gpt_sovits_root.exists():
                self.initialized = True
                return
            raise FileNotFoundError("GPT-SoVITS submodule not found.")
        except Exception as e:
            print(f"[GPTSoVITS] Model initialization warning: {e}")
            # Model will be loaded on first use or fail gracefully
    
    def _load_audio(self, audio_path: str, target_sr: int = 24000) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and resample to target sample rate.
        
        Args:
            audio_path: Path to audio file (.wav, .mp3, etc.)
            target_sr: Target sample rate (default 24000 Hz for SoVITS)
        
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        audio, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)
            sr = target_sr
        
        return audio, sr
    
    def _run_cli_inference(
        self,
        target_text: str,
        reference_audio_path: str,
        reference_text: str
    ) -> np.ndarray:
        if not self.gpt_path or not self.sovits_path:
            raise RuntimeError(
                "GPT-SoVITS model paths are not configured. Set GPT_SOVITS_GPT_PATH and "
                "GPT_SOVITS_SOVITS_PATH environment variables."
            )

        if not Path(self.gpt_path).exists() or not Path(self.sovits_path).exists():
            raise FileNotFoundError("GPT-SoVITS model files not found at configured paths.")

        cli_path = self.gpt_sovits_root / "inference_cli.py"
        if not cli_path.exists():
            raise FileNotFoundError(f"GPT-SoVITS CLI not found at {cli_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "synth.wav"

            cmd = [
                sys.executable,
                str(cli_path),
                "--gpt_path",
                self.gpt_path,
                "--sovits_path",
                self.sovits_path,
                "--ref_audio_path",
                reference_audio_path,
                "--ref_text",
                reference_text,
                "--text",
                target_text,
                "--text_lang",
                self.tgt_lang,
                "--prompt_lang",
                self.ref_lang,
                "--output_path",
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"GPT-SoVITS inference failed: {result.stderr.strip() or result.stdout.strip()}"
                )

            audio, _ = sf.read(output_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            return np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    def synthesize(
        self,
        target_text: str,
        reference_audio_path: str,
        reference_text: str,
        temperature: float = 0.3,
        top_p: float = 0.7,
        top_k: int = 20
    ) -> np.ndarray:
        """
        Synthesize audio by cloning the voice from reference audio.
        
        Args:
            target_text: Text to synthesize with the cloned voice
            reference_audio_path: Path to reference audio file for voice cloning
            reference_text: Transcript of the reference audio
            temperature: Sampling temperature for semantic tokens (0.0-1.0)
            top_p: Nucleus sampling parameter for semantic tokens
            top_k: Top-k sampling parameter for semantic tokens
        
        Returns:
            Synthesized audio as numpy array (Linear16 PCM)
        
        Raises:
            FileNotFoundError: If reference audio file not found
            ValueError: If inputs are invalid
        """
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"Reference audio file not found: {reference_audio_path}")
        
        if not target_text.strip():
            raise ValueError("Target text cannot be empty")
        
        if not reference_text.strip():
            raise ValueError("Reference text cannot be empty")
        
        try:
            if self.gpt_path and self.sovits_path:
                return self._run_cli_inference(target_text, reference_audio_path, reference_text)

            raise RuntimeError(
                "GPT-SoVITS inference requires model paths. Configure GPT_SOVITS_GPT_PATH and "
                "GPT_SOVITS_SOVITS_PATH environment variables."
            )

        except Exception as e:
            raise RuntimeError(f"Voice synthesis failed: {str(e)}")
    
    def get_device(self) -> str:
        """Get the device currently being used."""
        return self.device
    
    def is_initialized(self) -> bool:
        """Check if the model is initialized."""
        return self.initialized

