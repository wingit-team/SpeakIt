"""Voice models for SpeakIt microservice."""

from .replicator import GPTSoVITSReplicator
from .generator import XTTSGenerator

__all__ = ["GPTSoVITSReplicator", "XTTSGenerator"]

