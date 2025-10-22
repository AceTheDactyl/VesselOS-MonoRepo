"""SIGPRINT package: Ω-field EEG encoders and utilities.

Exports:
  - OmegaState: dataclass of per-channel ψ, Λ and global metrics
  - SigprintEncoder: numeric-feature 20-digit encoder (Hilbert-like)
  - StructuredSigprintEncoder: structured 20-digit encoder (phase/amp/coherence)
  - VoiceJournal: voice journaling with SIGPRINT synchronization (optional deps)
  - StylusInterface: serial interface for RHZ Stylus (optional deps)
"""

from .encoder import OmegaState, SigprintEncoder  # noqa: F401
from .system import StructuredSigprintEncoder, VoiceJournal, StylusInterface  # noqa: F401
