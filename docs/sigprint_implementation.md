# SIGPRINT Enhanced Implementation (Ω‑Field Semantics)

This page bridges theory → practice and shows how the monorepo implements the full pipeline: ψ/Λ extraction, 20‑digit codes, gates/loops, journaling, analysis, and closed‑loop control of the RHZ Stylus.

## Components (host/sigprint)
- `system.py`
  - `StructuredSigprintEncoder`: lock‑in demod at 8 Hz (configurable), `ψ` (amplitude) and `Λ` (phase) per channel, structured 20‑digit encoding, loop/gate detection.
  - `VoiceJournal`: speech→text synchronized with SIGPRINT + Ω‑metrics, append to immutable ledger.
  - `StylusInterface`: JSON serial I/O for RHZ Stylus.
- `controller.py`
  - Closed‑loop policy mapping transitions/metrics → stylus commands; JSONL event log.
- `ledger.py` / `ledger_cli.py`
  - Hash‑chained JSONL ledger, optional HMAC, Merkle checkpoints, verification tools.
- `analysis.py` / `ai_cli.py`
  - Gate/loop topics (TF‑IDF), simple sentiment, correlations to Ω‑metrics, suggestions.
- `db.py` / `db_cli.py`
  - SQLite “Consciousness DB”: import ledger, summarize sessions, persist AI insights.
- CLIs
  - `structured_cli.py` (encode 1 s windows), `run_cli.py` (closed loop), plus analysis/ledger/db CLIs.

## Usage Examples

### Encode (1‑second windows)
```
python3 -m host.sigprint.structured_cli data.npy --fs 250 --names Fp1,Fp2,F3,F4,P3,P4,O1,O2
```
- Outputs time‑stamped 20‑digit SIGPRINT codes per window.

### Closed Loop (Order‑3, dry run)
```
python3 -m host.sigprint.run_cli data.npy --fs 250 --names Fp1,Fp2,F3,F4,P3,P4,O1,O2 --dry
```
- Prints `start–end  code  transition  coh  plv` and (when not `--dry`) sends JSON commands to Stylus.

### Voice Journal (Order‑2)
- Captures speech, transcribes to text, computes concurrent SIGPRINT, appends to ledger with Ω‑metrics.
- Requires microphone and `speechrecognition`.

### Immutable Ledger
```
python3 -m host.sigprint.ledger_cli verify journal_ledger.jsonl
python3 -m host.sigprint.ledger_cli checkpoint journal_ledger.jsonl 0 100
```

### AI Analysis → Suggestions
```
python3 -m host.sigprint.ai_cli journal_ledger.jsonl --json
```
- Topics around gates vs loops, correlations (sentiment ↔ coherence/PLV), suggestions.

### Live Voice Journal ↔ AI Analysis
```
python3 -m host.sigprint.live_analysis --db db.sqlite --subject alice --ledger journal_ledger.jsonl --label "live-journal" --window 20
```
- Tails the ledger, inserts entries into SQLite, runs rolling analysis, and prints suggestions; persists insights.

### Consciousness Database (SQLite)
```
python3 -m host.sigprint.db_cli init --db db.sqlite --subject alice
python3 -m host.sigprint.db_cli import-ledger --db db.sqlite --subject alice journal_ledger.jsonl
python3 -m host.sigprint.db_cli summary --db db.sqlite --session 1
python3 -m host.sigprint.db_cli analyze --db db.sqlite --session 1 --json
```

## Mapping to Theory
- Lock‑in (radio analog) → `StructuredSigprintEncoder` recovers `ψ, Λ`.
- Ω‑metrics → coherence (∫ψ² proxy), PLV/R(t), phase entropy.
- 20 digits → phase topology (1–4), amplitude distribution (5–8), coherence/PLV (9–12), context (13–18), checksum (19–20).
- Loops vs Gates → Hamming distance threshold (default 8) across consecutive codes.
- Orders 2–3 → Voice Journal feedback, controller policy adjusting stylus protocol.

## Notes
- Center frequency can be adapted to IAF per subject.
- Use `--amp-gain` to calibrate coherence scale based on signal units (e.g., volts → µV).
- Not for medical diagnosis; research and exploratory feedback only.

## Complete Encoder (Reference)

The structured encoder implemented in `host.sigprint.system.StructuredSigprintEncoder` follows this reference design, mirroring the lock‑in (radio) demodulation to extract ψ (amplitude) and Λ (phase), and a 20‑digit code layout (phase topology, amplitude distribution, coherence/PLV, context, checksum):

```python
import numpy as np
import hashlib
from scipy import signal
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class OmegaState:
    psi: Dict[str, float]      # ψ per channel
    lambda_: Dict[str, float]  # Λ per channel (deg)
    coherence: float           # ∫ψ² proxy → 0..100
    plv: float                 # phase locking 0..1
    entropy: float             # 0..1
    timestamp: float

class SigprintEncoder:
    def __init__(self, channel_names: List[str], sample_rate: float = 250.0, lockin_freq: float = 8.0, gate_threshold: int = 8):
        self.channel_names = channel_names
        self.sample_rate = sample_rate
        self.lockin_freq = lockin_freq
        self.gate_threshold = gate_threshold
        self.epoch_length = int(sample_rate)
        t = np.arange(self.epoch_length) / sample_rate
        self.ref_sin = np.sin(2 * np.pi * lockin_freq * t)
        self.ref_cos = np.cos(2 * np.pi * lockin_freq * t)
        nyq = sample_rate / 2
        cutoff = 2.0 / nyq
        self.lp_b, self.lp_a = signal.butter(4, cutoff, btype='low')
        self.prev_signature = None
        self.gate_count = 0
        self.loop_count = 0

    def process_epoch(self, eeg_epoch: Dict[str, np.ndarray], stylus_context: Dict = None) -> Tuple[str, OmegaState]:
        omega = self._compute_omega_state(eeg_epoch)
        parts = [
            self._encode_phase_topology(omega),     # 4 digits
            self._encode_amplitude_distribution(omega),  # 4 digits
            self._encode_coherence(omega),         # 4 digits
            self._encode_reserved(omega, stylus_context) # 6 digits
        ]
        payload = ''.join(parts)
        checksum = self._compute_checksum(payload)
        sig = payload + checksum
        if self.prev_signature:
            transition = self._detect_transition(sig)
            setattr(omega, 'transition', transition)
        self.prev_signature = sig
        return sig, omega

    def _compute_omega_state(self, eeg_epoch: Dict[str, np.ndarray]) -> OmegaState:
        psi, lambda_ = {}, {}
        for ch in self.channel_names:
            sig = eeg_epoch.get(ch, np.zeros(self.epoch_length))
            if len(sig) != self.epoch_length:
                sig = sig[:self.epoch_length] if len(sig) > self.epoch_length else np.pad(sig, (0, self.epoch_length - len(sig)))
            I = signal.filtfilt(self.lp_b, self.lp_a, sig * self.ref_cos)[-1]
            Q = signal.filtfilt(self.lp_b, self.lp_a, sig * self.ref_sin)[-1]
            amplitude = float(np.hypot(I, Q))
            phase_deg = float(np.degrees(np.arctan2(Q, I)))
            psi[ch] = amplitude
            lambda_[ch] = phase_deg
        coherence = self._compute_global_coherence(psi)
        plv = self._compute_phase_locking_value(lambda_)
        entropy = self._compute_entropy(psi)
        return OmegaState(psi=psi, lambda_=lambda_, coherence=coherence, plv=plv, entropy=entropy, timestamp=float(__import__('time').time()))

    # ... encode_* helpers: phase topology, amplitude distribution, coherence/PLV, reserved, checksum ...
```

The production implementation lives in `host.sigprint.system.StructuredSigprintEncoder` (with optional SciPy fallback, per‑channel normalization, and robust metrics). Use the CLIs above for reproducible runs.

## Voice Journal and Stylus Hooks (Reference)

```python
class VoiceJournal:
    # Synchronizes speech→text with SIGPRINT; appends to immutable JSONL ledger.
    # See host.sigprint.system.VoiceJournal for the working implementation.
    ...

class StylusInterface:
    # Serial JSON I/O with RHZ Stylus; provides context and accepts control commands.
    # See host.sigprint.system.StylusInterface for the working implementation.
    ...
```
