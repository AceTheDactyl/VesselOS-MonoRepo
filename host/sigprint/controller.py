from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .system import StructuredSigprintEncoder, StylusInterface


@dataclass
class StylusPolicy:
    """Rule-based mapping from Ω-state to Stylus commands.

    Parameters influence when to send "GATE" vs "LOOP" commands and how to
    compute stage and intensity. This is intentionally simple and transparent.
    """

    gate_threshold: int = 8            # Hamming distance digits
    min_stage: int = 0
    max_stage: int = 6
    base_intensity: int = 50           # 0-255
    max_intensity: int = 200           # 0-255
    coherence_gain: float = 1.0        # scales intensity with coherence
    plv_to_freq_hz: Tuple[float, float] = (5.0, 20.0)  # min,max pulse freq

    def decide(self, code: str, omega) -> Optional[Dict]:
        """Return a Stylus command dict or None.

        Command schema (example):
          {
            "type": "set",
            "stage": 3,
            "intensity": 120,     # 0-255
            "pulse_hz": 12.0,
            "meta": {"transition": "GATE", "coherence": 78.2, ...}
          }
        """
        trans = getattr(omega, "transition", "INIT")
        # Stage progression: bump on GATE, decay on LOOP
        stage = int(np.clip(getattr(omega, "stage", 0) + (1 if trans == "GATE" else 0), self.min_stage, self.max_stage))
        setattr(omega, "stage", stage)

        # Intensity maps to coherence (0..100) → 0..max range
        coh = float(np.clip(omega.coherence, 0.0, 100.0))
        intensity = int(np.clip(self.base_intensity + self.coherence_gain * coh, 0, self.max_intensity))

        # Pulse frequency maps to PLV (0..1)
        plv = float(np.clip(omega.plv, 0.0, 1.0))
        fmin, fmax = self.plv_to_freq_hz
        pulse_hz = float(fmin + (fmax - fmin) * plv)

        return {
            "type": "set",
            "stage": stage,
            "intensity": intensity,
            "pulse_hz": round(pulse_hz, 2),
            "meta": {
                "transition": trans,
                "coherence": round(coh, 2),
                "plv": round(plv, 3),
                "entropy": round(omega.entropy, 3),
                "code": code,
            },
        }


def run_closed_loop(
    data: np.ndarray,
    ch_names: List[str],
    fs: float,
    center_hz: float = 8.0,
    amp_gain: float = 1.0,
    stylus: Optional[StylusInterface] = None,
    policy: Optional[StylusPolicy] = None,
    jsonl_path: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """Run a simple Order‑3 loop on array data (channels x samples).

    - Computes 1 s SIGPRINTs using StructuredSigprintEncoder
    - Applies policy to produce RHZ Stylus commands
    - Sends commands over serial if `stylus` provided
    - Logs events to JSONL if `jsonl_path` given
    """
    encoder = StructuredSigprintEncoder(ch_names, sample_rate=fs, lockin_freq=center_hz, amp_gain=amp_gain)
    pol = policy or StylusPolicy()

    epoch = int(round(fs))
    n = data.shape[1]
    t0 = 0
    f_json = open(jsonl_path, "a") if jsonl_path else None
    try:
        while t0 + epoch <= n:
            seg = data[:, t0:t0 + epoch]
            eeg_epoch = {name: seg[i] for i, name in enumerate(ch_names)}
            code, omega = encoder.process_epoch(eeg_epoch)
            cmd = pol.decide(code, omega)

            event = {
                "t_start": float(t0 / fs),
                "t_end": float((t0 + epoch) / fs),
                "code": code,
                "omega": {
                    "coherence": omega.coherence,
                    "plv": omega.plv,
                    "entropy": omega.entropy,
                    "transition": getattr(omega, "transition", "INIT"),
                },
                "stylus_cmd": cmd,
            }
            if verbose:
                msg = (
                    f"{event['t_start']:.3f}-{event['t_end']:.3f}s\t{code}\t"
                    f"{event['omega']['transition']}\t"
                    f"coh={omega.coherence:.1f}\tplv={omega.plv:.2f}"
                )
                print(msg)

            if stylus is not None and cmd is not None:
                stylus.send_command(cmd)

            if f_json is not None:
                f_json.write(json.dumps(event) + "\n")

            t0 += epoch
    finally:
        if f_json is not None:
            f_json.close()
