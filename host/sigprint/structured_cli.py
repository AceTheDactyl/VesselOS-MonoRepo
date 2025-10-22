from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from .system import StructuredSigprintEncoder


def _load_matrix(path: Path, delimiter: str = ",") -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
    else:
        arr = np.loadtxt(str(path), delimiter=delimiter)
    if arr.ndim == 1:
        arr = arr[None, :]
    # expect channels x samples; transpose if necessary (heuristic)
    if arr.shape[0] > arr.shape[1]:
        # likely samples x channels -> transpose
        arr = arr.T
    return arr.astype(float, copy=False)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Structured 20-digit SIGPRINT encoder (Ω semantics)")
    p.add_argument("file", type=str, help="Path to EEG array (.npy or .csv)")
    p.add_argument("--fs", type=float, required=True, help="Sampling rate (Hz)")
    p.add_argument("--center", type=float, default=8.0, help="Lock-in center frequency (Hz)")
    p.add_argument("--names", type=str, required=True, help="Comma-separated channel names matching rows")
    p.add_argument("--json", action="store_true", help="Emit JSON lines per 1s window")
    p.add_argument("--delimiter", type=str, default=",", help="CSV delimiter")
    p.add_argument("--amp-gain", type=float, default=1.0, help="Amplitude gain for coherence scaling")

    args = p.parse_args(argv)
    path = Path(args.file)
    data = _load_matrix(path, delimiter=args.delimiter)

    ch_names = [s.strip() for s in args.names.split(",")]
    if len(ch_names) != data.shape[0]:
        print(f"error: expected {data.shape[0]} channel names, got {len(ch_names)}", file=sys.stderr)
        return 2

    enc = StructuredSigprintEncoder(ch_names, sample_rate=args.fs, lockin_freq=args.center, amp_gain=args.amp_gain)

    fs = args.fs
    epoch = int(round(fs))
    n = data.shape[1]
    t0 = 0
    while t0 + epoch <= n:
        seg = data[:, t0 : t0 + epoch]
        eeg_epoch = {name: seg[i] for i, name in enumerate(ch_names)}
        code, omega = enc.process_epoch(eeg_epoch)
        if args.json:
            out = {
                "start": float(t0 / fs),
                "end": float((t0 + epoch) / fs),
                "code": code,
                "omega": {
                    "coherence": omega.coherence,
                    "plv": omega.plv,
                    "entropy": omega.entropy,
                    "psi": omega.psi,
                },
            }
            print(json.dumps(out))
        else:
            print(f"{t0/fs:.3f}-{(t0+epoch)/fs:.3f}s\t{code}")
        t0 += epoch
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
