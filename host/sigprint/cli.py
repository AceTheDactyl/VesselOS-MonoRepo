from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from .encoder import SigprintEncoder


def _load_array(path: Path, delimiter: str = ",") -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
    else:
        arr = np.loadtxt(str(path), delimiter=delimiter)
    if arr.ndim == 1:
        arr = arr[None, :]
    # Expect channels x samples; if the array is samples x channels, transpose
    if arr.shape[0] < arr.shape[1]:
        # likely channels x samples already; keep
        pass
    else:
        # If samples x channels, transpose to channels x samples
        # Heuristic: if columns >= rows and looks like time along axis 0, but we can't know.
        # Expose a flag via CLI if needed; for now transpose to be safe when rows > cols.
        arr = arr.T
    return arr.astype(float, copy=False)


def _window_indices(n_samples: int, fs: float, win_sec: float, step_sec: Optional[float]) -> List[tuple[int, int]]:
    w = int(round(win_sec * fs))
    if w <= 0:
        raise ValueError("window must be > 0 seconds")
    s = int(round((step_sec if step_sec else win_sec) * fs))
    idx = []
    i = 0
    while i + w <= n_samples:
        idx.append((i, i + w))
        i += s
    if not idx and n_samples >= 2:
        idx.append((0, n_samples))
    return idx


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="SIGPRINT encoder (Ω-field semantics)")
    p.add_argument("file", type=str, help="Path to EEG array (.npy or .csv)")
    p.add_argument("--fs", type=float, required=True, help="Sampling rate (Hz)")
    p.add_argument("--center", type=float, default=8.0, help="Carrier center frequency (Hz)")
    p.add_argument("--lpf", type=float, default=0.5, help="Low-pass smoothing (seconds)")
    p.add_argument("--window", type=float, default=None, help="Window length (seconds); default = whole file")
    p.add_argument("--step", type=float, default=None, help="Step length (seconds) for sliding windows")
    p.add_argument("--delimiter", type=str, default=",", help="CSV delimiter (if file is .csv)")
    p.add_argument("--names", type=str, default=None, help="Comma-separated channel names")
    p.add_argument("--json", action="store_true", help="Emit JSON lines with code + features")

    args = p.parse_args(argv)
    path = Path(args.file)

    data = _load_array(path, delimiter=args.delimiter)
    n_channels, n_samples = data.shape
    ch_names = None
    if args.names:
        ch_names = [s.strip() for s in args.names.split(",")]
        if len(ch_names) != n_channels:
            print(f"error: expected {n_channels} names, got {len(ch_names)}", file=sys.stderr)
            return 2

    enc = SigprintEncoder(fs=args.fs, center_hz=args.center, lpf_seconds=args.lpf)

    # Windowing
    if args.window is None:
        code, omega, feats = enc.encode(data, ch_names=ch_names, timestamp=time.time())
        if args.json:
            out = {"code": code, "timestamp": omega.timestamp, "features": feats, "psi": omega.psi, "lambda": omega.lambda_, "coherence": omega.coherence, "plv": omega.plv, "entropy": omega.entropy}
            print(json.dumps(out))
        else:
            print(code)
        return 0

    idx = _window_indices(n_samples, args.fs, args.window, args.step)
    for (a, b) in idx:
        code, omega, feats = enc.encode(data[:, a:b], ch_names=ch_names, timestamp=time.time())
        if args.json:
            out = {"code": code, "start": float(a/args.fs), "end": float(b/args.fs), "timestamp": omega.timestamp, "features": feats}
            print(json.dumps(out))
        else:
            print(f"{a/args.fs:.3f}-{b/args.fs:.3f}s\t{code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

