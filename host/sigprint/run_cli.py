from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from .controller import run_closed_loop, StylusPolicy
from .system import StylusInterface


def _load_matrix(path: Path, delimiter: str = ",") -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
    else:
        arr = np.loadtxt(str(path), delimiter=delimiter)
    if arr.ndim == 1:
        arr = arr[None, :]
    # Expect channels x samples; transpose if needed
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return arr.astype(float, copy=False)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run closed loop: SIGPRINT ↔ RHZ Stylus")
    p.add_argument("file", type=str, help="Path to EEG array (.npy or .csv)")
    p.add_argument("--fs", type=float, required=True, help="Sampling rate (Hz)")
    p.add_argument("--center", type=float, default=8.0, help="Lock-in center frequency (Hz)")
    p.add_argument("--names", type=str, required=True, help="Comma-separated channel names")
    p.add_argument("--amp-gain", type=float, default=1.0, help="Amplitude gain for coherence scaling")
    p.add_argument("--delimiter", type=str, default=",", help="CSV delimiter")
    p.add_argument("--jsonl", type=str, default=None, help="Path to JSONL event log")
    p.add_argument("--serial-port", type=str, default=None, help="Stylus serial port (e.g., /dev/ttyUSB0)")
    p.add_argument("--baud", type=int, default=115200, help="Stylus baud rate")
    p.add_argument("--dry", action="store_true", help="Do not send commands; print only")

    args = p.parse_args(argv)
    path = Path(args.file)
    data = _load_matrix(path, delimiter=args.delimiter)
    ch_names = [s.strip() for s in args.names.split(",")]
    if len(ch_names) != data.shape[0]:
        print(f"error: expected {data.shape[0]} channel names, got {len(ch_names)}", file=sys.stderr)
        return 2

    stylus = None
    if args.serial_port and not args.dry:
        try:
            stylus = StylusInterface(args.serial_port, args.baud)
            stylus.connect()
            stylus.start_stream()  # optional readback
        except Exception as e:
            print(f"warning: could not open stylus port: {e}")
            stylus = None

    policy = StylusPolicy()
    run_closed_loop(
        data,
        ch_names,
        fs=args.fs,
        center_hz=args.center,
        amp_gain=args.amp_gain,
        stylus=stylus,
        policy=policy,
        jsonl_path=args.jsonl,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
