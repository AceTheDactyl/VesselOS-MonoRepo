**SIGPRINT Encoders (Ω-Field Semantics)**
- Input: EEG array `channels x samples` (.npy or .csv)
- Core: lock-in demodulation at center frequency (default 8 Hz) using numpy-only I/Q and moving-average LPF
- Output: 20-digit SIGPRINT code, Ω-state summaries, and feature dictionary

Usage
- Single window (whole file):
  `python3 -m host.sigprint.cli data.npy --fs 256`

- Sliding windows:
  `python3 -m host.sigprint.cli data.csv --fs 250 --window 10 --step 5 --json`

- Structured 20-digit encoder (phase/amp/coherence mapping, 1 s windows):
  `python3 -m host.sigprint.structured_cli data.npy --fs 250 --names Fp1,Fp2,F3,F4,P3,P4,O1,O2`
  - Optional coherence scaling: `--amp-gain 1e6` (if input units are volts)

Input format
- `.npy`: numpy array of shape `[channels, samples]` preferred. If `[samples, channels]`, the CLI heuristically transposes.
- `.csv`: numeric CSV; use `--delimiter "\t"` for TSV.
- Channel names: `--names Cz,Pz,Oz`

Ω-field mapping
- ψ per channel: mean of normalized envelope from lock-in demod
- Λ per channel: circular mean of demodulated phase
- Coherence: mean of `A_norm**2` across channels/time (∫ψ² proxy)
- PLV: mean pairwise phase-locking value
- Entropy: normalized phase entropy across all samples/channels

20-digit code
- Deterministic quantization of 20 bounded features (amplitude stats, envelope correlation, phase synchrony, Kuramoto, phase distribution, temporal variability) into digits 0–9.

Structured mapping (alternative)
- Digits 1–4: Phase topology (front–back, left–right)
- Digits 5–8: Amplitude distribution (frontal%, left%)
- Digits 9–12: Coherence (0–99) and PLV (0–99)
- Digits 13–18: Context (entropy 0–99, stylus stage 00–99, reserved 00)
- Digits 19–20: Checksum (SHA‑256 mod 100)

Notes
- The lock-in approach approximates narrowband quadrature demodulation around the carrier; change `--center` to adapt to IAF.
- No medical use. For research and exploratory feedback only.
