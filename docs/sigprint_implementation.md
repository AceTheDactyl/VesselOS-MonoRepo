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

