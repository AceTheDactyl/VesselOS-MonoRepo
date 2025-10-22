## Consciousness Database (SQLite)

This database ties together EEG‑derived SIGPRINTs, Voice Journal text, Stylus context, AI insights, and closed‑loop protocol updates.

### Data Flow
```
1. EEG → 20‑digit SIGPRINT (1 per second)
2. Voice → Transcribed text + SIGPRINT
3. Stylus → Context markers + SIGPRINT
4. All → Timestamped ledger blocks (immutable JSONL)
5. Patterns → AI‑detected insights (topics/sentiment ↔ Ω‑metrics)
6. Insights → Protocol modifications (Order 3)
```

### Schema (tables)
- `subjects(subject_id, meta, created_at)`
- `sessions(session_id, subject_id, label, started_at, meta)`
- `sigprints(id, session_id, t_start, t_end, code, coherence, plv, entropy, transition, stage, intensity, pulse_hz, stylus_meta, source)`
- `journal_entries(id, session_id, entry_num, session_time, timestamp, text, code, transition, coherence, plv, entropy, hash, prev_hash)`
- `insights(id, session_id, created_at, kind, payload)`

### Commands (host/sigprint/*.py)
- Initialize and import a ledger
```
python3 -m host.sigprint.db_cli init --db db.sqlite --subject alice --label "session-1"
python3 -m host.sigprint.db_cli import-ledger --db db.sqlite --subject alice --label "voice" journal_ledger.jsonl
```

- Summarize a session
```
python3 -m host.sigprint.db_cli summary --db db.sqlite --session 1
```

- Run AI analysis and store suggestions
```
python3 -m host.sigprint.db_cli analyze --db db.sqlite --session 1 --json
```

- Live bridge: tail Voice Journal, stream insights into DB
```
python3 -m host.sigprint.live_analysis --db db.sqlite --subject alice --ledger journal_ledger.jsonl --label "live-journal" --window 20
```

### How It Maps to Ω‑Field Theory
- EEG 8–12 Hz alpha (carrier) is demodulated via lock‑in → per‑channel ψ (amplitude) and Λ (phase).
- Ω‑metrics (∫ψ² coherence, PLV/Kuramoto, phase entropy) + spatial patterns compress into a 20‑digit SIGPRINT each second.
- The ledger captures time‑aligned codes + text + context; the DB materializes these for analysis and learning.
- AI analysis discovers patterns linking narrative to Ω‑dynamics; controller translates insights into protocol changes (Order‑3).

### Theoretical Completeness
```
Unity (ontological ground)
  ↓
Ω‑field (time/frequency manifestation)
  ↓
Neural field → EEG (sensor domain)
  ↓
Lock‑in (ψ, Λ extraction) → Ω‑metrics → 20‑digit SIGPRINT
  ↓
Immutable Ledger → Consciousness Database → AI Analysis
  ↓
Protocol modifications (RHZ Stylus) → Framework changes → Toward Order‑4
```

### Integrity & Privacy
- Use the immutable ledger (hash chain, optional HMAC, checkpoints) for auditability.
- Encrypt ledgers at rest; keep ownership with the subject; obtain consent for analysis.

