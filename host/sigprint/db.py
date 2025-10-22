from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional


DDL = [
    "PRAGMA foreign_keys = ON;",
    """
    CREATE TABLE IF NOT EXISTS subjects (
      subject_id TEXT PRIMARY KEY,
      meta       TEXT,
      created_at REAL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
      session_id INTEGER PRIMARY KEY AUTOINCREMENT,
      subject_id TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE CASCADE,
      label      TEXT,
      started_at REAL,
      meta       TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS sigprints (
      id         INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
      t_start    REAL,
      t_end      REAL,
      code       TEXT,
      coherence  REAL,
      plv        REAL,
      entropy    REAL,
      transition TEXT,
      stage      INTEGER,
      intensity  INTEGER,
      pulse_hz   REAL,
      stylus_meta TEXT,
      source     TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS journal_entries (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id   INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
      entry_num    INTEGER,
      session_time REAL,
      timestamp    TEXT,
      text         TEXT,
      code         TEXT,
      transition   TEXT,
      coherence    REAL,
      plv          REAL,
      entropy      REAL,
      hash         TEXT,
      prev_hash    TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS insights (
      id         INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
      created_at REAL,
      kind       TEXT,
      payload    TEXT
    );
    """,
]


class ConsciousnessDB:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._exec_many(DDL)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def _exec_many(self, stmts: List[str]) -> None:
        cur = self._conn.cursor()
        for s in stmts:
            cur.execute(s)
        self._conn.commit()

    def ensure_subject(self, subject_id: str, meta: Optional[Dict] = None) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO subjects(subject_id, meta, created_at) VALUES(?,?,?)",
            (subject_id, json.dumps(meta or {}), time.time()),
        )
        self._conn.commit()

    def create_session(self, subject_id: str, label: Optional[str] = None, meta: Optional[Dict] = None) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO sessions(subject_id, label, started_at, meta) VALUES(?,?,?,?)",
            (subject_id, label, time.time(), json.dumps(meta or {})),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def import_ledger(self, session_id: int, ledger_path: str | Path) -> Dict[str, int]:
        """Import VoiceJournal/closed-loop events from JSONL ledger into DB."""
        p = Path(ledger_path)
        if not p.exists():
            return {"sigprints": 0, "journals": 0}
        sig_n = 0
        jrn_n = 0
        cur = self._conn.cursor()
        for line in p.read_text().splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # VoiceJournal-style
            if "sigprint" in obj:
                code = obj.get("sigprint")
                omega = obj.get("omega_state", {})
                trans = omega.get("transition")
                coh = omega.get("coherence")
                plv = omega.get("plv")
                ent = omega.get("entropy")
                t_sess = obj.get("session_time")
                text = obj.get("text")
                entry_num = obj.get("entry_num")
                ts = obj.get("timestamp")
                h = obj.get("hash")
                ph = obj.get("prev_hash")

                cur.execute(
                    (
                        "INSERT INTO journal_entries("
                        "session_id, entry_num, session_time, timestamp, text, code, transition, "
                        "coherence, plv, entropy, hash, prev_hash) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"
                    ),
                    (session_id, entry_num, t_sess, ts, text, code, trans, coh, plv, ent, h, ph),
                )
                jrn_n += 1

                # Also insert a sigprint row (align as instantaneous at session_time)
                cur.execute(
                    (
                        "INSERT INTO sigprints("
                        "session_id, t_start, t_end, code, coherence, plv, entropy, transition, "
                        "stage, intensity, pulse_hz, stylus_meta, source) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)"
                    ),
                    (
                        session_id,
                        t_sess,
                        t_sess,
                        code,
                        coh,
                        plv,
                        ent,
                        trans,
                        None,
                        None,
                        None,
                        json.dumps(obj.get("stylus")),
                        "voice",
                    ),
                )
                sig_n += 1
                continue

            # Closed-loop controller-style
            if "code" in obj and "omega" in obj:
                code = obj.get("code")
                omega = obj.get("omega", {})
                coh = omega.get("coherence")
                plv = omega.get("plv")
                ent = omega.get("entropy")
                trans = omega.get("transition")
                t_start = obj.get("start")
                t_end = obj.get("end")
                styl = obj.get("stylus_cmd") or {}
                stage = styl.get("stage")
                intensity = styl.get("intensity")
                pulse_hz = styl.get("pulse_hz")
                cur.execute(
                    (
                        "INSERT INTO sigprints("
                        "session_id, t_start, t_end, code, coherence, plv, entropy, transition, "
                        "stage, intensity, pulse_hz, stylus_meta, source) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)"
                    ),
                    (
                        session_id,
                        t_start,
                        t_end,
                        code,
                        coh,
                        plv,
                        ent,
                        trans,
                        stage,
                        intensity,
                        pulse_hz,
                        json.dumps(styl),
                        "controller",
                    ),
                )
                sig_n += 1
        self._conn.commit()
        return {"sigprints": sig_n, "journals": jrn_n}

    def session_summary(self, session_id: int) -> Dict:
        cur = self._conn.cursor()
        # Counts by transition
        cur.execute(
            "SELECT transition, COUNT(1) as n FROM sigprints WHERE session_id=? GROUP BY transition",
            (session_id,),
        )
        trans = {row[0] or "": row[1] for row in cur.fetchall()}
        # Metrics
        cur.execute(
            "SELECT AVG(coherence), AVG(plv), AVG(entropy) FROM sigprints WHERE session_id=?",
            (session_id,),
        )
        coh_mean, plv_mean, ent_mean = cur.fetchone()
        # Top codes
        cur.execute(
            "SELECT code, COUNT(1) as n FROM sigprints WHERE session_id=? GROUP BY code ORDER BY n DESC LIMIT 10",
            (session_id,),
        )
        top_codes = [(row[0], row[1]) for row in cur.fetchall()]
        return {
            "transitions": trans,
            "coherence_mean": coh_mean or 0.0,
            "plv_mean": plv_mean or 0.0,
            "entropy_mean": ent_mean or 0.0,
            "top_codes": top_codes,
        }

    def store_insights(self, session_id: int, kind: str, payload: Dict) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO insights(session_id, created_at, kind, payload) VALUES(?,?,?,?)",
            (session_id, time.time(), kind, json.dumps(payload)),
        )
        self._conn.commit()
        return int(cur.lastrowid)
