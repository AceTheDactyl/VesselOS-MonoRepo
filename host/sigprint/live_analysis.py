from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

from .analysis import AIAnalyzer
from .db import ConsciousnessDB


class LiveAnalyzer:
    """
    Tie Voice Journal ↔ AI Analysis in near‑real‑time by tailing the ledger,
    updating the SQLite database, and emitting rolling suggestions.

    Usage pattern:
      la = LiveAnalyzer(db_path, subject_id, ledger_path, session_id=None, label="journal")
      la.run(window=20, print_suggestions=True)
    """

    def __init__(
        self,
        db_path: str,
        subject_id: str,
        ledger_path: str,
        session_id: Optional[int] = None,
        label: Optional[str] = None,
    ) -> None:
        self.db = ConsciousnessDB(db_path)
        self.db.ensure_subject(subject_id)
        self.session_id = session_id or self.db.create_session(subject_id, label=label or "voice-journal")
        self.ledger_path = Path(ledger_path)
        self._fp = None  # type: ignore
        self._pos = 0
        self._an = AIAnalyzer()

    def close(self) -> None:
        try:
            if self._fp:
                self._fp.close()  # type: ignore
        finally:
            self.db.close()

    def _open(self) -> None:
        if not self.ledger_path.exists():
            # Create empty file to allow tail loop
            self.ledger_path.touch()
        self._fp = open(self.ledger_path, "r", encoding="utf-8")
        # Start at current end; we will import historical first via a one‑shot.
        self._pos = self._fp.tell()

    def _import_historical(self) -> None:
        # One‑shot import of full ledger content (idempotent for a fresh DB/session)
        try:
            self.db.import_ledger(self.session_id, self.ledger_path)
        except Exception:
            # Fallback: continue live
            pass

    def _insert_entry(self, obj: Dict) -> None:
        # Insert a single JSONL entry into DB, mirroring import_ledger logic
        cur = self.db._conn.cursor()
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
                    "INSERT INTO journal_entries(" \
                    "session_id, entry_num, session_time, timestamp, text, code, transition, " \
                    "coherence, plv, entropy, hash, prev_hash) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"
                ),
                (self.session_id, entry_num, t_sess, ts, text, code, trans, coh, plv, ent, h, ph),
            )
            # Also mirror into sigprints at the same session time
            cur.execute(
                (
                    "INSERT INTO sigprints(" \
                    "session_id, t_start, t_end, code, coherence, plv, entropy, transition, " \
                    "stage, intensity, pulse_hz, stylus_meta, source) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)"
                ),
                (
                    self.session_id,
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
            self.db._conn.commit()
            return

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
                    "INSERT INTO sigprints(" \
                    "session_id, t_start, t_end, code, coherence, plv, entropy, transition, " \
                    "stage, intensity, pulse_hz, stylus_meta, source) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)"
                ),
                (
                    self.session_id,
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
            self.db._conn.commit()

    def _latest_journal_entries(self, k: int = 20):
        cur = self.db._conn.cursor()
        cur.execute(
            "SELECT timestamp, session_time, entry_num, code, text, transition, coherence, plv, entropy "
            "FROM journal_entries WHERE session_id=? ORDER BY entry_num DESC LIMIT ?",
            (self.session_id, k),
        )
        rows = cur.fetchall()
        return list(reversed(rows))  # return ascending order

    def run(self, window: int = 20, print_suggestions: bool = True, poll_sec: float = 1.0) -> None:
        # Initial import + open tail
        self._import_historical()
        self._open()
        try:
            while True:
                line = self._fp.readline()  # type: ignore
                if not line:
                    time.sleep(poll_sec)
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                self._insert_entry(obj)

                # Rolling AI analysis over latest K journal entries
                rows = self._latest_journal_entries(k=window)
                from .analysis import JournalEntry as JE

                entries = [
                    JE(
                        timestamp=r[0] or "",
                        session_time=float(r[1] or 0.0),
                        entry_num=int(r[2] or 0),
                        sigprint=str(r[3] or ""),
                        text=str(r[4] or ""),
                        transition=str(r[5] or ""),
                        coherence=float(r[6] or 0.0),
                        plv=float(r[7] or 0.0),
                        entropy=float(r[8] or 0.0),
                    )
                    for r in rows
                ]
                if not entries:
                    continue
                out = self._an.analyze(entries)
                self.db.store_insights(self.session_id, kind="ai_live", payload=out)
                if print_suggestions:
                    sugg = out.get("suggestions", [])
                    if sugg:
                        print("[AI] Suggestions:")
                        for s in sugg:
                            print(" -", s)
        finally:
            self.close()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Live Voice Journal ↔ AI Analysis bridge")
    ap.add_argument("--db", type=str, required=True, help="SQLite DB path")
    ap.add_argument("--subject", type=str, required=True, help="Subject identifier")
    ap.add_argument("--ledger", type=str, required=True, help="Path to journal_ledger.jsonl")
    ap.add_argument("--session", type=int, default=None, help="Existing session_id (optional)")
    ap.add_argument("--label", type=str, default=None, help="Session label if creating new")
    ap.add_argument("--window", type=int, default=20, help="Entries window for rolling analysis")
    ap.add_argument("--quiet", action="store_true", help="Do not print suggestions to stdout")
    args = ap.parse_args()

    la = LiveAnalyzer(args.db, args.subject, args.ledger, session_id=args.session, label=args.label)
    la.run(window=args.window, print_suggestions=not args.quiet)


if __name__ == "__main__":
    main()
