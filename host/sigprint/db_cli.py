from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .db import ConsciousnessDB
from .analysis import AIAnalyzer, JournalEntry


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Consciousness Database (SQLite) tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init", help="Initialize database and create subject/session")
    i.add_argument("--db", type=str, required=True)
    i.add_argument("--subject", type=str, required=True)
    i.add_argument("--label", type=str, default=None)

    imp = sub.add_parser("import-ledger", help="Import a JSONL ledger into a session")
    imp.add_argument("--db", type=str, required=True)
    imp.add_argument("--subject", type=str, required=True)
    imp.add_argument("--session", type=int, default=None, help="Existing session_id; if missing, creates one")
    imp.add_argument("--label", type=str, default=None, help="Label for new session")
    imp.add_argument("ledger", type=str)

    s = sub.add_parser("summary", help="Print session summary")
    s.add_argument("--db", type=str, required=True)
    s.add_argument("--session", type=int, required=True)

    a = sub.add_parser("analyze", help="Run AI analysis on journal entries and store suggestions")
    a.add_argument("--db", type=str, required=True)
    a.add_argument("--session", type=int, required=True)
    a.add_argument("--json", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "init":
        db = ConsciousnessDB(args.db)
        db.ensure_subject(args.subject)
        sid = db.create_session(args.subject, label=args.label)
        print(json.dumps({"subject": args.subject, "session_id": sid}))
        db.close()
        return 0

    if args.cmd == "import-ledger":
        db = ConsciousnessDB(args.db)
        db.ensure_subject(args.subject)
        sid = args.session or db.create_session(args.subject, label=args.label)
        res = db.import_ledger(sid, args.ledger)
        print(json.dumps({"session_id": sid, **res}))
        db.close()
        return 0

    if args.cmd == "summary":
        db = ConsciousnessDB(args.db)
        res = db.session_summary(args.session)
        print(json.dumps(res))
        db.close()
        return 0

    if args.cmd == "analyze":
        db = ConsciousnessDB(args.db)
        # Pull journal entries for the session
        import sqlite3

        cur = db._conn.cursor()
        cur.execute(
            "SELECT timestamp, session_time, entry_num, sigprints.code, text, journal_entries.transition, journal_entries.coherence, journal_entries.plv, journal_entries.entropy "
            "FROM journal_entries LEFT JOIN sigprints ON journal_entries.code = sigprints.code AND journal_entries.session_id = sigprints.session_id "
            "WHERE journal_entries.session_id=? ORDER BY entry_num ASC",
            (args.session,),
        )
        rows = cur.fetchall()
        # Build analysis entries
        entries = []
        from .analysis import JournalEntry as JE  # alias
        for r in rows:
            entries.append(
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
            )
        an = AIAnalyzer()
        out = an.analyze(entries)
        db.store_insights(args.session, kind="ai_analysis", payload=out)
        if args.json:
            print(json.dumps(out))
        else:
            print(json.dumps({
                "count": out.get("count", 0),
                "gates": out.get("gates", 0),
                "loops": out.get("loops", 0),
                "suggestions": out.get("suggestions", []),
            }))
        db.close()
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

