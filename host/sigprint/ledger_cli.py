from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

from .ledger import append_signed, verify_ledger, write_checkpoint


def _load_secret(path: Optional[str]) -> bytes | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"secret file not found: {p}")
    return p.read_bytes().strip()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Immutable ledger tools for SIGPRINT")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="Verify ledger integrity")
    v.add_argument("ledger", type=str)
    v.add_argument("--secret", type=str, default=None, help="Path to secret for HMAC verification")
    v.add_argument("--json", action="store_true")

    a = sub.add_parser("append", help="Append a JSON entry (from stdin)")
    a.add_argument("ledger", type=str)
    a.add_argument("--secret", type=str, default=None, help="Path to secret for HMAC signature")

    c = sub.add_parser("checkpoint", help="Append a checkpoint with Merkle root for [start,end)")
    c.add_argument("ledger", type=str)
    c.add_argument("start", type=int)
    c.add_argument("end", type=int)

    args = p.parse_args(argv)

    if args.cmd == "verify":
        secret = _load_secret(args.secret)
        rep = verify_ledger(args.ledger, secret=secret)
        if args.json:
            print(json.dumps({
                "ok": rep.ok,
                "entries": rep.entries,
                "checkpoints": rep.checkpoints,
                "issues": [{"index": i.index, "message": i.message} for i in rep.issues],
            }))
        else:
            print(f"ok={rep.ok} entries={rep.entries} checkpoints={rep.checkpoints}")
            for issue in rep.issues:
                print(f"- L{issue.index}: {issue.message}")
        return 0 if rep.ok else 1

    if args.cmd == "append":
        try:
            entry = json.load(sys.stdin)
        except Exception:
            print("error: stdin must be a JSON object", file=sys.stderr)
            return 2
        secret = _load_secret(args.secret)
        out = append_signed(args.ledger, entry, secret=secret)
        print(json.dumps(out))
        return 0

    if args.cmd == "checkpoint":
        out = write_checkpoint(args.ledger, args.start, args.end)
        print(json.dumps(out))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
