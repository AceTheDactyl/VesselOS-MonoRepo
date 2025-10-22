from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


HEX0 = "0" * 64


def _canonical_json(obj: Dict) -> str:
    """Canonical JSON string for hashing (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def compute_entry_hash(entry: Dict, prev_hash: str) -> str:
    """Compute the SHA-256 hash of an entry using the VoiceJournal scheme.

    Hash payload = prev_hash + canonical_json(entry_without_hash_fields)
    Returns lowercase hex digest.
    """
    e = dict(entry)
    e.pop("hash", None)
    e.pop("prev_hash", None)
    e.pop("sig", None)
    payload = (prev_hash or HEX0) + _canonical_json(e)
    return hashlib.sha256(payload.encode()).hexdigest()


def compute_hmac(entry: Dict, prev_hash: str, secret: bytes) -> str:
    """Optional HMAC-SHA256 signature over the same payload as compute_entry_hash."""
    e = dict(entry)
    e.pop("hash", None)
    e.pop("prev_hash", None)
    e.pop("sig", None)
    payload = (prev_hash or HEX0) + _canonical_json(e)
    mac = hmac.new(secret, payload.encode(), hashlib.sha256).digest()
    return base64.b64encode(mac).decode()


@dataclass
class LedgerIssue:
    index: int
    message: str


@dataclass
class LedgerReport:
    ok: bool
    entries: int
    issues: List[LedgerIssue]
    checkpoints: int


def verify_ledger(path: str | Path, secret: Optional[bytes] = None) -> LedgerReport:
    """Verify JSONL ledger integrity and optional HMAC signatures.

    - Checks hash chain continuity (prev_hash, hash)
    - If secret is provided and entry has 'sig', verifies HMAC
    - Supports 'type': 'checkpoint' entries with a 'merkle_root' that is
      recomputed over the prior N entries listed in 'span': [start, end].
    """
    issues: List[LedgerIssue] = []
    checkpoints = 0
    p = Path(path)
    if not p.exists():
        return LedgerReport(False, 0, [LedgerIssue(0, "file not found")], 0)

    lines = p.read_text().splitlines()
    prev = HEX0
    objects: List[Dict] = []
    for i, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except Exception:
            issues.append(LedgerIssue(i, "invalid JSON"))
            continue
        h = str(obj.get("hash", ""))
        ph = str(obj.get("prev_hash", "")) or prev
        calc = compute_entry_hash(obj, ph)
        if h != calc:
            issues.append(LedgerIssue(i, "hash mismatch"))
        if ph != prev:
            issues.append(LedgerIssue(i, "prev_hash mismatch"))
        if secret is not None and "sig" in obj:
            sig = str(obj.get("sig", ""))
            calc_sig = compute_hmac(obj, ph, secret)
            if not hmac.compare_digest(sig, calc_sig):
                issues.append(LedgerIssue(i, "hmac signature mismatch"))
        prev = h
        objects.append(obj)

    # Verify checkpoints if present
    for i, obj in enumerate(objects):
        if obj.get("type") == "checkpoint":
            checkpoints += 1
            span = obj.get("span") or []
            if not (isinstance(span, list) and len(span) == 2):
                issues.append(LedgerIssue(i, "checkpoint missing span"))
                continue
            a, b = int(span[0]), int(span[1])
            if a < 0 or b > len(objects) or a >= b:
                issues.append(LedgerIssue(i, "checkpoint span invalid"))
                continue
            root = merkle_root([objects[j].get("hash", "") for j in range(a, b)])
            if obj.get("merkle_root") != root:
                issues.append(LedgerIssue(i, "checkpoint merkle_root mismatch"))

    return LedgerReport(ok=(len(issues) == 0), entries=len(objects), issues=issues, checkpoints=checkpoints)


def merkle_root(hashes: Iterable[str]) -> str:
    """Compute a SHA-256 Merkle root from an iterable of hex hashes."""
    layer = [bytes.fromhex(h) if isinstance(h, str) else h for h in hashes]
    if not layer:
        return hashlib.sha256(b"").hexdigest()
    while len(layer) > 1:
        nxt: List[bytes] = []
        for i in range(0, len(layer), 2):
            a = layer[i]
            b = layer[i + 1] if i + 1 < len(layer) else a
            nxt.append(hashlib.sha256(a + b).digest())
        layer = nxt
    return layer[0].hex()


def write_checkpoint(path: str | Path, start_index: int, end_index: int) -> Dict:
    """Append a checkpoint entry with Merkle root for [start_index, end_index)."""
    p = Path(path)
    lines = p.read_text().splitlines() if p.exists() else []
    prev_hash = json.loads(lines[-1]).get("hash", HEX0) if lines else HEX0
    objs = [json.loads(line) for line in lines]
    if start_index < 0:
        start_index = 0
    if end_index > len(objs):
        end_index = len(objs)
    if start_index >= end_index:
        raise ValueError("invalid span")
    root = merkle_root([objs[i].get("hash", "") for i in range(start_index, end_index)])
    entry = {"type": "checkpoint", "span": [start_index, end_index], "merkle_root": root}
    h = compute_entry_hash(entry, prev_hash)
    entry["prev_hash"] = prev_hash
    entry["hash"] = h
    with open(p, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def append_signed(path: str | Path, entry: Dict, secret: Optional[bytes] = None) -> Dict:
    """Append an entry to the ledger, computing prev/hash and optional HMAC sig."""
    p = Path(path)
    lines = p.read_text().splitlines() if p.exists() else []
    prev_hash = json.loads(lines[-1]).get("hash", HEX0) if lines else HEX0
    h = compute_entry_hash(entry, prev_hash)
    entry = dict(entry)
    entry["prev_hash"] = prev_hash
    entry["hash"] = h
    if secret is not None:
        entry["sig"] = compute_hmac(entry, prev_hash, secret)
    with open(p, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry
