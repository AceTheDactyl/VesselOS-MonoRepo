from __future__ import annotations

import argparse
import json
from typing import Optional

from .analysis import AIAnalyzer


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="AI analysis of Voice Journal ledger (JSONL)")
    p.add_argument("ledger", type=str, help="Path to journal_ledger.jsonl")
    p.add_argument("--json", action="store_true", help="Emit full JSON analysis")
    p.add_argument("--top", type=int, default=10, help="Top terms to display")
    args = p.parse_args(argv)

    an = AIAnalyzer()
    entries = an.load_ledger(args.ledger)
    res = an.analyze(entries)

    if args.json:
        print(json.dumps(res))
        return 0

    # Pretty text summary
    print(f"Entries: {res.get('count',0)}  Gates: {res.get('gates',0)}  Loops: {res.get('loops',0)}")
    m = res.get('metrics', {})
    coh = m.get('coherence', {})
    plv = m.get('plv', {})
    print(
        f"Coherence mean/p50/p90: {coh.get('mean', 0):.1f} / "
        f"{coh.get('p50', 0):.1f} / {coh.get('p90', 0):.1f}"
    )
    print(
        f"PLV mean/p50/p90:       {plv.get('mean', 0):.2f} / "
        f"{plv.get('p50', 0):.2f} / {plv.get('p90', 0):.2f}"
    )
    s = res.get('sentiment', {})
    print(f"Sentiment pos/neg mean:  {s.get('pos_mean',0):.2f} / {s.get('neg_mean',0):.2f}")
    print(f"corr(pos,coh)={s.get('corr_pos_coh',0):.2f}  corr(neg,coh)={s.get('corr_neg_coh',0):.2f}")

    print("\nTop gate terms:")
    for term, score in res.get('top_terms', {}).get('gates', [])[: args.top]:
        print(f"  {term:>16s}  {score:.3f}")
    print("Top loop terms:")
    for term, score in res.get('top_terms', {}).get('loops', [])[: args.top]:
        print(f"  {term:>16s}  {score:.3f}")

    print("\nSuggestions:")
    for line in res.get('suggestions', []):
        print(f"- {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
