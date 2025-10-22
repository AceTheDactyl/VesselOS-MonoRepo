from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class JournalEntry:
    timestamp: str
    session_time: float
    entry_num: int
    sigprint: str
    text: str
    transition: str
    coherence: float
    plv: float
    entropy: float


class AIAnalyzer:
    """
    Lightweight AI analysis over VoiceJournal ledger JSONL.

    Goals:
      - Summarize gates/loops and Ω metrics
      - Tokenize + TF-IDF topics around gates vs loops (no heavy deps)
      - Simple lexicon-based sentiment/arousal scoring
      - Correlate text features with coherence/PLV/entropy
    """

    _token_re = re.compile(r"[A-Za-z']+")
    _stop = {
        'the','and','a','an','to','of','in','on','for','with','is','it','this','that','was','as','at','be','by','from',
        'are','or','but','not','so','if','then','when','while','my','i','me','we','our','you','your','they','their','he','she','him','her',
        'there','here','into','out','up','down','over','under','about','just','like','really','very','have','has','had','do','did','done',
    }
    _pos = {
        'calm','clear','relaxed','open','grateful','loving','joy','ease','peace','flow','present','focused','connected','insight','understand','release','light',
    }
    _neg = {
        'anxious','anxiety','worry','fear','angry','sad','depressed','tired','pain','stress','tense','confused','stuck','rumination','noise','overwhelmed','doubt',
    }

    def load_ledger(self, path: str | Path) -> List[JournalEntry]:
        entries: List[JournalEntry] = []
        p = Path(path)
        if not p.exists():
            return entries
        for line in p.read_text().splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            omega = obj.get('omega_state', {})
            entries.append(
                JournalEntry(
                    timestamp=obj.get('timestamp', ''),
                    session_time=float(obj.get('session_time', 0.0)),
                    entry_num=int(obj.get('entry_num', 0)),
                    sigprint=str(obj.get('sigprint', '')),
                    text=str(obj.get('text', '')),
                    transition=str(omega.get('transition', obj.get('transition', '')) or ''),
                    coherence=float(omega.get('coherence', 0.0)),
                    plv=float(omega.get('plv', 0.0)),
                    entropy=float(omega.get('entropy', 0.0)),
                )
            )
        return entries

    def tokenize(self, text: str) -> List[str]:
        toks = [t.lower() for t in self._token_re.findall(text)]
        return [t for t in toks if t not in self._stop and len(t) >= 3]

    def sentiment(self, tokens: Iterable[str]) -> Tuple[float, float]:
        pos = sum(1 for t in tokens if t in self._pos)
        neg = sum(1 for t in tokens if t in self._neg)
        total = max(1, pos + neg)
        return pos / total, neg / total

    def tfidf(self, docs: List[List[str]]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        vocab: Dict[str, int] = {}
        for doc in docs:
            for t in set(doc):
                vocab[t] = vocab.get(t, 0) + 1
        N = max(1, len(docs))
        idf: Dict[str, float] = {t: math.log(N / (df)) for t, df in vocab.items()}
        per_doc: List[Dict[str, float]] = []
        for doc in docs:
            tf: Dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            denom = float(len(doc) or 1)
            per_doc.append({t: (tf[t] / denom) * idf.get(t, 0.0) for t in tf})
        # Return corpus-level IDF and per-doc TF-IDF
        return idf, per_doc

    def analyze(self, entries: List[JournalEntry]) -> Dict:
        if not entries:
            return {"count": 0}

        # Tokenize and sentiments
        tokens = [self.tokenize(e.text) for e in entries]
        sentiments = [self.sentiment(t) for t in tokens]

        # Split gates vs loops
        gates_idx = [i for i, e in enumerate(entries) if e.transition == 'GATE']
        loops_idx = [i for i, e in enumerate(entries) if e.transition == 'LOOP']

        idf, tfidf_docs = self.tfidf(tokens)

        def top_terms(idxs: List[int], k: int = 12) -> List[Tuple[str, float]]:
            agg: Dict[str, float] = {}
            for i in idxs:
                for t, w in tfidf_docs[i].items():
                    agg[t] = agg.get(t, 0.0) + w
            return sorted(agg.items(), key=lambda x: x[1], reverse=True)[:k]

        def stats(vals: List[float]) -> Dict[str, float]:
            if not vals:
                return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
            a = np.array(vals, dtype=float)
            return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90))}

        coh = [e.coherence for e in entries]
        plv = [e.plv for e in entries]
        ent = [e.entropy for e in entries]

        # Sentiment correlations (Pearson with simple scores)
        pos = [p for p, n in sentiments]
        neg = [n for p, n in sentiments]
        def corr(a: List[float], b: List[float]) -> float:
            if len(a) < 2:
                return 0.0
            A = np.array(a, dtype=float)
            B = np.array(b, dtype=float)
            if A.std() == 0.0 or B.std() == 0.0:
                return 0.0
            return float(np.corrcoef(A, B)[0, 1])

        results = {
            "count": len(entries),
            "gates": len(gates_idx),
            "loops": len(loops_idx),
            "metrics": {
                "coherence": stats(coh),
                "plv": stats(plv),
                "entropy": stats(ent),
            },
            "sentiment": {
                "pos_mean": float(np.mean(pos)) if pos else 0.0,
                "neg_mean": float(np.mean(neg)) if neg else 0.0,
                "corr_pos_coh": corr(pos, coh),
                "corr_neg_coh": corr(neg, coh),
                "corr_pos_plv": corr(pos, plv),
                "corr_neg_plv": corr(neg, plv),
            },
            "top_terms": {
                "gates": top_terms(gates_idx),
                "loops": top_terms(loops_idx),
            },
            "highlights": self._gate_highlights(entries, window=1),
        }
        results["suggestions"] = self._suggest(results)
        return results

    def _gate_highlights(self, entries: List[JournalEntry], window: int = 1) -> List[Dict]:
        out: List[Dict] = []
        for i, e in enumerate(entries):
            if e.transition != 'GATE':
                continue
            before = entries[i - 1].text if i - 1 >= 0 else ""
            after = entries[i + 1].text if i + 1 < len(entries) else ""
            out.append({
                "entry": e.entry_num,
                "time": e.session_time,
                "coherence": e.coherence,
                "plv": e.plv,
                "text_before": before,
                "text": e.text,
                "text_after": after,
            })
        return out

    def _suggest(self, analysis: Dict) -> List[str]:
        sug: List[str] = []
        gates = analysis.get("gates", 0)
        loops = analysis.get("loops", 0)
        coh = analysis.get("metrics", {}).get("coherence", {}).get("mean", 0.0)
        pos_mean = analysis.get("sentiment", {}).get("pos_mean", 0.0)
        neg_mean = analysis.get("sentiment", {}).get("neg_mean", 0.0)

        if gates > 0 and pos_mean >= neg_mean:
            sug.append("Increase stylus stage by +1 during positive gates to consolidate insights.")
        if gates > 0 and neg_mean > pos_mean:
            sug.append("Reduce pulse intensity during negative-valence gates; add breath guidance.")
        if loops > 3 and coh < 30:
            sug.append("Add short perturbations (e.g., brief breath holds) to escape low-coherence loops.")
        if analysis.get("sentiment", {}).get("corr_pos_coh", 0.0) > 0.2:
            sug.append("Positive affect correlates with coherence; reinforce practices that increase positive terms.")
        if analysis.get("sentiment", {}).get("corr_neg_coh", 0.0) < -0.2:
            sug.append("Negative affect inversely correlates with coherence; schedule downshifts before challenging tasks.")
        if not sug:
            sug.append("Maintain current protocol; collect more data for pattern discovery.")
        return sug

