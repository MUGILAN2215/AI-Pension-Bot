"""A small BM25 keyword index (no extra dependency).

Why: dense vectors capture meaning but can miss exact codes such as "PC10S" or "BR19".
BM25 scores chunks by how many (rare) query words they contain, so exact terms are never missed.
The index is built in memory from the chunk payloads stored in Qdrant (a few seconds for ~4,000 chunks).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD = re.compile(r"[a-z0-9£%]+(?:[./-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


class BM25Index:
    def __init__(self, payloads: Iterable[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.payloads: list[dict[str, Any]] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for p in payloads:
            # title and heading are searchable too: they often carry the topic words
            toks = tokenize(f"{p.get('title', '')} {p.get('heading_path', '')} {p.get('text', '')}")
            i = len(self.payloads)
            self.payloads.append(p)
            self.lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        n = len(self.payloads)
        self.avg_len = (sum(self.lengths) / n) if n else 0.0
        self.idf = {t: math.log(1 + (n - len(po) + 0.5) / (len(po) + 0.5)) for t, po in self.postings.items()}

    def __len__(self) -> int:
        return len(self.payloads)

    def search(self, query: str, top_k: int, filters: dict[str, Any] | None = None) -> list[tuple[dict, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1))
                scores[i] += idf * tf * (self.k1 + 1) / norm
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        out: list[tuple[dict, float]] = []
        for i, sc in ranked:
            p = self.payloads[i]
            if filters and not all((p.get(k) in v) if isinstance(v, (list, tuple, set)) else (p.get(k) == v) for k, v in filters.items()):
                continue
            out.append((p, sc))
            if len(out) >= top_k:
                break
        return out
