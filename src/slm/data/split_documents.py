"""STEP 1 - split documents into train / val / test BEFORE any questions are written.

Why first: if two chunks of the same document land in train and test, the test looks better than it is.
Documents are grouped so that these stay together on one side of the split:
  * documents sharing a source URL (guide parts), and
  * documents with identical text (repeated HMRC pages).
Documents used as relevant sources in the evaluation set are put in `eval_holdout` and never used for training data.

Run:  python -m src.slm.data.split_documents
"""
from __future__ import annotations

import json
from collections import defaultdict

from src.ingestion.validate_dataset import validate_jsonl
from src.rag.config import Settings
from src.rag.schema import Document, sha256_text
from src.slm.data.common import PATHS, bucket


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def make_split(documents: list[Document], eval_set: dict | None = None, train_pct: int = 80, val_pct: int = 10) -> dict:
    uf = _UnionFind()
    by_url: dict[str, str] = {}
    by_text: dict[str, str] = {}
    for d in documents:
        uf.find(d.document_id)
        for table, key in ((by_url, d.source_url.rstrip("/").lower()), (by_text, sha256_text(" ".join(d.text.split())))):
            if key in table:
                uf.union(table[key], d.document_id)
            else:
                table[key] = d.document_id
    groups: dict[str, list[str]] = defaultdict(list)
    for d in documents:
        groups[uf.find(d.document_id)].append(d.document_id)

    eval_docs = {i for q in (eval_set or {}).get("questions", []) for i in q.get("relevant_document_ids", [])}
    held_groups = {uf.find(i) for i in eval_docs if i in uf.parent}
    out = {"train": [], "val": [], "test": [], "eval_holdout": []}
    for root, ids in sorted(groups.items()):
        if root in held_groups:
            out["eval_holdout"] += ids
            continue
        b = bucket(root)
        out["train" if b < train_pct else "val" if b < train_pct + val_pct else "test"] += ids
    return {**out, "counts": {k: len(v) for k, v in out.items()}, "groups": len(groups)}


def main() -> int:
    s = Settings.from_env()
    docs = validate_jsonl(s.documents_path).documents
    es = json.loads(s.eval_set_path.read_text(encoding="utf-8")) if s.eval_set_path.exists() else None
    split = make_split(docs, es)
    PATHS["split"].parent.mkdir(parents=True, exist_ok=True)
    PATHS["split"].write_text(json.dumps(split, indent=1), encoding="utf-8")
    print("split counts:", split["counts"], "| groups:", split["groups"], "->", PATHS["split"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
