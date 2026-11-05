"""Shared helpers for the SLM training-data scripts: JSONL I/O, resumable output, file locations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

from src.rag.config import PROJECT_ROOT

SLM_DIR = PROJECT_ROOT / "data" / "slm"
PATHS = {
    "split": SLM_DIR / "split_ids.json",
    "questions": SLM_DIR / "questions.jsonl",
    "inputs": SLM_DIR / "inputs.jsonl",
    "drafts": SLM_DIR / "drafts.jsonl",
    "review": SLM_DIR / "review.jsonl",
    "offtopic": SLM_DIR / "offtopic_questions.txt",
}


def read_jsonl(path: Path) -> Iterator[dict]:
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def append_jsonl(path: Path, record: dict) -> None:
    """Append one record and flush, so an interrupted run loses at most the record in progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def done_ids(path: Path, key: str) -> set[str]:
    return {r[key] for r in read_jsonl(path)}


def stable_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def bucket(key: str) -> int:
    """Deterministic 0-99 bucket for a key (same key -> same bucket on every machine)."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
