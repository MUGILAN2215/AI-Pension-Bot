"""STEP 2 - ask a language model (Gemini) to write one realistic user question per chunk.

Resumable: every question is saved as soon as it is written, and chunks already done are skipped on rerun.
Run:  python -m src.slm.data.generate_questions --limit 100 [--split train]
"""
from __future__ import annotations

import argparse
import json
import random

from src.rag.answer_generation import LLMProvider
from src.rag.chunking import read_chunks_file
from src.rag.config import Settings
from src.slm.data.common import PATHS, append_jsonl, done_ids, stable_id

QUESTION_SYSTEM = ("You write realistic questions that a UK member of the public might ask about pensions. "
                   "Given a passage, write ONE question that the passage answers. Do not copy sentences from the passage, "
                   "do not mention 'the passage' or 'the text', and do not include the answer. Reply with the question only.")


def clean_question(raw: str) -> str | None:
    q = " ".join(raw.strip().strip('"').split())
    if len(q) < 10 or len(q) > 300 or not q.endswith("?"):
        return None
    return q


def generate_questions(llm: LLMProvider, chunks_by_doc: dict[str, list[dict]], split: dict, limit: int, which: str = "train",
                       seed: int = 7, min_tokens: int = 60, out_path=None) -> dict:
    out_path = out_path or PATHS["questions"]
    allowed = set(split[which]) if which != "all" else set(split["train"]) | set(split["val"]) | set(split["test"])
    pools = {"train": set(split["train"]), "val": set(split["val"]), "test": set(split["test"])}
    done = done_ids(out_path, "qid")
    candidates = [c for d in sorted(allowed) for c in chunks_by_doc.get(d, []) if c["token_count"] >= min_tokens and "[TABLE]" not in c["text"]]
    random.Random(seed).shuffle(candidates)
    made = skipped = bad = 0
    for c in candidates:
        if made >= limit:
            break
        qid = stable_id("q", c["chunk_id"])
        if qid in done:
            continue
        try:
            q = clean_question(llm.generate(QUESTION_SYSTEM, f"Title: {c['title']}\n\nPassage:\n{c['text']}"))
        except Exception as exc:  # stop cleanly on API errors (e.g. daily cap); progress is already saved
            print("stopping:", exc)
            break
        if q is None:
            bad += 1
            continue
        split_name = next(k for k, v in pools.items() if c["document_id"] in v)
        append_jsonl(out_path, {"qid": qid, "split": split_name, "chunk_id": c["chunk_id"], "document_id": c["document_id"], "question": q})
        made += 1
    return {"made": made, "rejected_bad_format": bad, "already_done": len(done)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    args = ap.parse_args()
    from src.rag.gemini import GeminiLLM
    s = Settings.from_env()
    split = json.loads(PATHS["split"].read_text(encoding="utf-8"))
    print(generate_questions(GeminiLLM(), read_chunks_file(s.chunks_path), split, args.limit, args.split))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
