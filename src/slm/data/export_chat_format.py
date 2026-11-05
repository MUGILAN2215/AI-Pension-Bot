"""STEP 7 - write train.jsonl / val.jsonl / test.jsonl (chat format) from APPROVED or EDITED examples only.

Safety checks: examples whose question matches an evaluation-set question are dropped (no test leakage).
Run:  python -m src.slm.data.export_chat_format
"""
from __future__ import annotations

import json
from collections import Counter

from src.rag.answer_generation import SYSTEM_PROMPT, build_user_prompt
from src.rag.config import Settings
from src.slm.data.common import PATHS, SLM_DIR, read_jsonl, write_jsonl
from src.slm.data.draft_answers import to_evidence


def _norm(q: str) -> str:
    return " ".join(q.lower().split())


def export(review_path=None, inputs_path=None, out_dir=None, eval_set=None) -> dict:
    out_dir = out_dir or SLM_DIR
    inputs = {i["qid"]: i for i in read_jsonl(inputs_path or PATHS["inputs"])}
    eval_questions = {_norm(q["question"]) for q in (eval_set or {}).get("questions", [])}
    outputs: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    stats, leaked = Counter(), 0
    for r in read_jsonl(review_path or PATHS["review"]):
        if r["status"] not in ("approved", "edited") or not r.get("final_answer"):
            stats["not_approved"] += 1
            continue
        if _norm(r["question"]) in eval_questions:
            leaked += 1
            continue
        item = inputs[r["qid"]]
        outputs[r["split"]].append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(r["question"], to_evidence(item["evidence"]))},
            {"role": "assistant", "content": r["final_answer"]}],
            "meta": {"qid": r["qid"], "label": r["label"], "variant": r["variant"]}})
        stats[f"{r['split']}:{r['label']}"] += 1
    for name, rows in outputs.items():
        write_jsonl(out_dir / f"{name}.jsonl", rows)
    return {"written": {k: len(v) for k, v in outputs.items()}, "dropped_eval_leakage": leaked, "breakdown": dict(stats)}


def main() -> int:
    s = Settings.from_env()
    es = json.loads(s.eval_set_path.read_text(encoding="utf-8")) if s.eval_set_path.exists() else None
    print(export(eval_set=es))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
