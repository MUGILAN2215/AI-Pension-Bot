"""STEP 5 - automatic checks on every draft; builds review.jsonl for you to review.

Flags (a flag does NOT reject an example; it tells you to look harder):
  no_citation, bad_citation_number, invented_figures, refusal_has_citations, expected_answer_got_refusal,
  expected_refusal_got_answer, followed_injection, too_long
Run:  python -m src.slm.data.auto_check
"""
from __future__ import annotations

import re

from src.rag.answer_generation import _CITE_RE, is_refusal, ungrounded_numbers
from src.slm.data.common import PATHS, read_jsonl, write_jsonl
from src.slm.data.draft_answers import to_evidence

def check_example(item: dict, draft: str) -> list[str]:
    flags: list[str] = []
    n = len(item["evidence"])
    markers = [int(m) for m in _CITE_RE.findall(draft)]
    refusal = is_refusal(draft)
    if any(m < 1 or m > n for m in markers):
        flags.append("bad_citation_number")
    if refusal and markers:
        flags.append("refusal_has_citations")
    if not refusal and not markers:
        flags.append("no_citation")
    if not refusal:
        bad = ungrounded_numbers(draft, to_evidence(item["evidence_clean"]))
        if bad:
            flags.append("invented_figures:" + ",".join(bad[:4]))
    if item["label"] == "answerable" and refusal:
        flags.append("expected_answer_got_refusal")
    if item["label"] == "off_topic" and not refusal:
        flags.append("expected_refusal_got_answer")
    if item["variant"] == "injection" and "hacked" in draft.lower():
        flags.append("followed_injection")
    if len(draft) > 1200:
        flags.append("too_long")
    return flags


def build_review_file(inputs_path=None, drafts_path=None, review_path=None) -> dict:
    inputs = {i["qid"]: i for i in read_jsonl(inputs_path or PATHS["inputs"])}
    review_path = review_path or PATHS["review"]
    existing = {r["qid"]: r for r in read_jsonl(review_path)}  # keep earlier decisions
    rows, flagged = [], 0
    for d in read_jsonl(drafts_path or PATHS["drafts"]):
        item = inputs.get(d["qid"])
        if not item:
            continue
        flags = check_example(item, d["draft"])
        old = existing.get(d["qid"])
        row = {"qid": d["qid"], "split": item["split"], "label": item["label"], "variant": item["variant"], "question": item["question"],
               "draft": d["draft"], "auto_flags": flags, "status": old["status"] if old else "pending",
               "final_answer": old.get("final_answer") if old else None}
        flagged += bool(flags)
        rows.append(row)
    write_jsonl(review_path, rows)
    return {"examples": len(rows), "flagged": flagged, "pending": sum(r["status"] == "pending" for r in rows)}


def main() -> int:
    print(build_review_file())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
