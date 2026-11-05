"""STEP 4 - draft the target answer for each example.

* answerable / no_support: the language model answers from the CLEAN evidence using the same SYSTEM_PROMPT the product uses
  (so training and production prompts match). For no_support examples it may legitimately refuse.
* off_topic: a refusal is written from templates (no model call needed).
Resumable; progress is saved after every example.

Run:  python -m src.slm.data.draft_answers --limit 100
"""
from __future__ import annotations

import argparse

from src.rag.answer_generation import INSUFFICIENT_TEXT, SYSTEM_PROMPT, LLMProvider, build_user_prompt
from src.rag.schema import Evidence
from src.slm.data.common import PATHS, append_jsonl, bucket, done_ids, read_jsonl

REFUSALS = [
    "The evidence provided does not contain the information needed to answer this question, so I can't answer it from these sources.",
    "I could not find this in the evidence provided. I don't want to guess, so please check GOV.UK or HMRC directly.",
    "These passages do not answer your question, so I can't give a reliable answer from them.",
    "The sources I was given don't cover this, so I can't answer it. Please check the official GOV.UK pages.",
]


def to_evidence(items: list[dict]) -> list[Evidence]:
    return [Evidence(rank=i, score=e.get("score", 0.0), chunk_id=e["chunk_id"], document_id=e["document_id"], title=e["title"],
                     source_url=e["source_url"], heading_path="", text=e["text"]) for i, e in enumerate(items, 1)]


def draft_answers(llm: LLMProvider, limit: int, inputs_path=None, out_path=None) -> dict:
    inputs_path, out_path = inputs_path or PATHS["inputs"], out_path or PATHS["drafts"]
    done = done_ids(out_path, "qid")
    made = 0
    for item in read_jsonl(inputs_path):
        if made >= limit:
            break
        if item["qid"] in done:
            continue
        if item["label"] == "off_topic":
            draft, source = REFUSALS[bucket(item["qid"]) % len(REFUSALS)], "template"
        else:
            try:
                draft = llm.generate(SYSTEM_PROMPT, build_user_prompt(item["question"], to_evidence(item["evidence_clean"]))).strip()
            except Exception as exc:
                print("stopping:", exc)
                break
            source = "llm"
        append_jsonl(out_path, {"qid": item["qid"], "draft": draft, "draft_source": source})
        made += 1
    return {"drafted": made, "already_done": len(done)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()
    from src.rag.gemini import GeminiLLM
    print(draft_answers(GeminiLLM(), args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
