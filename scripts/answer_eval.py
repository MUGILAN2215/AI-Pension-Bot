"""ANSWER evaluation: does a language model write good, cited, honest answers from our retrieved passages?

Retrieval evaluation (src/rag/evaluate.py) asks "did search find the right page?". This asks "did the writer use the page properly?".
Run it once per model (for example base model vs fine-tuned model) and compare the numbers.

    python -m scripts.answer_eval --provider gemini --tag gemini
    OLLAMA_MODEL=llama-small python -m scripts.answer_eval --provider ollama --tag base-slm
    OLLAMA_MODEL=pension-slm  python -m scripts.answer_eval --provider ollama --tag tuned-slm

The safety nets (number check, relevance judge) are switched OFF here so you see the model's own behaviour.
All metrics are automatic proxies. Read data/eval/answer_eval_<tag>.json yourself for a real judgement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rag.answer_generation import _CITE_RE, _NUMBER_RE, AnswerGenerator, LLMProvider, is_refusal, ungrounded_numbers
from src.rag.config import Settings
from src.rag.retriever import Retriever


def _key_numbers_present(snippet: str, answer: str) -> bool | None:
    """True if every figure in the labelled answer sentence (e.g. "10", "£12,570") also appears in the reply. None if the sentence has no figures."""
    wanted = {t.replace(",", "") for t in _NUMBER_RE.findall(snippet)}
    wanted = {w for w in wanted if len(w.strip("£$%")) >= 2 or w.startswith(("£", "$")) or w.endswith("%")}
    if not wanted:
        return None
    got = answer.replace(",", "")
    return all(w in got or w.strip("£$%") in got for w in wanted)


def evaluate_answers(llm: LLMProvider | None, retriever: Retriever, eval_set: dict, min_score: float, top_k: int = 5, limit: int | None = None) -> dict:
    gen = AnswerGenerator(llm, min_score=min_score, check_numbers=False)  # nets off: measure the model itself
    answerable, unanswerable = [], []
    for q in eval_set["questions"]:
        bucket = answerable if q["type"] == "answerable" else unanswerable
        if limit and len(bucket) >= limit:
            continue
        res = retriever.retrieve(q["question"], top_k=top_k)
        ans = gen.answer(q["question"], res)
        shown = [e for e in res.evidence if e.score >= min_score][:5]  # what the writer saw
        text = ans.text
        refused = ans.status == "insufficient_evidence" or is_refusal(text)
        row = {"id": q["id"], "question": q["question"], "status": ans.status, "mode": ans.mode, "refused": refused, "answer": text}
        if q["type"] == "answerable":
            rel = set(q["relevant_document_ids"])
            cited = [c for c in ans.citations]
            # If the generator rejected the model's reply (no/invalid citations) it shows the extractive fallback instead.
            # That fallback text has citations of its own, so count citations ONLY for replies the model really wrote.
            written_by_model = ans.mode == "generated"
            markers = {int(m) for m in _CITE_RE.findall(text)} if written_by_model else set()
            valid = {c.marker for c in cited}
            row.update(
                has_citation=bool(markers),
                citations_valid=bool(markers) and markers <= valid,
                cites_right_document=any(c.document_id in rel for c in cited if c.marker in markers),
                invented_numbers=ungrounded_numbers(text, shown) if written_by_model else [],
                model_reply_rejected=ans.status == "ungrounded",
                key_numbers_present=_key_numbers_present(q["evidence_snippet"], text) if written_by_model else False,
                evidence_had_answer=any(" ".join(q["evidence_snippet"].split()).lower() in " ".join(e.text.split()).lower() for e in shown),
            )
            answerable.append(row)
        else:
            unanswerable.append(row)

    def rate(rows, f):
        return round(sum(1 for r in rows if f(r)) / len(rows), 3) if rows else None

    # Only judge the writer on questions where the passages really contained the answer.
    fair = [r for r in answerable if r["evidence_had_answer"]]
    return {
        "counts": {"answerable": len(answerable), "answerable_where_evidence_had_the_answer": len(fair), "unanswerable": len(unanswerable)},
        "metrics_on_fair_questions": {
            "answered_not_refused": rate(fair, lambda r: not r["refused"]),
            "reply_rejected_for_bad_citations": rate(fair, lambda r: r["model_reply_rejected"]),
            "has_citation": rate(fair, lambda r: r["has_citation"] and not r["refused"]),
            "citations_valid": rate(fair, lambda r: r["citations_valid"] and not r["refused"]),
            "cites_right_document": rate(fair, lambda r: r["cites_right_document"] and not r["refused"]),
            "key_numbers_present": rate([r for r in fair if r["key_numbers_present"] is not None], lambda r: r["key_numbers_present"] and not r["refused"]),
            "no_invented_numbers": rate(fair, lambda r: not r["refused"] and r["has_citation"] and not r["invented_numbers"]),
        },
        "refusal_on_unanswerable": {"refused": sum(r["refused"] for r in unanswerable), "of": len(unanswerable)},
        "answerable": answerable, "unanswerable": unanswerable,
    }


def write_review_sheet(out: dict, path: Path, n: int = 20) -> None:
    """A markdown sheet for YOU to judge answers by reading them: automatic metrics cannot tell if a sentence is really supported."""
    rows = out["answerable"][: n - n // 4] + out["unanswerable"][: n // 4]
    lines = ["# Answer review sheet", "", "For each answer mark: **correct** / **partly** / **wrong**. Check every fact against the cited source.", ""]
    for i, r in enumerate(rows, 1):
        lines += [f"## {i}. {r['question']}", f"*Expected:* {'a cited answer' if 'has_citation' in r else 'a refusal'}   |   *System status:* {r['status']}", "",
                  r["answer"], "", "Verdict: [ ] correct  [ ] partly  [ ] wrong      Notes: ", "", "---", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def compare(a: dict, b: dict, name_a: str, name_b: str) -> str:
    keys = sorted(set(a["metrics_on_fair_questions"]) | set(b["metrics_on_fair_questions"]))
    out = [f"{'metric':36}{name_a:>14}{name_b:>14}"]
    for k in keys:
        out.append(f"{k:36}{str(a['metrics_on_fair_questions'].get(k)):>14}{str(b['metrics_on_fair_questions'].get(k)):>14}")
    ra, rb = a["refusal_on_unanswerable"], b["refusal_on_unanswerable"]
    out.append(f"{'refused unanswerable':36}{ra['refused']}/{ra['of']:>11}{rb['refused']}/{rb['of']:>11}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["none", "gemini", "ollama"])
    ap.add_argument("--tag", help="name for the output file, e.g. gemini or base-slm")
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"), help="print two earlier runs side by side (no model calls)")
    ap.add_argument("--review-sheet", type=int, default=20, help="how many answers to put in the markdown review sheet")
    ap.add_argument("--limit", type=int, help="only the first N answerable and N unanswerable questions (quick trial)")
    args = ap.parse_args(argv)
    s = Settings.from_env()
    folder = Path(s.eval_set_path).parent
    if args.compare:
        a, b = (json.loads((folder / f"answer_eval_{t}.json").read_text(encoding="utf-8")) for t in args.compare)
        print(compare(a, b, *args.compare))
        return 0
    if not (args.provider and args.tag):
        ap.error("--provider and --tag are required unless you use --compare")
    from src.api.service import make_llm
    es = json.loads(s.eval_set_path.read_text(encoding="utf-8"))
    retriever = Retriever.from_settings(s)
    try:
        out = evaluate_answers(make_llm(args.provider), retriever, es, s.min_score, s.top_k, args.limit)
    finally:
        retriever.close()
    path = folder / f"answer_eval_{args.tag}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    write_review_sheet(out, folder / f"answer_review_{args.tag}.md", args.review_sheet)
    print(json.dumps({k: out[k] for k in ("counts", "metrics_on_fair_questions", "refusal_on_unanswerable")}, indent=1))
    print("full answers saved to", path, "| review sheet:", folder / f"answer_review_{args.tag}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
