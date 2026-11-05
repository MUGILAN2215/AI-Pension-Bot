"""Retrieval and abstention evaluation against a small hand-checked question set.

Run:  python -m src.rag.evaluate                 (measure; writes data/eval/eval_results.json)
      python -m src.rag.evaluate --check-labels  (only verify the labels against the documents)

Metrics are computed ONLY from the labels in data/eval/eval_set.json:
* Success@k  - fraction of answerable questions with at least one relevant document in the top k
* Recall@k   - mean fraction of a question's relevant documents found in the top k
               (capped by k/R when a question has R > k relevant documents)
* MRR        - mean reciprocal rank of the first relevant document (0 if none in the top K)
* Evidence@k - fraction of questions where a top-k chunk literally contains the labelled evidence snippet
For unanswerable questions it reports the top similarity score and whether the answer layer abstained.
The set is tiny, so treat results as a smoke test, not a benchmark.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.ingestion.validate_dataset import validate_jsonl
from src.rag.answer_generation import AnswerGenerator
from src.rag.config import Settings
from src.rag.retriever import Retriever

KS = (1, 3, 5, 10)


def _norm(s: str) -> str:
    return " ".join(s.split())


def check_labels(eval_set: dict, documents) -> list[str]:
    """Return a list of label problems (empty means labels are consistent with the documents)."""
    texts = {d.document_id: _norm(d.text) for d in documents}
    problems: list[str] = []
    for q in eval_set["questions"]:
        if q["type"] == "answerable":
            snippet = _norm(q["evidence_snippet"])
            actual = {k for k, v in texts.items() if snippet in v}
            labelled = set(q["relevant_document_ids"])
            if not labelled:
                problems.append(f"{q['id']}: no relevant documents labelled")
            if labelled - actual:
                problems.append(f"{q['id']}: labelled docs without the snippet: {sorted(labelled - actual)[:3]}")
            if actual - labelled:
                problems.append(f"{q['id']}: documents containing the snippet but not labelled: {sorted(actual - labelled)[:3]}")
        else:
            for term in q.get("absent_terms", []):
                hits = [k for k, v in texts.items() if re.search(re.escape(term), v, re.I)]
                if hits:
                    problems.append(f"{q['id']}: absent term '{term}' occurs in {len(hits)} documents")
    return problems


def evaluate(eval_set: dict, retriever: Retriever, generator: AnswerGenerator, max_k: int = 10) -> dict:
    rows, un_rows = [], []
    for q in eval_set["questions"]:
        res = retriever.retrieve(q["question"], top_k=max_k)
        top_score = res.evidence[0].score if res.evidence else None
        if q["type"] == "answerable":
            rel = set(q["relevant_document_ids"])
            docs_ranked = [e.document_id for e in res.evidence]
            first = next((i for i, d in enumerate(docs_ranked, 1) if d in rel), None)
            snippet = _norm(q["evidence_snippet"]).lower()
            ans = generator.answer(q["question"], res)
            rows.append({"id": q["id"], "topic": q["topic"], "status": res.status, "n_relevant": len(rel), "first_relevant_rank": first,
                         "recall": {k: len(rel & set(docs_ranked[:k])) / len(rel) for k in KS},
                         "success": {k: first is not None and first <= k for k in KS},
                         "evidence_hit": {k: any(snippet in _norm(e.text).lower() for e in res.evidence[:k]) for k in KS},
                         "top_score": top_score, "answer_status": ans.status})
        else:
            ans = generator.answer(q["question"], res)
            un_rows.append({"id": q["id"], "top_score": top_score, "answer_status": ans.status, "citations_returned": len(ans.citations)})
    n = len(rows)
    agg = {
        "answerable_questions": n,
        **{f"success@{k}": sum(r["success"][k] for r in rows) / n for k in KS},
        **{f"recall@{k}": sum(r["recall"][k] for r in rows) / n for k in KS},
        **{f"evidence@{k}": sum(r["evidence_hit"][k] for r in rows) / n for k in KS},
        "mrr": sum((1 / r["first_relevant_rank"]) if r["first_relevant_rank"] else 0 for r in rows) / n,
    } if n else {}
    abst = {
        "unanswerable_questions": len(un_rows),
        "abstained": sum(r["answer_status"] == "insufficient_evidence" for r in un_rows),
        "answerable_wrongly_abstained": sum(r["answer_status"] == "insufficient_evidence" for r in rows),
        "min_score_threshold": generator.min_score,
    }
    return {"metrics": agg, "abstention": abst, "answerable": rows, "unanswerable": un_rows}


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-set", type=Path, default=settings.eval_set_path)
    ap.add_argument("--output", type=Path, default=settings.eval_results_path)
    ap.add_argument("--check-labels", action="store_true")
    args = ap.parse_args(argv)
    if not args.eval_set.exists():
        print(f"ERROR: evaluation set not found: {args.eval_set}")
        return 2
    eval_set = json.loads(args.eval_set.read_text(encoding="utf-8"))
    problems = check_labels(eval_set, validate_jsonl(settings.documents_path).documents)
    if problems:
        print("LABEL PROBLEMS - metrics not computed:")
        print("\n".join("  " + p for p in problems))
        return 1
    print(f"Labels consistent with documents ({len(eval_set['questions'])} questions).")
    if args.check_labels:
        return 0
    retriever = Retriever.from_settings(settings)
    try:
        out = evaluate(eval_set, retriever, AnswerGenerator(min_score=settings.min_score))
    finally:
        retriever.close()
    out["config"] = {"embedding": settings.embedding.model, "chunking": settings.chunking.__dict__, "collection": settings.collection_name}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["metrics"], indent=1))
    print("Abstention:", out["abstention"])
    for r in out["answerable"]:
        print(f"  {r['id']:20} first relevant rank: {r['first_relevant_rank']}  (R={r['n_relevant']}, top score {r['top_score']:.3f}, answer: {r['answer_status']})")
    for r in out["unanswerable"]:
        print(f"  {r['id']:20} top score {r['top_score']:.3f}  answer: {r['answer_status']}  citations: {r['citations_returned']}")
    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
