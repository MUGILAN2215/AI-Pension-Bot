"""Compare retrieval variants on the evaluation set and print one table (nothing is invented: every number is measured).

Run:  python -m scripts.compare_retrieval            (uses the configured index and embedder)
Writes data/eval/retrieval_comparison.json
"""
from __future__ import annotations

import json

from src.ingestion.validate_dataset import validate_jsonl
from src.rag.answer_generation import AnswerGenerator
from src.rag.config import Settings
from src.rag.embeddings import create_embedding_provider
from src.rag.evaluate import check_labels, evaluate
from src.rag.retriever import Retriever
from src.rag.vector_store import QdrantVectorStore

VARIANTS = {
    "dense": dict(mode="dense"),
    "bm25 (keywords)": dict(mode="bm25"),
    "hybrid": dict(mode="hybrid"),
    "dense + dedupe": dict(mode="dense", dedupe=True),
    "hybrid + dedupe": dict(mode="hybrid", dedupe=True),
    "hybrid + dedupe + max 2 per doc": dict(mode="hybrid", dedupe=True, max_per_document=2),
}


def main() -> int:
    s = Settings.from_env()
    es = json.loads(s.eval_set_path.read_text(encoding="utf-8"))
    problems = check_labels(es, validate_jsonl(s.documents_path).documents)
    if problems:
        print("Label problems:", problems)
        return 1
    embedder = create_embedding_provider(s.embedding)
    store = QdrantVectorStore(s.collection_name, path=str(s.qdrant_path), url=s.qdrant_url, api_key=s.qdrant_api_key)
    rows = {}
    try:
        for name, kw in VARIANTS.items():
            r = evaluate(es, Retriever(embedder, store, **kw), AnswerGenerator(min_score=s.min_score))
            rows[name] = {**r["metrics"], **{"abstained": r["abstention"]["abstained"], "of_unanswerable": r["abstention"]["unanswerable_questions"]}}
    finally:
        store.close()
    print(f"{'variant':34} {'S@1':>5} {'S@3':>5} {'S@5':>5} {'S@10':>5} {'MRR':>6} {'Ev@5':>5}")
    for n, m in rows.items():
        print(f"{n:34} {m['success@1']:5.2f} {m['success@3']:5.2f} {m['success@5']:5.2f} {m['success@10']:5.2f} {m['mrr']:6.3f} {m['evidence@5']:5.2f}")
    out = s.eval_set_path.parent / "retrieval_comparison.json"
    out.write_text(json.dumps({"embedding": s.embedding.model, "chunking": s.chunking.__dict__, "rows": rows}, indent=1))
    print("answerable questions:", next(iter(rows.values()))["answerable_questions"], "| saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
