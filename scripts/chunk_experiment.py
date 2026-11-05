"""Chunk-size experiment: build a temporary index for each chunk size and measure retrieval on the evaluation set.

Run:  python -m scripts.chunk_experiment            (takes a few minutes; uses temp folders, never touches the real index)
Writes data/eval/chunk_experiment.json
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

from src.ingestion.validate_dataset import validate_jsonl
from src.rag.answer_generation import AnswerGenerator
from src.rag.config import ChunkingConfig, Settings
from src.rag.embeddings import create_embedding_provider
from src.rag.evaluate import evaluate
from src.rag.ingest import ingest_documents
from src.rag.retriever import Retriever
from src.rag.vector_store import QdrantVectorStore

SIZES = [(250, 40), (350, 55), (500, 75)]  # (target tokens, overlap tokens)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, help="run only this target size (results are merged into the output file)")
    args = ap.parse_args(argv)
    base = Settings.from_env()
    out = base.eval_set_path.parent / "chunk_experiment.json"
    previous = {r["target_tokens"]: r for r in json.loads(out.read_text())["rows"]} if out.exists() else {}
    docs = validate_jsonl(base.documents_path).documents
    es = json.loads(base.eval_set_path.read_text(encoding="utf-8"))
    rows = []
    for target, overlap in [x for x in SIZES if args.size in (None, x[0])]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            s = dataclasses.replace(base, chunks_path=tmp / "c.jsonl", manifest_path=tmp / "m.json", qdrant_path=tmp / "q",
                                    collection_name="exp", chunking=ChunkingConfig(target_tokens=target, overlap_tokens=overlap))
            emb = create_embedding_provider(s.embedding)
            store = QdrantVectorStore(s.collection_name, path=str(s.qdrant_path))
            try:
                ingest_documents(docs, s, emb, store)
                n = store.count()
                r = evaluate(es, Retriever(emb, store, mode=s.retrieval_mode, dedupe=s.dedupe, max_per_document=s.max_per_document),
                             AnswerGenerator(min_score=s.min_score))
            finally:
                store.close()
        m = r["metrics"]
        previous[target] = ({"target_tokens": target, "overlap_tokens": overlap, "chunks": n, **{k: m[k] for k in ("success@1", "success@3", "success@5", "success@10", "mrr", "evidence@5")}})
        print(previous[target], flush=True)
    rows = [previous[k] for k in sorted(previous)]
    out.write_text(json.dumps({"embedding": base.embedding.model, "retrieval": base.retrieval_mode, "rows": rows}, indent=1))
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
