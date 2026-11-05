"""Demonstrate incremental ingestion safely.

Everything happens in a throw-away temporary folder with its own Qdrant database.
The real dataset, manifest and production collection are only READ (a few real documents
are used as background) and are never modified.

Run:  python -m scripts.demo_incremental
"""
from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

from src.ingestion.validate_dataset import validate_jsonl
from src.rag.config import Settings
from src.rag.embeddings import create_embedding_provider
from src.rag.ingest import ingest_documents
from src.rag.retriever import Retriever
from src.rag.schema import Document
from src.rag.vector_store import QdrantVectorStore


def main() -> int:
    real = Settings.from_env()
    background = validate_jsonl(real.documents_path).documents[:25]  # read-only sample
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        s = dataclasses.replace(real, chunks_path=tmp / "chunks.jsonl", manifest_path=tmp / "manifest.json",
                                qdrant_path=tmp / "qdrant", qdrant_url=None, collection_name="demo_temp_collection")
        embedder = create_embedding_provider(s.embedding)
        store = QdrantVectorStore(s.collection_name, path=str(s.qdrant_path))
        try:
            def run(label, docs, **kw):
                r = ingest_documents(docs, s, embedder, store, **kw)
                print(f"{label:<46} {r.summary()}  points={store.count()}")
                return r

            run("1. first ingest of 25 real docs (temp index)", background)
            run("2. rerun, nothing changed", background)
            test_doc = Document(document_id="TEMP-DEMO-1", title="TEMP DEMO: test document (not real guidance)",
                                source_url="https://example.invalid/temp-demo", licence="n/a - test data",
                                text="# Temp demo\n\nThe zebra marmalade allowance is a made-up test fact used only to demonstrate ingestion.")
            run("3. add one temporary test document", background + [test_doc])
            r = Retriever(embedder, store, top_k=1).retrieve("zebra marmalade allowance")
            print(f"   retrieved: {r.evidence[0].document_id} - {r.evidence[0].title}")
            run("4. rerun again (still no duplicates)", background + [test_doc])
            edited = dataclasses.replace(test_doc, text=test_doc.text + "\n\nUpdated test sentence about a quokka bursary.")
            run("5. update the test document", background + [edited])
            r = Retriever(embedder, store, top_k=1).retrieve("quokka bursary")
            print(f"   retrieved after update: {r.evidence[0].document_id}")
            run("6. partial input (only the test doc): nothing deleted", [edited])
            print("   real documents still present:", len(store.chunk_counts_by_document()) - 1 == len(background))
        finally:
            store.close()
    print("Temporary index deleted. The real collection was not touched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
