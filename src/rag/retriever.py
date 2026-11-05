"""Retrieve evidence chunks for a question.

Usable as a Python module:

    retriever = Retriever.from_settings(Settings.from_env())
    result = retriever.retrieve("How do I claim Pension Credit?", top_k=5)
    for ev in result.evidence: print(ev.title, ev.source_url, ev.score)

or from the command line:  python -m src.rag.retriever "your question"

Scores are vector similarities. They indicate how close a chunk is to the question in
embedding space; they do NOT measure whether the chunk is factually correct or complete.
"""
from __future__ import annotations

import argparse
import json
from abc import ABC, abstractmethod
from typing import Any

from src.rag.bm25 import BM25Index
from src.rag.config import Settings
from src.rag.embeddings import EmbeddingError, EmbeddingProvider, create_embedding_provider
from src.rag.schema import Evidence, RetrievalResult
from src.rag.vector_store import CollectionMissingError, QdrantVectorStore, VectorStore

# Payload keys copied into Evidence.metadata (everything needed for citation/filtering).
_METADATA_KEYS = ("parent_source_url", "section", "publisher", "source", "source_type", "category",
                  "original_format", "jurisdiction", "language", "publication_date", "last_updated_date",
                  "date", "licence", "chunk_index", "chunk_count", "content_hash", "document_hash",
                  "review_flags", "provenance")


class Reranker(ABC):
    """Optional second-stage ordering of retrieved evidence. Not required."""

    @abstractmethod
    def rerank(self, query: str, evidence: list[Evidence]) -> list[Evidence]: ...


class Retriever:
    def __init__(self, embedder: EmbeddingProvider, store: VectorStore, top_k: int = 5,
                 reranker: Reranker | None = None, candidate_multiplier: int = 4,
                 mode: str = "dense", dedupe: bool = False, max_per_document: int | None = None,
                 candidate_pool: int = 30, rrf_k: int = 60):
        """mode: "dense" (vectors), "bm25" (keywords) or "hybrid" (both, fused with reciprocal rank fusion).
        dedupe: collapse chunks with identical text (many HMRC pages repeat the same sentences).
        max_per_document: cap how many chunks of one document may appear (more variety in the top k)."""
        if mode not in ("dense", "bm25", "hybrid"):
            raise ValueError("mode must be dense, bm25 or hybrid")
        self.embedder = embedder
        self.store = store
        self.default_top_k = top_k
        self.reranker = reranker
        self.candidate_multiplier = candidate_multiplier
        self.mode, self.dedupe, self.max_per_document = mode, dedupe, max_per_document
        self.candidate_pool, self.rrf_k = candidate_pool, rrf_k
        self._bm25: BM25Index | None = None

    def refresh_index(self) -> None:
        """Forget the in-memory keyword index (call after documents are added or changed)."""
        self._bm25 = None

    def _keyword_index(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(self.store.iter_payloads())
        return self._bm25

    def _candidates(self, question: str, vector, fetch: int, filters):
        """Return [(payload, cosine_or_None, info)] best first, according to the retrieval mode."""
        pool = max(fetch, self.candidate_pool)
        dense = self.store.search(vector, pool, filters) if self.mode in ("dense", "hybrid") else []
        if self.mode == "dense":
            return [(p, sc, {"dense_rank": i}) for i, (p, sc) in enumerate(dense, 1)]
        kw = self._keyword_index().search(question, pool, filters)
        if self.mode == "bm25":
            cands = [(p, None, {"bm25_rank": i, "bm25_score": sc}) for i, (p, sc) in enumerate(kw, 1)]
        else:  # hybrid: reciprocal rank fusion
            table: dict[str, dict] = {}
            for i, (p, sc) in enumerate(dense, 1):
                table.setdefault(p["chunk_id"], {"p": p, "cos": sc, "info": {}})["info"]["dense_rank"] = i
            for i, (p, sc) in enumerate(kw, 1):
                e = table.setdefault(p["chunk_id"], {"p": p, "cos": None, "info": {}})
                e["info"]["bm25_rank"], e["info"]["bm25_score"] = i, sc
            for e in table.values():
                e["info"]["rrf"] = sum(1.0 / (self.rrf_k + e["info"][r]) for r in ("dense_rank", "bm25_rank") if r in e["info"])
            cands = [(e["p"], e["cos"], e["info"]) for e in sorted(table.values(), key=lambda e: -e["info"]["rrf"])]
        # chunks found only by keywords have no cosine yet; compute it so the score is comparable
        need = [c[0]["chunk_id"] for c in cands if c[1] is None]
        if need:
            vecs = self.store.get_vectors(need)
            q = list(vector)
            cands = [(p, cos if cos is not None else (sum(a * b for a, b in zip(q, vecs[p["chunk_id"]])) if p["chunk_id"] in vecs else 0.0), info)
                     for p, cos, info in cands]
        return cands

    def _tidy(self, cands, k: int):
        """Collapse identical chunk text and cap chunks per document, keeping the original order."""
        out, seen, per_doc = [], {}, {}
        for p, sc, info in cands:
            if self.dedupe:
                h = p.get("content_hash")
                if h in seen:
                    seen[h]["info"]["duplicates_collapsed"] = seen[h]["info"].get("duplicates_collapsed", 0) + 1
                    seen[h]["info"].setdefault("duplicate_urls", []).append(p.get("source_url"))
                    continue
                seen[h] = {"info": info}
            if self.max_per_document and per_doc.get(p.get("document_id"), 0) >= self.max_per_document:
                continue
            per_doc[p.get("document_id")] = per_doc.get(p.get("document_id"), 0) + 1
            out.append((p, sc, info))
        return out

    @classmethod
    def from_settings(cls, settings: Settings, reranker: Reranker | None = None) -> "Retriever":
        store = QdrantVectorStore(settings.collection_name, path=str(settings.qdrant_path),
                                  url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        return cls(create_embedding_provider(settings.embedding), store, settings.top_k, reranker, mode=settings.retrieval_mode,
                   dedupe=settings.dedupe, max_per_document=settings.max_per_document)

    def retrieve(self, question: str, top_k: int | None = None, filters: dict[str, Any] | None = None) -> RetrievalResult:
        """Return evidence for `question`. Never raises for expected problems; check `result.status`."""
        k = top_k or self.default_top_k
        if not question or not question.strip():
            return RetrievalResult(question or "", "empty_query", message="The question is empty.", top_k=k, filters=filters)
        try:
            vector = self.embedder.embed_query(question)
        except EmbeddingError as exc:
            return RetrievalResult(question, "empty_query", message=f"The question has no searchable content: {exc}", top_k=k, filters=filters)
        except Exception as exc:
            return RetrievalResult(question, "error", message=f"Embedding failed: {exc}", top_k=k, filters=filters)
        try:
            fetch = k * self.candidate_multiplier if self.reranker else k
            cands = self._tidy(self._candidates(question, vector, fetch, filters), k)[:fetch]
        except CollectionMissingError as exc:
            return RetrievalResult(question, "collection_missing", message=str(exc), top_k=k, filters=filters)
        except Exception as exc:
            return RetrievalResult(question, "error", message=f"Vector search failed: {type(exc).__name__}: {exc}", top_k=k, filters=filters)
        if not cands:
            return RetrievalResult(question, "no_results", message="No matching passages were found.", top_k=k, filters=filters)
        evidence = [Evidence(rank=i, score=score, chunk_id=p.get("chunk_id", ""), document_id=p.get("document_id", ""),
                             title=p.get("title", ""), source_url=p.get("source_url", ""), heading_path=p.get("heading_path", ""),
                             text=p.get("text", ""), metadata={**{m: p.get(m) for m in _METADATA_KEYS}, "retrieval": info})
                    for i, (p, score, info) in enumerate(cands, 1)]
        if self.reranker:
            try:
                evidence = self.reranker.rerank(question, evidence)
            except Exception as exc:
                return RetrievalResult(question, "error", message=f"Reranking failed: {exc}", top_k=k, filters=filters)
            evidence = evidence[:k]
            for i, ev in enumerate(evidence, 1):
                ev.rank = i
        return RetrievalResult(question, "ok", evidence=evidence, top_k=k, filters=filters)

    def close(self) -> None:
        self.store.close()


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    ap = argparse.ArgumentParser(description="Ask the index a question and print the retrieved evidence.")
    ap.add_argument("question")
    ap.add_argument("--top-k", type=int, default=settings.top_k)
    ap.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE", help="e.g. --filter publisher=GOV.UK")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--chars", type=int, default=400, help="characters of each passage to print")
    args = ap.parse_args(argv)
    filters = dict(f.split("=", 1) for f in args.filter) or None
    retriever = Retriever.from_settings(settings)
    try:
        result = retriever.retrieve(args.question, args.top_k, filters)
    finally:
        retriever.close()
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0 if result.ok else 1
    print(f"Question: {result.query}\nStatus: {result.status}" + (f" - {result.message}" if result.message else ""))
    for ev in result.evidence:
        print(f"\n[{ev.rank}] score={ev.score:.3f}  {ev.title}" + (f"  >  {ev.heading_path}" if ev.heading_path else ""))
        print(f"    {ev.source_url}\n    chunk: {ev.chunk_id}  licence: {ev.metadata.get('licence')}  updated: {ev.metadata.get('last_updated_date') or ev.metadata.get('date')}")
        text = ev.text.replace("\n", " ")
        print("    " + text[:args.chars] + ("..." if len(text) > args.chars else ""))
    if result.evidence:
        print("\n(Scores are vector similarities, not measures of factual correctness.)")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
