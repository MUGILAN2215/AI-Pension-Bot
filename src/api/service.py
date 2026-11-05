"""The RAG service used by the API: builds the retriever, answer generator and agent once, and serialises access to Qdrant.

LLM_PROVIDER (env):  none (default; extractive answers, no model)  |  gemini  |  ollama
Local Qdrant allows one process, so the API must run a single worker (or use a Qdrant server via RAG_QDRANT_URL).
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

from src.agent.graph import AgentResult, PensionAgent
from src.agent.tools import DocumentLibrary
from src.rag.answer_generation import AnswerGenerator, LLMProvider
from src.rag.config import PROJECT_ROOT, Settings
from src.rag.embeddings import create_embedding_provider
from src.rag.ingest import build_store, ingest_documents
from src.rag.retriever import Retriever
from src.rag.schema import Document


def short_date(value: str | None) -> str | None:
    """2026-09-09T14:55:07+01:00 -> 2026-09-09 (sources mix date formats; show one)."""
    m = re.match(r"\d{4}-\d{2}-\d{2}", value or "")
    return m.group(0) if m else value


def make_generator(llm: LLMProvider | None, min_score: float, use_judge: bool) -> AnswerGenerator:
    """use_judge adds one extra model call per question: "do these passages answer the question?" (NO -> refuse)."""
    return AnswerGenerator(llm, min_score=min_score, judge=llm if (use_judge and llm is not None) else None)


def make_reranker(llm: LLMProvider | None, mode: str):
    """mode "llm" reorders the top candidates with the model (one extra call per question); anything else = off."""
    if mode == "llm" and llm is not None:
        from src.rag.gemini import LLMReranker
        return LLMReranker(llm)
    return None


def source_label(meta: dict) -> dict:
    if meta.get("source_type") == "admin_upload" or meta.get("source") == "admin_upload":
        return {"label": "Admin-uploaded document", "admin_uploaded": True}
    return {"label": meta.get("publisher") or {"gov.uk": "GOV.UK", "hmrc": "HMRC"}.get(meta.get("source") or "", "Source not stated"), "admin_uploaded": False}


def make_llm(provider: str) -> LLMProvider | None:
    provider = (provider or "none").lower()
    if provider == "none":
        return None
    if provider == "gemini":
        from src.rag.gemini import GeminiLLM
        return GeminiLLM()
    if provider == "ollama":
        from src.slm.ollama_llm import OllamaLLM
        return OllamaLLM()
    raise ValueError(f"unknown LLM_PROVIDER {provider!r} (use none, gemini or ollama)")


class RagService:
    def __init__(self, settings: Settings, embedder, store, retriever: Retriever, generator: AnswerGenerator, agent: PensionAgent,
                 uploads_path: Path, llm_provider: str = "none"):
        self.settings, self.embedder, self.store, self.retriever = settings, embedder, store, retriever
        self.generator, self.agent, self.uploads_path, self.llm_provider = generator, agent, uploads_path, llm_provider
        self.lock = threading.RLock()

    @classmethod
    def from_env(cls) -> "RagService":
        settings = Settings.from_env()
        provider = os.environ.get("LLM_PROVIDER", "none")
        llm = make_llm(provider)
        embedder = create_embedding_provider(settings.embedding)
        store = build_store(settings)
        retriever = Retriever(embedder, store, settings.top_k, mode=settings.retrieval_mode, dedupe=settings.dedupe, max_per_document=settings.max_per_document,
                              reranker=make_reranker(llm, os.environ.get("RAG_RERANK", "off").lower()))
        generator = make_generator(llm, settings.min_score, os.environ.get("RAG_USE_JUDGE", "false").lower() == "true")
        agent = PensionAgent(retriever, generator, rewriter=llm, library=DocumentLibrary(settings.documents_path))
        return cls(settings, embedder, store, retriever, generator, agent, PROJECT_ROOT / "data/processed/documents_uploads.jsonl", provider)

    def ask(self, question: str, history: list[dict] | None = None) -> AgentResult:
        with self.lock:
            return self.agent.ask(question, history)

    def get_chunk(self, chunk_id: str) -> dict | None:
        doc_id = chunk_id.split("#")[0]
        with self.lock:
            for c in self.store.get_document_chunks(doc_id):
                if c.get("chunk_id") == chunk_id:
                    return {"chunk_id": chunk_id, "document_id": doc_id, "title": c.get("title"), "source_url": c.get("source_url"),
                            "heading_path": c.get("heading_path"), "text": c.get("text"), "licence": c.get("licence"),
                            "publisher": c.get("publisher"), "last_updated": short_date(c.get("last_updated_date") or c.get("date")), **source_label(c)}
        return None

    def ingest_upload(self, doc: Document) -> dict:
        with self.lock:
            report = ingest_documents([doc], self.settings, self.embedder, self.store)
            if report.failed:
                return {"status": "failed", "error": report.failed[0]["error"], "document_id": doc.document_id}
            self.uploads_path.parent.mkdir(parents=True, exist_ok=True)
            with self.uploads_path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps({"doc_id": doc.document_id, **{k: v for k, v in doc.to_dict().items() if k not in ("document_id", "metadata")}}, ensure_ascii=False) + "\n")
            self.retriever.refresh_index()
            chunks = len(self.store.get_document_chunks(doc.document_id))
            return {"status": "skipped_unchanged" if report.skipped else "indexed", "document_id": doc.document_id, "chunks": chunks}

    def health(self) -> dict:
        with self.lock:
            try:
                points = self.store.count()
                db_ok = True
            except Exception as exc:  # report, don't crash the health check
                points, db_ok = 0, False
                err = str(exc)
        out = {"vector_store": {"ok": db_ok, "collection": self.settings.collection_name, "chunks": points},
               "embedding": {"provider": self.settings.embedding.provider, "dim": self.embedder.dim},
               "retrieval_mode": self.settings.retrieval_mode, "llm_provider": self.llm_provider,
               "note": "llm_provider is configuration only; no model call is made by this check"}
        if not db_ok:
            out["vector_store"]["error"] = err
        return out
