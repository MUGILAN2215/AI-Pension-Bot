"""Central configuration for the RAG module.

Every tunable setting lives here. Values come from environment variables
(optionally loaded from a local `.env` file) and fall back to safe defaults.
Nothing else in the code base should read `os.environ` directly.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

try:  # python-dotenv is optional; the code works without it.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Bump these when the algorithm changes in a way that alters output.
CHUNKER_VERSION = "1"
EMBEDDING_PREPROCESS_VERSION = "1"


@dataclass(frozen=True)
class ChunkingConfig:
    """Settings that control how documents are split into chunks."""

    target_tokens: int = 500
    overlap_tokens: int = 75
    version: str = CHUNKER_VERSION

    def __post_init__(self) -> None:
        if self.target_tokens < 50:
            raise ValueError("target_tokens must be at least 50")
        if not 0 <= self.overlap_tokens < self.target_tokens // 2:
            raise ValueError("overlap_tokens must be >= 0 and less than half of target_tokens")

    def fingerprint(self) -> str:
        """Short hash; a change means every document must be re-chunked and re-embedded."""
        return _short_hash({"target": self.target_tokens, "overlap": self.overlap_tokens, "v": self.version})


@dataclass(frozen=True)
class EmbeddingConfig:
    """Settings that identify the embedding model. A change requires re-embedding."""

    provider: str = "hashing"  # "hashing" (no download) or "sentence-transformers"
    model: str = "hashing-word-1-2gram"  # for sentence-transformers: a model name
    dim: int = 2048  # used by the hashing provider; sentence-transformers reports its own
    batch_size: int = 64
    query_prefix: str = ""
    document_prefix: str = ""

    def fingerprint(self, actual_dim: int | None = None) -> str:
        return _short_hash(
            {
                "provider": self.provider,
                "model": self.model,
                "dim": actual_dim or self.dim,
                "qp": self.query_prefix,
                "dp": self.document_prefix,
                "v": EMBEDDING_PREPROCESS_VERSION,
            }
        )


@dataclass(frozen=True)
class Settings:
    """All settings for the RAG module."""

    documents_path: Path
    chunks_path: Path
    manifest_path: Path
    validation_report_path: Path
    validation_review_path: Path
    eval_set_path: Path
    eval_results_path: Path
    qdrant_path: Path
    qdrant_url: str | None
    qdrant_api_key: str | None
    collection_name: str
    top_k: int
    min_score: float
    retrieval_mode: str = "hybrid"
    dedupe: bool = True
    max_per_document: int | None = 2
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        """Build settings from environment variables, then apply keyword overrides."""
        if load_dotenv is not None:
            load_dotenv(PROJECT_ROOT / ".env", override=False)
        env = os.environ.get

        def path(name: str, default: str) -> Path:
            p = Path(env(name, default))
            return p if p.is_absolute() else PROJECT_ROOT / p

        settings = cls(
            documents_path=path("RAG_DOCUMENTS_PATH", "data/processed/documents_clean.jsonl"),
            chunks_path=path("RAG_CHUNKS_PATH", "data/processed/chunks.jsonl"),
            manifest_path=path("RAG_MANIFEST_PATH", "data/processed/ingestion_manifest.json"),
            validation_report_path=path("RAG_VALIDATION_REPORT_PATH", "data/processed/validation_report.json"),
            validation_review_path=path("RAG_VALIDATION_REVIEW_PATH", "data/processed/validation_review.jsonl"),
            eval_set_path=path("RAG_EVAL_SET_PATH", "data/eval/eval_set.json"),
            eval_results_path=path("RAG_EVAL_RESULTS_PATH", "data/eval/eval_results.json"),
            qdrant_path=path("RAG_QDRANT_PATH", "data/qdrant"),
            qdrant_url=env("RAG_QDRANT_URL") or None,
            qdrant_api_key=env("RAG_QDRANT_API_KEY") or None,
            collection_name=env("RAG_COLLECTION", "uk_pension_chunks"),
            top_k=int(env("RAG_TOP_K", "5")),
            min_score=float(env("RAG_MIN_SCORE", "0.2")),
            retrieval_mode=env("RAG_RETRIEVAL_MODE", "hybrid"),
            dedupe=env("RAG_DEDUPE", "true").lower() == "true",
            max_per_document=int(env("RAG_MAX_PER_DOC", "2")) or None,
            chunking=ChunkingConfig(
                target_tokens=int(env("RAG_CHUNK_TARGET_TOKENS", "500")),
                overlap_tokens=int(env("RAG_CHUNK_OVERLAP_TOKENS", "75")),
            ),
            embedding=EmbeddingConfig(
                provider=env("RAG_EMBEDDING_PROVIDER", "hashing"),
                model=env("RAG_EMBEDDING_MODEL", "hashing-word-1-2gram"),
                dim=int(env("RAG_EMBEDDING_DIM", "2048")),
                batch_size=int(env("RAG_EMBED_BATCH_SIZE", "64")),
                query_prefix=env("RAG_QUERY_PREFIX", ""),
                document_prefix=env("RAG_DOCUMENT_PREFIX", ""),
            ),
        )
        return replace(settings, **overrides) if overrides else settings


def _short_hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]
