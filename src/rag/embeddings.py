"""Embedding providers behind one small interface.

Two providers are included:

* ``hashing`` (default): a deterministic, fully local, no-download baseline built on
  scikit-learn's HashingVectorizer (word unigrams + bigrams, sub-linear term frequency,
  L2-normalised). It matches *words*, not meaning, so it is a working baseline - not a
  substitute for a semantic model.
* ``gemini``: Google Gemini embeddings (needs GEMINI_API_KEY; see src/rag/gemini.py).
* ``sentence-transformers``: a real semantic model (default suggestion: BAAI/bge-small-en-v1.5).
  Needs `pip install sentence-transformers` and a one-off model download from Hugging Face.

Documents and queries go through the same `preprocess` step. Changing provider, model,
dimension or prefixes changes `fingerprint()`, which makes ingestion demand a re-index.
"""
from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Sequence

from src.rag.config import EmbeddingConfig

Vector = list[float]


class EmbeddingError(ValueError):
    """Raised when embedding fails or produces invalid vectors."""


def preprocess(text: str) -> str:
    """Shared preprocessing for documents and queries: collapse all whitespace."""
    return re.sub(r"\s+", " ", text or "").strip()


def validate_vectors(vectors: Sequence[Sequence[float]], expected_dim: int, expected_count: int | None = None) -> None:
    """Reject wrong counts, wrong dimensions, NaN/inf values and zero vectors."""
    if expected_count is not None and len(vectors) != expected_count:
        raise EmbeddingError(f"expected {expected_count} vectors, got {len(vectors)}")
    for i, vec in enumerate(vectors):
        if len(vec) != expected_dim:
            raise EmbeddingError(f"vector {i} has dimension {len(vec)}, expected {expected_dim}")
        if not all(math.isfinite(x) for x in vec):
            raise EmbeddingError(f"vector {i} contains NaN or infinite values")
        if not any(x != 0.0 for x in vec):
            raise EmbeddingError(f"vector {i} is all zeros (text has no embeddable content)")


class EmbeddingProvider(ABC):
    """Interface every embedding provider implements."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimension."""

    @abstractmethod
    def _embed(self, texts: list[str]) -> list[Vector]:
        """Embed already-preprocessed texts."""

    def _embed_kind(self, texts: list[str], kind: str) -> list[Vector]:
        """kind is "document" or "query". Most models ignore it; Gemini uses a different task type for each."""
        return self._embed(texts)

    def fingerprint(self) -> str:
        return self.config.fingerprint(self.dim)

    def embed_documents(self, texts: Sequence[str], batch_size: int | None = None) -> list[Vector]:
        """Embed documents in batches (bounded memory) and validate the result."""
        batch = batch_size or self.config.batch_size
        out: list[Vector] = []
        prepared = [self.config.document_prefix + preprocess(t) for t in texts]
        for i in range(0, len(prepared), batch):
            out.extend(self._embed_kind(prepared[i:i + batch], "document"))
        validate_vectors(out, self.dim, len(texts))
        return out

    def embed_query(self, text: str) -> Vector:
        if not preprocess(text):
            raise EmbeddingError("query is empty")
        vecs = self._embed_kind([self.config.query_prefix + preprocess(text)], "query")
        validate_vectors(vecs, self.dim, 1)
        return vecs[0]


class HashingEmbedder(EmbeddingProvider):
    """Deterministic lexical embeddings; no model download, no network."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vectorizer = HashingVectorizer(
            n_features=config.dim, analyzer="word", ngram_range=(1, 2), lowercase=True,
            token_pattern=r"(?u)\b\w+\b", stop_words="english",
            alternate_sign=False, norm=None, dtype="float64",
        )

    @property
    def dim(self) -> int:
        return self.config.dim

    def _embed(self, texts: list[str]) -> list[Vector]:
        import numpy as np
        from sklearn.preprocessing import normalize

        matrix = self._vectorizer.transform(texts)
        matrix.data = 1.0 + np.log(matrix.data)  # sub-linear term frequency
        return normalize(matrix, norm="l2").toarray().tolist()


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Semantic embeddings via sentence-transformers (lazy import; downloads the model once)."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError("sentence-transformers is not installed. Run: pip install sentence-transformers") from exc
        try:
            self._model = SentenceTransformer(config.model)
        except Exception as exc:  # network/model-name problems
            raise EmbeddingError(f"could not load model '{config.model}': {exc}") from exc
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def _embed(self, texts: list[str]) -> list[Vector]:
        arr = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in row] for row in arr]


def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Factory: pick the provider named in the configuration."""
    if config.provider == "hashing":
        return HashingEmbedder(config)
    if config.provider in ("sentence-transformers", "sentence_transformers"):
        return SentenceTransformerEmbedder(config)
    if config.provider == "gemini":
        from src.rag.gemini import GeminiEmbedder  # lazy: only needs google-genai when used

        return GeminiEmbedder(config)
    raise EmbeddingError(f"unknown embedding provider: {config.provider!r}")
