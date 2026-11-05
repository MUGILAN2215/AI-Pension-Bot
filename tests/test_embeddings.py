import pytest

from src.rag.config import EmbeddingConfig
from src.rag.embeddings import EmbeddingError, EmbeddingProvider, HashingEmbedder, create_embedding_provider, validate_vectors


def test_dimensions_and_normalisation(embedder):  # 11
    vecs = embedder.embed_documents(["State Pension age", "Pension Credit for low income"])
    assert len(vecs) == 2 and all(len(v) == embedder.dim == 256 for v in vecs)
    assert abs(sum(x * x for x in vecs[0]) - 1.0) < 1e-6


def test_invalid_vectors_rejected():  # 11
    with pytest.raises(EmbeddingError, match="dimension"):
        validate_vectors([[1.0, 2.0]], expected_dim=3)
    with pytest.raises(EmbeddingError, match="NaN"):
        validate_vectors([[float("nan"), 1.0]], expected_dim=2)
    with pytest.raises(EmbeddingError, match="infinite"):
        validate_vectors([[float("inf"), 1.0]], expected_dim=2)
    with pytest.raises(EmbeddingError, match="zeros"):
        validate_vectors([[0.0, 0.0]], expected_dim=2)
    with pytest.raises(EmbeddingError, match="expected 2 vectors"):
        validate_vectors([[1.0, 0.0]], expected_dim=2, expected_count=2)


def test_stopword_only_text_is_rejected(embedder):
    with pytest.raises(EmbeddingError):
        embedder.embed_documents(["the of and a"])
    with pytest.raises(EmbeddingError):
        embedder.embed_query("   ")


def test_query_and_document_use_same_preprocessing(embedder):
    assert embedder.embed_query("State   Pension\nage") == embedder.embed_documents(["State Pension age"])[0]


def test_batching_gives_same_result(embedder):
    texts = [f"pension rule number {i} about national insurance" for i in range(10)]
    assert embedder.embed_documents(texts, batch_size=3) == embedder.embed_documents(texts, batch_size=100)


def test_bad_provider_output_is_caught():
    class Broken(EmbeddingProvider):
        dim = 4
        def _embed(self, texts):
            return [[1.0, 2.0]] * len(texts)  # wrong dimension
    with pytest.raises(EmbeddingError):
        Broken(EmbeddingConfig()).embed_documents(["x"])


def test_fingerprint_changes_with_configuration():
    a = HashingEmbedder(EmbeddingConfig(dim=128)).fingerprint()
    assert a != HashingEmbedder(EmbeddingConfig(dim=256)).fingerprint()
    assert a == HashingEmbedder(EmbeddingConfig(dim=128)).fingerprint()


def test_unknown_provider_rejected():
    with pytest.raises(EmbeddingError):
        create_embedding_provider(EmbeddingConfig(provider="nope"))
