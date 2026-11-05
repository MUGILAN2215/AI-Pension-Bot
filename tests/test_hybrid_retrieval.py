import pytest

from src.rag.bm25 import BM25Index, tokenize
from src.rag.ingest import ingest_documents
from src.rag.retriever import Retriever
from tests.conftest import PENSION_CREDIT, STATE_PENSION, WORKPLACE, make_doc

CODE_DOC = make_doc("form", "Claim forms", "Use form PC10S to apply by post. Send the completed form to the Pension Service address on the form.")


@pytest.fixture
def indexed(settings, embedder, store, docs):
    ingest_documents(docs + [CODE_DOC], settings, embedder, store)
    return store


def test_tokenize_keeps_codes_and_drops_stopwords():
    assert "pc10s" in tokenize("Use form PC10S to apply") and "the" not in tokenize("the form")


def test_bm25_ranks_exact_term_first_and_applies_filters():
    idx = BM25Index([{"chunk_id": "a", "title": "A", "text": "form PC10S apply", "publisher": "X"},
                     {"chunk_id": "b", "title": "B", "text": "general pension text about forms", "publisher": "Y"}])
    assert idx.search("PC10S", 5)[0][0]["chunk_id"] == "a"
    assert idx.search("forms", 5, {"publisher": "Y"})[0][0]["chunk_id"] == "b"
    assert idx.search("nothingmatches", 5) == []


def test_hybrid_finds_exact_code(indexed, embedder):
    res = Retriever(embedder, indexed, top_k=3, mode="hybrid").retrieve("PC10S")
    assert res.ok and res.evidence[0].document_id == "form"
    assert "bm25_rank" in res.evidence[0].metadata["retrieval"]


def test_keyword_only_hits_get_a_cosine_score(indexed, embedder):
    res = Retriever(embedder, indexed, top_k=3, mode="bm25").retrieve("qualifying years National Insurance")
    assert res.ok and all(isinstance(e.score, float) and -1.0 <= e.score <= 1.0 for e in res.evidence)


def test_modes_validated_and_dense_still_default(indexed, embedder):
    with pytest.raises(ValueError):
        Retriever(embedder, indexed, mode="magic")
    assert Retriever(embedder, indexed).mode == "dense"


def test_dedupe_collapses_identical_chunks(settings, embedder, store):
    same = "Identical boilerplate sentence about the lifetime allowance being abolished from April 2024."
    ingest_documents([make_doc("a", "Manual A", same), make_doc("b", "Manual B", same), STATE_PENSION], settings, embedder, store)
    plain = Retriever(embedder, store, top_k=5).retrieve("lifetime allowance abolished")
    dedup = Retriever(embedder, store, top_k=5, dedupe=True).retrieve("lifetime allowance abolished")
    assert {"a", "b"} <= {e.document_id for e in plain.evidence}
    top = [e for e in dedup.evidence if e.document_id in ("a", "b")]
    assert len(top) == 1 and top[0].metadata["retrieval"]["duplicates_collapsed"] == 1


def test_max_per_document(settings, embedder, store):
    long_text = "\n".join(f"Pension rule {i} about the State Pension and qualifying years." for i in range(60))
    ingest_documents([make_doc("long", "Long", long_text), PENSION_CREDIT], settings, embedder, store)
    res = Retriever(embedder, store, top_k=5, max_per_document=1).retrieve("State Pension qualifying years")
    ids = [e.document_id for e in res.evidence]
    assert ids.count("long") == 1


def test_refresh_index_sees_new_documents(settings, embedder, store, docs):
    ingest_documents(docs, settings, embedder, store)
    r = Retriever(embedder, store, top_k=3, mode="bm25")
    assert not [e for e in r.retrieve("zebra").evidence if e.document_id == "z"]
    ingest_documents(docs + [make_doc("z", "Zebra", "The zebra marmalade rule is only a test fact.")], settings, embedder, store)
    r.refresh_index()
    assert r.retrieve("zebra marmalade").evidence[0].document_id == "z"
