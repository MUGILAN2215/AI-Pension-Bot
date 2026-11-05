import dataclasses

import pytest

from src.rag.embeddings import HashingEmbedder
from src.rag.config import EmbeddingConfig
from src.rag.ingest import ConfigMismatchError, ingest_documents, load_manifest
from src.rag.vector_store import CollectionConfigError, QdrantVectorStore, point_id
from tests.conftest import PENSION_CREDIT, STATE_PENSION, WORKPLACE, make_doc


def run(docs, settings, embedder, store, **kw):
    return ingest_documents(docs, settings, embedder, store, **kw)


def test_qdrant_persistence_and_payload_retrieval(settings, embedder, docs):  # 12
    s1 = QdrantVectorStore(settings.collection_name, path=str(settings.qdrant_path))
    run(docs, settings, embedder, s1)
    n = s1.count()
    s1.close()
    s2 = QdrantVectorStore(settings.collection_name, path=str(settings.qdrant_path))  # reopen from disk
    assert s2.count() == n > 0
    payload = s2.get_document_chunks("pc")[0]
    assert payload["title"] == "Pension Credit" and payload["source_url"] == PENSION_CREDIT.source_url
    assert "extra money" in payload["text"] and payload["licence"] == "OGL v3.0"
    s2.close()


def test_point_ids_are_stable():
    assert point_id("a#00000") == point_id("a#00000") != point_id("a#00001")


def test_repeated_ingestion_is_safe(settings, embedder, store, docs):  # 13
    r1 = run(docs, settings, embedder, store)
    n = store.count()
    r2 = run(docs, settings, embedder, store)
    assert len(r1.new) == 3 and len(r2.skipped) == 3 and not r2.new and not r2.changed
    assert store.count() == n


def test_incremental_addition_of_new_document(settings, embedder, store, docs):  # 14
    run(docs, settings, embedder, store)
    before = store.count()
    extra = make_doc("tx", "Tax on pensions", "Most pensions can be taken partly as a tax free lump sum of up to 25 per cent.")
    r = run(docs + [extra], settings, embedder, store)
    assert r.new == ["tx"] and len(r.skipped) == 3
    assert store.count() > before and store.get_document_chunks("tx")


def test_incremental_update_replaces_old_chunks(settings, embedder, store):  # 15
    long_text = "\n".join(f"Old rule {i} about the State Pension and contributions record." for i in range(60))
    v1 = make_doc("sp", "State Pension", long_text)
    run([v1, PENSION_CREDIT], settings, embedder, store)
    old_count = len(store.get_document_chunks("sp"))
    assert old_count > 2
    v2 = make_doc("sp", "State Pension", "The State Pension is now paid weekly.")  # much shorter
    r = run([v2, PENSION_CREDIT], settings, embedder, store)
    chunks = store.get_document_chunks("sp")
    assert r.changed == ["sp"] and r.skipped == ["pc"]
    assert len(chunks) == 1 and "paid weekly" in chunks[0]["text"]
    assert all(c["document_hash"] == v2.content_hash() for c in chunks)  # no mixture of old and new
    assert load_manifest(settings.manifest_path)["documents"]["sp"]["chunk_count"] == 1
    assert store.count() == 1 + len(store.get_document_chunks("pc"))


def test_partial_input_never_deletes(settings, embedder, store, docs):  # 16
    run(docs, settings, embedder, store)
    n = store.count()
    r = run([STATE_PENSION], settings, embedder, store)  # partial input
    assert r.deleted == [] and r.not_in_input_kept == 2 and store.count() == n
    with pytest.raises(ValueError, match="full-snapshot"):
        run([STATE_PENSION], settings, embedder, store, delete_missing=True)
    assert store.count() == n


def test_explicit_full_snapshot_reconciliation_deletes(settings, embedder, store, docs):  # 16
    run(docs, settings, embedder, store)
    r = run([STATE_PENSION], settings, embedder, store, delete_missing=True, full_snapshot=True)
    assert sorted(r.deleted) == ["pc", "wp"]
    assert store.get_document_chunks("pc") == [] and store.get_document_chunks("sp")
    assert "pc" not in load_manifest(settings.manifest_path)["documents"]


def test_dry_run_changes_nothing(settings, embedder, store, docs):
    r = run(docs, settings, embedder, store, dry_run=True)
    assert len(r.new) == 3 and store.count() == 0 and not settings.manifest_path.exists()


def test_failure_on_one_document_does_not_stop_others(settings, embedder, store, docs):
    class Flaky(HashingEmbedder):
        def _embed(self, texts):
            if any("workplace pension" in t.lower() for t in texts):
                raise RuntimeError("boom")
            return super()._embed(texts)
    r = run(docs, settings, Flaky(settings.embedding), store)
    assert [f["document_id"] for f in r.failed] == ["wp"] and len(r.new) == 2
    assert store.get_document_chunks("wp") == []
    r2 = run(docs, settings, embedder, store)  # retry succeeds, others skipped
    assert r2.new == ["wp"] and len(r2.skipped) == 2


def test_lost_index_is_detected_and_repaired(settings, embedder, store, docs):
    run(docs, settings, embedder, store)
    store.delete_document("sp")  # index loses data but the manifest still says "indexed"
    r = run(docs, settings, embedder, store)
    assert r.changed == ["sp"] and store.get_document_chunks("sp")


def test_embedding_config_change_requires_rebuild(settings, embedder, store, docs):
    run(docs, settings, embedder, store)
    other = HashingEmbedder(EmbeddingConfig(provider="hashing", dim=128))
    with pytest.raises(ConfigMismatchError):
        run(docs, settings, other, store)
    assert store.count() > 0  # nothing was changed or deleted


def test_dimension_mismatch_on_existing_collection_is_refused(settings, embedder, store, docs):
    run(docs, settings, embedder, store)
    with pytest.raises(CollectionConfigError):
        store.ensure_collection(128)


def test_rebuild_only_affects_configured_collection(settings, embedder, store, docs):
    from qdrant_client.models import Distance, VectorParams
    store.client.create_collection("unrelated", vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    run(docs, settings, embedder, store)
    other = HashingEmbedder(EmbeddingConfig(provider="hashing", dim=128))
    settings128 = dataclasses.replace(settings, embedding=EmbeddingConfig(provider="hashing", dim=128))
    run(docs, settings128, other, store, rebuild_collection=True)
    assert store.client.collection_exists("unrelated") and store.count() > 0


def test_chunks_file_is_updated_without_losing_other_documents(settings, embedder, store, docs):
    import json
    run(docs, settings, embedder, store)
    run([make_doc("sp", "State Pension", "Brand new short text about State Pension.")], settings, embedder, store)
    ids = {json.loads(l)["document_id"] for l in settings.chunks_path.read_text(encoding="utf-8").splitlines()}
    assert ids == {"pc", "sp", "wp"}
