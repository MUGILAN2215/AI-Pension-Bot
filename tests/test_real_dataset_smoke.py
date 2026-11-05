"""Read-only checks on the real merged dataset (skipped if it is not present). Nothing is written."""
import pytest

from src.ingestion.validate_dataset import validate_jsonl
from src.rag.chunking import chunk_document
from src.rag.config import Settings

PATH = Settings.from_env().documents_path
pytestmark = pytest.mark.skipif(not PATH.exists(), reason="data/processed/documents_merged.jsonl not present")


def test_real_dataset_validates_and_chunks_deterministically():
    res = validate_jsonl(PATH)
    assert res.report["invalid_json_lines"] == [] and res.report["duplicate_ids_conflicting"] == []
    assert len(res.documents) + len(res.review) == res.report["valid_json_objects"]
    cfg = Settings.from_env().chunking
    sample = res.documents[:60]
    a = [c.chunk_id for d in sample for c in chunk_document(d, cfg)]
    b = [c.chunk_id for d in sample for c in chunk_document(d, cfg)]
    assert a == b and len(set(a)) == len(a) and a
