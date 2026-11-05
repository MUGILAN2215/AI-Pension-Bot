import json

from src.ingestion.validate_dataset import validate_jsonl
from tests.conftest import write_jsonl


def rec(i="a", **kw):
    base = {"doc_id": i, "title": "T " + i, "source_url": "https://x/" + i, "text": "Some text. " * 5}
    base.update(kw)
    return base


def test_valid_jsonl_parsing(tmp_path):  # 1
    p = tmp_path / "d.jsonl"
    write_jsonl(p, [rec("a"), rec("b")])
    r = validate_jsonl(p)
    assert r.report["valid_json_objects"] == 2 and len(r.documents) == 2 and not r.review
    assert r.documents[0].document_id == "a"


def test_invalid_jsonl_lines_reported_not_dropped(tmp_path):  # 2
    p = tmp_path / "d.jsonl"
    write_jsonl(p, [rec("a"), "{not json", "[1,2,3]", rec("b")])
    r = validate_jsonl(p)
    assert len(r.documents) == 2
    reasons = sorted(x["reason"] for x in r.review)
    assert reasons == ["invalid_json", "not_a_json_object"]
    assert r.report["invalid_json_lines"][0]["line"] == 2


def test_missing_required_and_optional_metadata(tmp_path):  # 3
    p = tmp_path / "d.jsonl"
    no_title = rec("c")
    del no_title["title"]
    write_jsonl(p, [rec("a"), no_title])
    r = validate_jsonl(p)
    assert [d.document_id for d in r.documents] == ["a"]
    assert r.review[0]["reason"] == "missing_required_field" and "title" in r.review[0]["detail"]
    assert r.documents[0].licence is None  # optional metadata stays missing, not invented
    assert r.report["optional_fields_missing_in_usable_docs"]["licence"] == 1


def test_duplicate_ids_identical(tmp_path):  # 4
    p = tmp_path / "d.jsonl"
    write_jsonl(p, [rec("a"), rec("a")])
    r = validate_jsonl(p)
    assert len(r.documents) == 1
    assert r.report["duplicate_ids_identical"] == [{"document_id": "a", "lines": [1, 2]}]


def test_conflicting_versions_go_to_review(tmp_path):  # 5
    p = tmp_path / "d.jsonl"
    write_jsonl(p, [rec("a", text="Version one text."), rec("a", text="Version two text."), rec("b")])
    r = validate_jsonl(p)
    assert [d.document_id for d in r.documents] == ["b"]
    assert [x["reason"] for x in r.review] == ["conflicting_versions_same_id"] * 2
    assert r.report["duplicate_ids_conflicting"][0]["document_id"] == "a"


def test_empty_documents_reported(tmp_path):  # 6
    p = tmp_path / "d.jsonl"
    write_jsonl(p, [rec("a", text="   \n"), rec("b", text="")])
    r = validate_jsonl(p)
    assert not r.documents and sorted(r.report["empty_text_documents"]) == ["a", "b"]
    assert all(x["reason"] == "empty_text" for x in r.review)
    assert r.review[0]["record"]["doc_id"] == "a"  # full record preserved for review


def test_original_file_not_modified(tmp_path):
    p = tmp_path / "d.jsonl"
    write_jsonl(p, [rec("a"), "{bad"])
    before = p.read_bytes()
    validate_jsonl(p)
    assert p.read_bytes() == before
