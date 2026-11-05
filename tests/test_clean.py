import json

from src.ingestion.clean import clean_documents, markup_noise_count
from tests.conftest import write_jsonl


def rec(i, text, **kw):
    return {"doc_id": i, "title": "T" + i, "source_url": "https://x/" + i, "text": text, **kw}


def test_noise_document_excluded_and_logged(tmp_path):
    src = tmp_path / "in.jsonl"
    write_jsonl(src, [rec("a", "Normal pension text."), rec("b", '<svg viewBox="0 0 1 1"><path d="M1 2"/></svg> ' * 3)])
    report = clean_documents(src, tmp_path / "out.jsonl", tmp_path / "r.json")
    kept = [json.loads(l)["doc_id"] for l in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert kept == ["a"] and report["excluded"][0]["document_id"] == "b" and "markup_noise" in report["excluded"][0]["reason"]


def test_text_unchanged_and_duplicates_reported_not_removed(tmp_path):
    src = tmp_path / "in.jsonl"
    write_jsonl(src, [rec("a", "Same   text here.", licence="OGL"), rec("b", "Same text here."), rec("c", "Different.")])
    report = clean_documents(src, tmp_path / "out.jsonl", tmp_path / "r.json")
    out = [json.loads(l) for l in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert [o["doc_id"] for o in out] == ["a", "b", "c"] and out[0]["text"] == "Same   text here."
    assert report["identical_text_groups"] == 1 and report["largest_identical_group"] == 2 and report["missing_licence"] == 2


def test_noise_counter():
    assert markup_noise_count("plain") == 0 and markup_noise_count("<svg><path>") == 2
