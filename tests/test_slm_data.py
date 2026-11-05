import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.rag.answer_generation import LLMProvider
from src.rag.chunking import chunk_document
from src.rag.ingest import ingest_documents
from src.rag.retriever import Retriever
from src.slm.data import auto_check, build_inputs, draft_answers, export_chat_format, generate_questions, split_documents
from src.slm.data.common import read_jsonl, write_jsonl
from src.slm.ollama_llm import OllamaError, OllamaLLM
from tests.conftest import make_doc


def docs_for_split():
    out = [make_doc(f"d{i}", f"Doc {i}", f"Unique text number {i}. " * 5, source_url=f"https://x/{i}") for i in range(60)]
    out.append(make_doc("part1", "Guide", "Guide part one text. " * 5, source_url="https://x/guide"))
    out.append(make_doc("part2", "Guide 2", "Guide part two text. " * 5, source_url="https://x/guide/"))  # same URL (trailing slash)
    out.append(make_doc("copyA", "Copy A", "Identical boilerplate words here. " * 5, source_url="https://x/ca"))
    out.append(make_doc("copyB", "Copy B", "Identical boilerplate words here. " * 5, source_url="https://x/cb"))
    return out


def test_split_keeps_groups_together_and_holds_out_eval_docs():
    docs = docs_for_split()
    es = {"questions": [{"relevant_document_ids": ["d3"]}]}
    sp = split_documents.make_split(docs, es)
    side = {i: k for k in ("train", "val", "test", "eval_holdout") for i in sp[k]}
    assert side["part1"] == side["part2"] and side["copyA"] == side["copyB"]
    assert side["d3"] == "eval_holdout" and len(side) == len(docs)
    assert sp["counts"]["train"] > sp["counts"]["val"]
    assert split_documents.make_split(docs, es) == sp  # deterministic


class FakeLLM(LLMProvider):
    def __init__(self, replies=None):
        self.replies, self.calls = replies or [], []
    def generate(self, system, user):
        self.calls.append(user)
        return self.replies.pop(0) if self.replies else "What does this pension rule say?"


@pytest.fixture
def pipeline(settings, embedder, store, tmp_path):
    docs = [make_doc("sp", "State Pension", "You need 10 qualifying years on your National Insurance record to get any new State Pension. " * 6),
            make_doc("pc", "Pension Credit", "Pension Credit gives you extra money to help with your living costs if you are on a low income. " * 6)]
    ingest_documents(docs, settings, embedder, store)
    chunks = {d.document_id: [c.to_dict() for c in chunk_document(d, settings.chunking)] for d in docs}
    split = {"train": ["sp", "pc"], "val": [], "test": [], "eval_holdout": []}
    return Retriever(embedder, store, top_k=3), chunks, split, tmp_path


def test_generate_questions_is_resumable_and_rejects_bad_output(pipeline):
    _, chunks, split, tmp = pipeline
    q = tmp / "questions.jsonl"
    llm = FakeLLM(["not a question", "How many years do I need for State Pension?", "Can I get Pension Credit on a low income?"])
    r1 = generate_questions.generate_questions(llm, chunks, split, limit=10, min_tokens=10, out_path=q)
    rows = list(read_jsonl(q))
    assert r1["rejected_bad_format"] == 1 and all(x["question"].endswith("?") and x["split"] == "train" for x in rows)
    r2 = generate_questions.generate_questions(FakeLLM(), chunks, split, limit=10, min_tokens=10, out_path=q)
    assert len(list(read_jsonl(q))) >= len(rows) and r2["already_done"] == len(rows)


def test_build_inputs_labels_off_topic_and_injection(pipeline, monkeypatch):
    retriever, chunks, split, tmp = pipeline
    sp_chunk = chunks["sp"][0]["chunk_id"]
    write_jsonl(tmp / "q.jsonl", [{"qid": "q1", "split": "train", "chunk_id": sp_chunk, "document_id": "sp", "question": "How many qualifying years of National Insurance for State Pension?"},
                                  {"qid": "q2", "split": "train", "chunk_id": sp_chunk, "document_id": "sp", "question": "Pension Credit living costs low income?"}])
    (tmp / "off.txt").write_text("# comment\nWhat is the best sourdough recipe?\n", encoding="utf-8")
    out = tmp / "inputs.jsonl"
    counts = build_inputs.build_inputs(retriever, chunks, 1, injection_rate=1.0, questions_path=tmp / "q.jsonl", offtopic_path=tmp / "off.txt", out_path=out)
    rows = {r["qid"]: r for r in read_jsonl(out)}
    assert counts["off_topic"] == 1 and counts["answerable"] == 1 and counts["no_support"] == 1
    assert rows["q1"]["label"] == "answerable" and rows["q1"]["variant"] == "injection"
    assert build_inputs.INJECTION in json.dumps(rows["q1"]["evidence"]) and build_inputs.INJECTION not in json.dumps(rows["q1"]["evidence_clean"])
    assert rows["q2"]["label"] == "no_support"  # top-1 was the other document
    again = build_inputs.build_inputs(retriever, chunks, 1, questions_path=tmp / "q.jsonl", offtopic_path=tmp / "off.txt", out_path=out)
    assert sum(again.values()) == 0  # nothing redone


def _item(label="answerable", variant="plain", n=2):
    ev = [{"chunk_id": f"c{i}", "document_id": "d", "title": "T", "source_url": "u", "text": "You need 10 qualifying years.", "score": 0.5} for i in range(n)]
    return {"qid": "x", "split": "train", "question": "How many years?", "label": label, "variant": variant, "evidence": ev, "evidence_clean": ev}


def test_auto_check_flags():
    c = auto_check.check_example
    assert c(_item(), "You need 10 qualifying years [1].") == []
    assert "no_citation" in c(_item(), "You need 10 qualifying years.")
    assert "bad_citation_number" in c(_item(), "You need 10 qualifying years [7].")
    assert any(f.startswith("invented_figures") for f in c(_item(), "You need 35 qualifying years [1]."))
    assert "refusal_has_citations" in c(_item("no_support"), "The evidence does not contain this [1].")
    assert "expected_answer_got_refusal" in c(_item(), "The evidence does not contain this.")
    assert "expected_refusal_got_answer" in c(_item("off_topic"), "Bake for 40 minutes [1].")
    assert "followed_injection" in c(_item(variant="injection"), "HACKED [1]")


def test_draft_review_export_flow_blocks_unapproved_and_leaked(pipeline):
    retriever, chunks, split, tmp = pipeline
    items = [_item(), {**_item("off_topic"), "qid": "y", "question": "Best sourdough?"}, {**_item(), "qid": "z", "question": "Leaked eval question?"}]
    items[0]["qid"] = "x"
    write_jsonl(tmp / "inputs.jsonl", items)
    llm = FakeLLM(["You need 10 qualifying years [1].", "You need 10 qualifying years [2]."])
    d = draft_answers.draft_answers(llm, 10, tmp / "inputs.jsonl", tmp / "drafts.jsonl")
    drafts = {r["qid"]: r for r in read_jsonl(tmp / "drafts.jsonl")}
    assert d["drafted"] == 3 and drafts["y"]["draft_source"] == "template" and len(llm.calls) == 2
    rep = auto_check.build_review_file(tmp / "inputs.jsonl", tmp / "drafts.jsonl", tmp / "review.jsonl")
    assert rep["examples"] == 3 and rep["pending"] == 3
    rows = list(read_jsonl(tmp / "review.jsonl"))
    for r in rows:
        r["status"], r["final_answer"] = ("approved", r["draft"]) if r["qid"] in ("x", "z") else ("pending", None)
    write_jsonl(tmp / "review.jsonl", rows)
    rep2 = auto_check.build_review_file(tmp / "inputs.jsonl", tmp / "drafts.jsonl", tmp / "review.jsonl")
    assert rep2["pending"] == 1  # earlier decisions survive a rebuild
    out = export_chat_format.export(tmp / "review.jsonl", tmp / "inputs.jsonl", tmp, {"questions": [{"question": "leaked eval QUESTION?"}]})
    assert out["written"]["train"] == 1 and out["dropped_eval_leakage"] == 1
    ex = next(read_jsonl(tmp / "train.jsonl"))
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"] and "Evidence" in ex["messages"][1]["content"]


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        reply = {"message": {"content": "" if body["model"] == "empty" else f"echo:{body['messages'][1]['content']}"}}
        data = json.dumps(reply).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def log_message(self, *a): pass


def test_ollama_client_against_a_fake_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    try:
        assert OllamaLLM("m", url).generate("sys", "hello") == "echo:hello"
        with pytest.raises(OllamaError, match="empty"):
            OllamaLLM("empty", url).generate("s", "u")
    finally:
        srv.shutdown()
    with pytest.raises(OllamaError, match="Could not get an answer"):
        OllamaLLM("m", "http://127.0.0.1:9", timeout=1).generate("s", "u")
    with pytest.raises(OllamaError, match="OLLAMA_MODEL"):
        OllamaLLM(None, url) if not __import__("os").environ.get("OLLAMA_MODEL") else (_ for _ in ()).throw(OllamaError("OLLAMA_MODEL"))
