import io
import json
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from src.agent.graph import PensionAgent
from src.api.db import Database
from src.api.main import create_app
from src.api.security import create_token, decode_token, hash_password, verify_password
from src.api.service import RagService
from src.api.uploads import UploadError, extract_text
from src.rag.answer_generation import AnswerGenerator
from src.rag.ingest import ingest_documents
from src.rag.retriever import Retriever

KEY = "k" * 40


@pytest.fixture
def client(settings, embedder, store, docs, tmp_path):
    ingest_documents(docs, settings, embedder, store)
    retriever = Retriever(embedder, store, top_k=3, mode="hybrid", dedupe=True)
    generator = AnswerGenerator(min_score=0.2)
    service = RagService(settings, embedder, store, retriever, generator, PensionAgent(retriever, generator), tmp_path / "uploads.jsonl")
    db = Database(tmp_path / "app.sqlite")
    db.create_user("alice", hash_password("password123"), "user")
    db.create_user("boss", hash_password("adminpass123"), "admin")
    return TestClient(create_app(service, db, KEY))


def token(client, user="alice", pw="password123"):
    r = client.post("/api/auth/login", json={"username": user, "password": pw})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["access_token"]}


def test_password_hashing_and_tokens():
    h = hash_password("secret-pass")
    assert verify_password("secret-pass", h) and not verify_password("wrong", h) and not verify_password("x", "garbage")
    assert h != hash_password("secret-pass")  # salted
    t = create_token("a", "user", KEY)
    assert decode_token(t, KEY)["sub"] == "a" and decode_token(t, "other" * 10) is None and decode_token("junk", KEY) is None
    assert decode_token(create_token("a", "user", KEY, hours=-1), KEY) is None  # expired


def test_login_and_protected_routes(client):
    assert client.post("/api/chat", json={"question": "State Pension"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "alice", "password": "wrongpass1"}).status_code == 401
    h = token(client)
    assert client.get("/api/me", headers=h).json() == {"username": "alice", "role": "user"}
    assert client.get("/api/me", headers={"Authorization": "Bearer junk"}).status_code == 401


def test_login_lockout_after_repeated_failures(client):
    for _ in range(5):
        client.post("/api/auth/login", json={"username": "alice", "password": "badpassword"})
    assert client.post("/api/auth/login", json={"username": "alice", "password": "password123"}).status_code == 429


def test_registration_disabled_by_default_then_enabled(client, monkeypatch):
    body = {"username": "newbie", "password": "longenough1"}
    assert client.post("/api/auth/register", json=body).status_code == 403
    monkeypatch.setenv("ALLOW_REGISTRATION", "true")
    assert client.post("/api/auth/register", json=body).status_code == 201
    assert client.post("/api/auth/register", json=body).status_code == 409
    assert client.post("/api/auth/register", json={"username": "x", "password": "short"}).status_code == 422
    assert token(client, "newbie", "longenough1")


def test_chat_returns_cited_answer_with_trust_fields(client):
    r = client.post("/api/chat", headers=token(client), json={"question": "How do I apply for Pension Credit?"})
    d = r.json()
    assert r.status_code == 200 and d["status"] == "ok" and d["citations"]
    c = d["citations"][0]
    assert c["source_url"].startswith("https://www.gov.uk/") and c["licence"] == "OGL v3.0" and c["last_updated"] == "2024-01-01" and c["label"] == "GOV.UK"
    assert d["message_id"] and "steps" in d


def test_chat_refuses_unsupported_and_validates_input(client):
    h = token(client)
    d = client.post("/api/chat", headers=h, json={"question": "What is the best recipe for sourdough bread?"}).json()
    assert d["status"] == "insufficient_evidence" and d["citations"] == []
    assert client.post("/api/chat", headers=h, json={"question": ""}).status_code == 422
    assert client.post("/api/chat", headers=h, json={"question": "x" * 1001}).status_code == 422
    assert client.post("/api/chat", headers=h, json={"question": "hi", "history": [{"role": "system", "content": "x"}]}).status_code == 422


def test_chat_stream_events(client):
    r = client.post("/api/chat/stream", headers=token(client), json={"question": "How do I apply for Pension Credit?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [b.split("\n")[0].removeprefix("event: ") for b in r.text.strip().split("\n\n")]
    assert events[0] == "meta" and events[-1] == "done" and "token" in events and "citations" in events
    text = "".join(json.loads(b.split("data: ", 1)[1])["text"] for b in r.text.strip().split("\n\n") if b.startswith("event: token"))
    assert "Pension Credit" in text


def test_chunk_preview_and_feedback(client):
    h = token(client)
    cit = client.post("/api/chat", headers=h, json={"question": "How do I apply for Pension Credit?"}).json()["citations"][0]
    c = client.get("/api/chunks/" + quote(cit["chunk_id"], safe=""), headers=h).json()  # "#" must be URL-encoded
    assert c["title"] == "Pension Credit" and "Pension Credit" in c["text"] and c["admin_uploaded"] is False
    assert client.get("/api/chunks/nope%2300000", headers=h).status_code == 404
    assert client.post("/api/feedback", headers=h, json={"message_id": "m1", "rating": "up", "chunk_id": cit["chunk_id"]}).status_code == 201
    assert client.post("/api/feedback", headers=h, json={"message_id": "m1", "rating": "meh"}).status_code == 422
    assert client.get("/api/admin/feedback", headers=h).status_code == 403
    assert client.get("/api/admin/feedback", headers=token(client, "boss", "adminpass123")).json() == {"up": 1}


def test_health_needs_no_login(client):
    d = client.get("/api/health").json()
    assert d["vector_store"]["ok"] and d["vector_store"]["chunks"] > 0 and d["embedding"]["dim"] == 256 and "note" in d


def test_upload_is_admin_only_and_makes_document_searchable(client):
    text = b"# Zebra scheme\n\nThe zebra marmalade allowance is a made-up fact used only in this test."
    files = {"file": ("zebra.txt", text, "text/plain")}
    data = {"title": "Zebra scheme", "source_url": "https://example.test/zebra", "licence": "Test licence"}
    assert client.post("/api/admin/documents", files=files, data=data).status_code == 401
    assert client.post("/api/admin/documents", files=files, data=data, headers=token(client)).status_code == 403
    admin = token(client, "boss", "adminpass123")
    r = client.post("/api/admin/documents", files={"file": ("zebra.txt", text, "text/plain")}, data=data, headers=admin)
    assert r.status_code == 201 and r.json()["status"] == "indexed" and r.json()["chunks"] >= 1
    d = client.post("/api/chat", headers=token(client), json={"question": "What is the zebra marmalade allowance?"}).json()
    assert d["citations"][0]["title"] == "Zebra scheme" and d["citations"][0]["admin_uploaded"] is True
    again = client.post("/api/admin/documents", files={"file": ("zebra.txt", text, "text/plain")}, data=data, headers=admin)
    assert again.json()["status"] == "skipped_unchanged"
    assert [u["title"] for u in client.get("/api/admin/documents", headers=admin).json()][0] == "Zebra scheme"


def test_upload_rejects_bad_files(client):
    admin = token(client, "boss", "adminpass123")
    def post(name, data, title="T"):
        return client.post("/api/admin/documents", files={"file": (name, data, "application/octet-stream")}, data={"title": title}, headers=admin)
    assert post("x.exe", b"abc").status_code == 422
    assert post("x.txt", b"   \n").status_code == 422
    assert post("x.txt", b"\xff\xfe\x00bad").status_code == 422
    assert post("x.txt", b"text", title="  ").status_code == 422
    assert post("x.pdf", b"not a pdf").status_code == 422


def test_pdf_and_docx_text_extraction():
    from docx import Document as Docx
    from fpdf import FPDF
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Helvetica", size=12); pdf.cell(0, 10, "Quokka bursary rules for testing"); out = bytes(pdf.output())
    assert "Quokka bursary" in extract_text("a.pdf", out)
    d = Docx(); d.add_heading("Main title", level=1); d.add_paragraph("Body sentence about wombats."); t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "A", "B"; buf = io.BytesIO(); d.save(buf)
    txt = extract_text("a.docx", buf.getvalue())
    assert txt.startswith("# Main title") and "wombats" in txt and "A | B" in txt
    with pytest.raises(UploadError):
        extract_text("big.txt", b"x" * (10 * 1024 * 1024 + 1))


def test_short_date_and_switches():
    from src.api.service import make_generator, make_reranker, short_date
    from src.rag.answer_generation import LLMProvider
    assert short_date("2026-09-09T14:55:07+01:00") == "2026-09-09" and short_date("2025-01-31") == "2025-01-31" and short_date(None) is None
    class L(LLMProvider):
        def generate(self, s, u): return "x"
    llm = L()
    assert make_generator(llm, 0.5, True).judge is llm and make_generator(llm, 0.5, False).judge is None and make_generator(None, 0.5, True).judge is None
    assert make_reranker(llm, "off") is None and make_reranker(None, "llm") is None and make_reranker(llm, "llm") is not None


def test_dates_in_citations_are_short_and_raw_reply_only_in_debug(client, monkeypatch):
    h = token(client)
    d = client.post("/api/chat", headers=h, json={"question": "How do I apply for Pension Credit?"}).json()
    assert "raw_reply" not in d and all(len(c["last_updated"] or "") == 10 for c in d["citations"])
    monkeypatch.setenv("DEBUG_RAW_REPLIES", "true")
    assert "raw_reply" in client.post("/api/chat", headers=h, json={"question": "How do I apply for Pension Credit?"}).json()
