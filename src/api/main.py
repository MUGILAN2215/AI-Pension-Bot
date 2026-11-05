"""FastAPI backend.   Start:  uvicorn src.api.main:create_app --factory --port 8000   (single worker; see README)

Endpoints
  POST /api/auth/login        {username, password} -> token
  POST /api/auth/register     only when ALLOW_REGISTRATION=true (creates a normal user)
  GET  /api/me
  POST /api/chat              JSON answer with citations
  POST /api/chat/stream       same answer as Server-Sent Events (the finished answer is sent in small pieces; it is not model-token streaming)
  GET  /api/chunks/{chunk_id} passage behind a citation (for the preview drawer)
  GET  /api/health            vector store / embedding / configuration status (no login needed)
  POST /api/feedback          thumbs up/down on an answer or a citation
  POST /api/admin/documents   admin only: upload a document and index it (no model training needed)
  GET  /api/admin/documents   admin only: upload history
  GET  /api/admin/feedback    admin only: feedback totals
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from src.api.db import Database
from src.api.security import create_token, decode_token, hash_password, secret_key, verify_password
from src.api.service import RagService, short_date, source_label
from src.api.uploads import UploadError, build_document
from src.rag.config import PROJECT_ROOT

bearer = HTTPBearer(auto_error=False)


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=200)


class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class FeedbackRequest(BaseModel):
    message_id: str = Field(min_length=1, max_length=100)
    rating: str = Field(pattern="^(up|down)$")
    chunk_id: str | None = Field(default=None, max_length=200)
    comment: str | None = Field(default=None, max_length=1000)


def create_app(service: RagService | None = None, db: Database | None = None, key: str | None = None) -> FastAPI:
    app = FastAPI(title="UK Pension AI", version="0.1.0")
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass
    origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.state.service = service or RagService.from_env()
    app.state.db = db or Database(os.environ.get("API_DB_PATH", PROJECT_ROOT / "data/app.sqlite"))
    app.state.key = key or secret_key()
    failures: dict[str, list[float]] = defaultdict(list)

    admin_user, admin_pw = os.environ.get("ADMIN_USERNAME"), os.environ.get("ADMIN_PASSWORD")
    if admin_user and admin_pw and len(admin_pw) >= 8:
        pw_hash = hash_password(admin_pw)
        if not app.state.db.create_user(admin_user, pw_hash, "admin"):
            app.state.db._run("UPDATE users SET password_hash = ? WHERE username = ?", (pw_hash, admin_user), write=True)

    def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
        data = decode_token(creds.credentials, app.state.key) if creds else None
        if not data:
            raise HTTPException(401, "Please log in", headers={"WWW-Authenticate": "Bearer"})
        return {"username": data["sub"], "role": data["role"]}

    def admin_user_dep(user: dict = Depends(current_user)) -> dict:
        if user["role"] != "admin":
            raise HTTPException(403, "Admin access required")
        return user

    # ---- auth -----------------------------------------------------------------------------------
    @app.post("/api/auth/login")
    def login(c: Credentials):
        now = time.time()
        failures[c.username] = [t for t in failures[c.username] if now - t < 300]
        if len(failures[c.username]) >= 5:
            raise HTTPException(429, "Too many failed attempts; try again in a few minutes")
        row = app.state.db.get_user(c.username)
        if not row or not verify_password(c.password, row["password_hash"]):
            failures[c.username].append(now)
            raise HTTPException(401, "Wrong username or password")
        return {"access_token": create_token(row["username"], row["role"], app.state.key), "token_type": "bearer",
                "username": row["username"], "role": row["role"]}

    @app.post("/api/auth/register", status_code=201)
    def register(c: Credentials):
        if os.environ.get("ALLOW_REGISTRATION", "false").lower() != "true":
            raise HTTPException(403, "Self-registration is disabled; ask an admin")
        if not app.state.db.create_user(c.username, hash_password(c.password), "user"):
            raise HTTPException(409, "That username is taken")
        return {"username": c.username, "role": "user"}

    @app.get("/api/me")
    def me(user: dict = Depends(current_user)):
        return user

    # ---- chat -----------------------------------------------------------------------------------
    def run_chat(req: ChatRequest) -> dict:
        history = [t.model_dump() for t in req.history]
        result = app.state.service.ask(req.question.strip(), history)
        ans = result.answer
        meta_by_chunk = {e.chunk_id: e.metadata for e in ans.evidence}
        citations = []
        for c in ans.citations:
            m = meta_by_chunk.get(c.chunk_id, {})
            citations.append({"marker": c.marker, "chunk_id": c.chunk_id, "document_id": c.document_id, "title": c.title, "source_url": c.source_url,
                              "heading_path": c.heading_path, "last_updated": short_date(m.get("last_updated_date") or m.get("date")), "licence": m.get("licence"),
                              **source_label(m)})
        evidence_list = []
        for e in ans.evidence:
            m = e.metadata or {}
            evidence_list.append({"rank": e.rank, "score": round(float(e.score), 4), "chunk_id": e.chunk_id, "document_id": e.document_id,
                                  "title": e.title, "source_url": e.source_url, "heading_path": e.heading_path, "text": e.text,
                                  "last_updated": short_date(m.get("last_updated_date") or m.get("date")), **source_label(m)})
        out = {"message_id": uuid.uuid4().hex, "status": ans.status, "mode": ans.mode, "answer": ans.text, "citations": citations,
               "evidence": evidence_list, "notes": ans.notes,
               "rewritten_question": result.rewritten_question, "route": result.route, "steps": result.steps, "calculator": result.calculator,
               "suggestions": result.suggestions}
        if os.environ.get("DEBUG_RAW_REPLIES", "false").lower() == "true":
            out["raw_reply"] = ans.raw_reply  # what the model actually wrote (debugging only; leave off in production)
        return out

    @app.post("/api/chat")
    def chat(req: ChatRequest, user: dict = Depends(current_user)):
        return run_chat(req)

    @app.post("/api/chat/stream")
    def chat_stream(req: ChatRequest, user: dict = Depends(current_user)):
        data = run_chat(req)

        def sse(event: str, payload) -> str:
            return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        def gen():
            yield sse("meta", {k: data[k] for k in ("message_id", "status", "mode", "rewritten_question", "route", "steps") if k in data})
            words = data["answer"].split(" ")
            for i in range(0, len(words), 4):
                yield sse("token", {"text": " ".join(words[i:i + 4]) + (" " if i + 4 < len(words) else "")})
            yield sse("citations", {"citations": data["citations"], "evidence": data.get("evidence", []), "notes": data["notes"], "calculator": data["calculator"], "suggestions": data.get("suggestions", []), **({"raw_reply": data["raw_reply"]} if "raw_reply" in data else {})})
            yield sse("done", {})
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/chunks/{chunk_id:path}")
    def chunk(chunk_id: str, user: dict = Depends(current_user)):
        c = app.state.service.get_chunk(chunk_id)
        if not c:
            raise HTTPException(404, "Passage not found")
        return c

    # ---- health / feedback ----------------------------------------------------------------------
    @app.get("/api/health")
    def health():
        return app.state.service.health()

    @app.post("/api/feedback", status_code=201)
    def feedback(f: FeedbackRequest, user: dict = Depends(current_user)):
        return {"id": app.state.db.add_feedback(user["username"], f.message_id, f.rating, f.chunk_id, f.comment)}

    # ---- admin ----------------------------------------------------------------------------------
    @app.post("/api/admin/documents", status_code=201)
    async def upload(file: UploadFile = File(...), title: str = Form(...), source_url: str = Form(""), publisher: str = Form(""),
                     licence: str = Form(""), date: str = Form(""), user: dict = Depends(admin_user_dep)):
        data = await file.read()
        try:
            doc = build_document(file.filename or "upload", data, title, source_url, publisher, licence, date)
        except UploadError as exc:
            raise HTTPException(422, str(exc))
        result = app.state.service.ingest_upload(doc)
        app.state.db.add_upload(user["username"], file.filename or "upload", doc.document_id, doc.title, result.get("chunks"), result["status"])
        if result["status"] == "failed":
            raise HTTPException(500, f"Indexing failed: {result['error']}")
        return result

    @app.get("/api/admin/documents")
    def uploads(user: dict = Depends(admin_user_dep)):
        return app.state.db.list_uploads()

    @app.get("/api/admin/feedback")
    def feedback_totals(user: dict = Depends(admin_user_dep)):
        return app.state.db.feedback_summary()

    return app
