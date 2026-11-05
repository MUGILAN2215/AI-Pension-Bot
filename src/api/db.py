"""Tiny SQLite store for users, feedback and upload records (separate from the vector database)."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user','admin')), created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY, username TEXT NOT NULL, message_id TEXT NOT NULL,
                rating TEXT NOT NULL CHECK (rating IN ('up','down')), chunk_id TEXT, comment TEXT, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS uploads (id INTEGER PRIMARY KEY, admin TEXT NOT NULL, filename TEXT NOT NULL, document_id TEXT NOT NULL,
                title TEXT NOT NULL, chunks INTEGER, status TEXT NOT NULL, created_at TEXT NOT NULL);
            """)

    def _run(self, sql: str, args: tuple = (), one: bool = False, write: bool = False):
        with self.lock:
            cur = self.conn.execute(sql, args)
            if write:
                self.conn.commit()
                return cur.lastrowid
            rows = cur.fetchall()
        return (rows[0] if rows else None) if one else rows

    def create_user(self, username: str, password_hash: str, role: str) -> bool:
        try:
            self._run("INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)", (username, password_hash, role, now()), write=True)
            return True
        except sqlite3.IntegrityError:
            return False

    def get_user(self, username: str):
        return self._run("SELECT * FROM users WHERE username = ?", (username,), one=True)

    def add_feedback(self, username, message_id, rating, chunk_id, comment) -> int:
        return self._run("INSERT INTO feedback (username, message_id, rating, chunk_id, comment, created_at) VALUES (?,?,?,?,?,?)",
                         (username, message_id, rating, chunk_id, comment, now()), write=True)

    def add_upload(self, admin, filename, document_id, title, chunks, status) -> int:
        return self._run("INSERT INTO uploads (admin, filename, document_id, title, chunks, status, created_at) VALUES (?,?,?,?,?,?,?)",
                         (admin, filename, document_id, title, chunks, status, now()), write=True)

    def list_uploads(self) -> list[dict]:
        return [dict(r) for r in self._run("SELECT * FROM uploads ORDER BY id DESC LIMIT 200")]

    def feedback_summary(self) -> dict:
        rows = self._run("SELECT rating, COUNT(*) AS n FROM feedback GROUP BY rating")
        return {r["rating"]: r["n"] for r in rows}
