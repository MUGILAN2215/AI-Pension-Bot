"""Typed data structures shared by every part of the RAG module.

Missing information stays `None`; nothing is invented.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any

# Fields that define a document's *content/identity*. If any change, the document
# counts as changed and is re-ingested. Fields like `retrieved_at` and `provenance`
# are deliberately excluded so a re-download of identical content is not "a change".
CONTENT_HASH_FIELDS = (
    "title", "text", "source_url", "section", "publisher", "source", "source_type",
    "category", "licence", "publication_date", "last_updated_date", "date",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Document:
    """One full source document (or one guide part / manual page)."""

    document_id: str
    title: str
    source_url: str
    text: str
    parent_source_url: str | None = None
    section: str | None = None
    publisher: str | None = None
    source: str | None = None
    source_type: str | None = None
    category: str | None = None
    original_format: str | None = None
    jurisdiction: str | None = None
    language: str | None = None
    publication_date: str | None = None
    last_updated_date: str | None = None
    date: str | None = None
    retrieved_at: str | None = None
    licence: str | None = None
    source_content_hash: str | None = None
    extraction_status: str | None = None
    review_flags: list[str] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)  # any extra original keys

    @classmethod
    def from_record(cls, rec: dict) -> "Document":
        """Build a Document from a JSON record, accepting the merged or original key names."""
        data = dict(rec)
        aliases = {"doc_id": "document_id", "id": "document_id", "url": "source_url",
                   "content_hash": "source_content_hash"}
        for old, new in aliases.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
        known = {f.name for f in fields(cls)} - {"metadata"}
        kwargs = {k: data[k] for k in known if k in data}
        kwargs.setdefault("document_id", "")
        kwargs.setdefault("title", "")
        kwargs.setdefault("source_url", "")
        kwargs.setdefault("text", "")
        for k in ("review_flags", "provenance"):
            if kwargs.get(k) is None:
                kwargs.pop(k, None)
        extras = {k: v for k, v in data.items() if k not in known and k != "metadata"}
        extras.update(data.get("metadata") or {})
        return cls(metadata=extras, **kwargs)

    def content_hash(self) -> str:
        payload = {k: getattr(self, k) for k in CONTENT_HASH_FIELDS}
        return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chunk:
    """A retrievable piece of a document, with everything needed for citation."""

    chunk_id: str
    document_id: str
    chunk_index: int
    chunk_count: int
    text: str
    heading_path: str
    heading: str | None
    title: str
    source_url: str
    token_count: int
    overlap_tokens: int
    content_hash: str
    document_hash: str
    parent_source_url: str | None = None
    section: str | None = None
    publisher: str | None = None
    source: str | None = None
    source_type: str | None = None
    category: str | None = None
    original_format: str | None = None
    jurisdiction: str | None = None
    language: str | None = None
    publication_date: str | None = None
    last_updated_date: str | None = None
    date: str | None = None
    licence: str | None = None
    review_flags: list[str] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)
    chunking: dict[str, Any] = field(default_factory=dict)

    def embedding_text(self) -> str:
        """The text that is embedded: title + heading context + chunk text.

        Context is added for *embedding only*; the stored `text` stays exactly as split.
        """
        head = " | ".join(p for p in (self.title, self.heading_path) if p)
        return f"{head}\n{self.text}" if head else self.text

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class Evidence:
    """A retrieved chunk plus its relevance score (a similarity, not a truth measure)."""

    rank: int
    score: float
    chunk_id: str
    document_id: str
    title: str
    source_url: str
    heading_path: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    """Outcome of a retrieval call. `status` tells callers what happened."""

    query: str
    status: str  # ok | no_results | empty_query | collection_missing | error
    evidence: list[Evidence] = field(default_factory=list)
    message: str = ""
    top_k: int = 0
    filters: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Citation:
    marker: int
    chunk_id: str
    document_id: str
    title: str
    source_url: str
    heading_path: str = ""


@dataclass
class Answer:
    """Evidence-grounded answer. `status` is ok | insufficient_evidence | retrieval_failed | ungrounded."""

    question: str
    status: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # e.g. conflict disclosures
    mode: str = "extractive"  # extractive (no model) or generated
    raw_reply: str | None = None  # what the model actually wrote (kept so rejected replies can be inspected)

    def to_dict(self) -> dict:
        return asdict(self)
