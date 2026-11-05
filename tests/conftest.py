"""Shared test helpers. Every test uses a temporary directory and its own Qdrant folder,
so the real dataset, manifest and production collection are never touched."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.rag.config import ChunkingConfig, EmbeddingConfig, Settings
from src.rag.embeddings import HashingEmbedder
from src.rag.schema import Document
from src.rag.vector_store import QdrantVectorStore


def make_doc(doc_id: str, title: str, text: str, **kw) -> Document:
    defaults = dict(source_url=f"https://www.gov.uk/{doc_id}", publisher="GOV.UK", licence="OGL v3.0", date="2024-01-01")
    defaults.update(kw)
    return Document(document_id=doc_id, title=title, text=text, **defaults)


PENSION_CREDIT = make_doc("pc", "Pension Credit", "# Pension Credit\n\n## Overview\n\nPension Credit gives you extra money to help with your living costs if you are over State Pension age and on a low income.\n\n## How to claim\n\nYou can apply for Pension Credit by phone or online through the Pension Service.")
STATE_PENSION = make_doc("sp", "The new State Pension", "# The new State Pension\n\n## Eligibility\n\nYou need 10 qualifying years on your National Insurance record to get any new State Pension.\n\n## When you are paid\n\nYour State Pension is paid every four weeks into your bank account.")
WORKPLACE = make_doc("wp", "Workplace pensions", "# Workplace pensions\n\nYour employer must automatically enrol you in a workplace pension if you earn more than a set amount and are aged between 22 and State Pension age.")


@pytest.fixture
def docs():
    return [PENSION_CREDIT, STATE_PENSION, WORKPLACE]


@pytest.fixture
def settings(tmp_path) -> Settings:
    base = Settings.from_env()
    return replace(
        base,
        documents_path=tmp_path / "documents.jsonl", chunks_path=tmp_path / "chunks.jsonl",
        manifest_path=tmp_path / "manifest.json", validation_report_path=tmp_path / "report.json",
        validation_review_path=tmp_path / "review.jsonl", qdrant_path=tmp_path / "qdrant", qdrant_url=None,
        collection_name="test_collection", min_score=0.1,
        chunking=ChunkingConfig(target_tokens=60, overlap_tokens=10),
        embedding=EmbeddingConfig(provider="hashing", dim=256),
    )


@pytest.fixture
def embedder(settings):
    return HashingEmbedder(settings.embedding)


@pytest.fixture
def store(settings):
    s = QdrantVectorStore(settings.collection_name, path=str(settings.qdrant_path))
    yield s
    s.close()


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write((r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)) + "\n")
