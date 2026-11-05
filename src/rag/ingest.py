"""Safe, incremental ingestion: documents -> chunks -> embeddings -> Qdrant.

Run:  python -m src.rag.ingest            (ingest/update from the configured documents file)
      python -m src.rag.ingest --dry-run  (show what would change, write nothing)

Rules (see README "Incremental ingestion"):
* new document          -> chunk, embed, upsert
* changed document      -> new chunks are upserted first, then the old version's chunks are deleted
* unchanged document    -> skipped
* document missing from the input -> KEPT. Deletion only happens with BOTH
  --delete-missing and --full-snapshot (i.e. you assert the input is the complete corpus).
* embedding model/dimension change -> refused unless --rebuild-collection is given.
* a failure on one document is recorded; the others continue; the failed document is retried next run.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src.rag.chunking import chunk_document, read_chunks_file, write_chunks_file
from src.rag.config import Settings
from src.rag.embeddings import EmbeddingProvider, create_embedding_provider
from src.rag.schema import Document
from src.rag.vector_store import QdrantVectorStore, VectorStore

MANIFEST_VERSION = 1


class ConfigMismatchError(RuntimeError):
    """The embedding configuration differs from the one used to build the index."""


@dataclass
class IngestReport:
    new: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    review_required: list[dict] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    not_in_input_kept: int = 0
    dry_run: bool = False

    def summary(self) -> dict:
        return {"new": len(self.new), "changed": len(self.changed), "skipped": len(self.skipped),
                "failed": len(self.failed), "review_required": len(self.review_required),
                "deleted": len(self.deleted), "not_in_input_kept": self.not_in_input_kept, "dry_run": self.dry_run}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"manifest_version": MANIFEST_VERSION, "created_at": _now(), "config": {}, "documents": {}, "last_run": {}}


def save_manifest(path: Path, manifest: dict) -> None:
    """Atomic write so an interruption never leaves a half-written manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = _now()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def ingest_documents(
    documents: Sequence[Document],
    settings: Settings,
    embedder: EmbeddingProvider,
    store: VectorStore,
    *,
    review: Sequence[dict] = (),
    delete_missing: bool = False,
    full_snapshot: bool = False,
    rebuild_collection: bool = False,
    dry_run: bool = False,
    update_chunks_file: bool = True,
) -> IngestReport:
    """Ingest `documents` incrementally. See the module docstring for the rules."""
    if delete_missing and not full_snapshot:
        raise ValueError("--delete-missing requires --full-snapshot: deletion is never inferred from a partial input.")

    manifest = load_manifest(settings.manifest_path)
    emb_fp, chunk_fp = embedder.fingerprint(), settings.chunking.fingerprint()
    old_cfg = manifest.get("config", {})
    if old_cfg.get("embedding_fingerprint") and old_cfg["embedding_fingerprint"] != emb_fp and not rebuild_collection:
        raise ConfigMismatchError(
            "The embedding configuration changed since the index was built. Old vectors are not comparable with new ones. "
            "Re-run with --rebuild-collection to drop and rebuild ONLY the configured collection.")

    report = IngestReport(dry_run=dry_run)
    report.review_required = [{"line": r.get("line"), "reason": r.get("reason")} for r in review]

    if rebuild_collection and not dry_run:
        store.drop_collection()
        manifest["documents"] = {}
    if not dry_run:
        store.ensure_collection(embedder.dim)

    stored_counts = store.chunk_counts_by_document() if (store.collection_exists() and not dry_run) else None
    docs_state = manifest.setdefault("documents", {})
    chunks_by_doc = read_chunks_file(settings.chunks_path) if (update_chunks_file and not dry_run) else {}
    input_ids = {d.document_id for d in documents}

    for i, doc in enumerate(documents, 1):
        doc_hash = doc.content_hash()
        state = docs_state.get(doc.document_id)
        in_sync = (
            state is not None and state.get("status") == "indexed" and state.get("document_hash") == doc_hash
            and state.get("chunking_fingerprint") == chunk_fp and state.get("embedding_fingerprint") == emb_fp
            and (stored_counts is None or stored_counts.get(doc.document_id, 0) == state.get("chunk_count"))
        )
        if in_sync:
            report.skipped.append(doc.document_id)
            continue
        # A document that failed before it was ever indexed has no stored hash: it is still "new".
        (report.changed if (state and state.get("document_hash")) else report.new).append(doc.document_id)
        if dry_run:
            continue
        try:
            chunks = chunk_document(doc, settings.chunking)
            if not chunks:
                raise ValueError("document produced no chunks (no usable text)")
            vectors = embedder.embed_documents([c.embedding_text() for c in chunks])
            store.upsert_chunks(chunks, vectors, emb_fp)  # 1) write the new version
            store.delete_stale_chunks(doc.document_id, doc_hash)  # 2) then remove old-version leftovers
            docs_state[doc.document_id] = {"document_hash": doc_hash, "chunk_count": len(chunks), "status": "indexed",
                                           "indexed_at": _now(), "chunking_fingerprint": chunk_fp,
                                           "embedding_fingerprint": emb_fp, "source_url": doc.source_url}
            chunks_by_doc[doc.document_id] = [c.to_dict() for c in chunks]
        except Exception as exc:  # keep going; record and retry next run
            report.failed.append({"document_id": doc.document_id, "error": f"{type(exc).__name__}: {exc}"})
            docs_state[doc.document_id] = {**(state or {}), "status": "failed", "error": str(exc), "failed_at": _now()}
            for lst in (report.new, report.changed):
                if doc.document_id in lst:
                    lst.remove(doc.document_id)
        if i % 100 == 0 and not dry_run:
            save_manifest(settings.manifest_path, manifest)

    missing = [d for d in docs_state if d not in input_ids]
    if delete_missing and full_snapshot:
        for d in missing:
            if not dry_run:
                store.delete_document(d)
                docs_state.pop(d, None)
                chunks_by_doc.pop(d, None)
            report.deleted.append(d)
    else:
        report.not_in_input_kept = len(missing)

    if not dry_run:
        manifest["config"] = {
            "embedding_fingerprint": emb_fp, "embedding": {"provider": embedder.config.provider, "model": embedder.config.model,
                                                           "dim": embedder.dim, "query_prefix": embedder.config.query_prefix,
                                                           "document_prefix": embedder.config.document_prefix},
            "chunking_fingerprint": chunk_fp,
            "chunking": {"target_tokens": settings.chunking.target_tokens, "overlap_tokens": settings.chunking.overlap_tokens,
                         "version": settings.chunking.version},
            "collection": settings.collection_name,
            "qdrant": settings.qdrant_url or str(settings.qdrant_path),
        }
        manifest["last_run"] = {"finished_at": _now(), "summary": report.summary(), "failures": report.failed,
                                "review_required": report.review_required, "input_documents": len(documents)}
        save_manifest(settings.manifest_path, manifest)
        if update_chunks_file:
            write_chunks_file(settings.chunks_path, {k: v for k, v in chunks_by_doc.items() if v})
    return report


def build_store(settings: Settings) -> QdrantVectorStore:
    return QdrantVectorStore(settings.collection_name, path=str(settings.qdrant_path),
                             url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def main(argv: list[str] | None = None) -> int:
    from src.ingestion.validate_dataset import validate_jsonl, write_outputs

    settings = Settings.from_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=settings.documents_path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, help="only process the first N usable documents (for quick trials)")
    ap.add_argument("--delete-missing", action="store_true", help="delete indexed documents absent from the input (needs --full-snapshot)")
    ap.add_argument("--full-snapshot", action="store_true", help="assert the input file is the COMPLETE corpus")
    ap.add_argument("--rebuild-collection", action="store_true", help="drop and rebuild only the configured collection")
    args = ap.parse_args(argv)
    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}")
        return 2
    if args.limit and args.delete_missing:
        print("ERROR: --limit cannot be combined with --delete-missing")
        return 2

    result = validate_jsonl(args.input)
    write_outputs(result, settings.validation_report_path, settings.validation_review_path)
    docs = result.documents[: args.limit] if args.limit else result.documents
    print(f"Validated: {len(result.documents)} usable, {len(result.review)} need review (see {settings.validation_review_path.name})")

    embedder = create_embedding_provider(settings.embedding)
    store = build_store(settings)
    try:
        t0 = time.time()
        report = ingest_documents(docs, settings, embedder, store, review=result.review,
                                  delete_missing=args.delete_missing, full_snapshot=args.full_snapshot,
                                  rebuild_collection=args.rebuild_collection, dry_run=args.dry_run)
        print(f"Ingestion summary: {report.summary()}  ({time.time() - t0:.1f}s)")
        for f in report.failed[:10]:
            print("  FAILED:", f)
        if not args.dry_run:
            print(f"Points in collection '{settings.collection_name}': {store.count()}")
    except ConfigMismatchError as exc:
        print(f"ERROR: {exc}")
        return 3
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
