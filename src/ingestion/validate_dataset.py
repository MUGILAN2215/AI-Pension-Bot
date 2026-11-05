"""Validate a documents JSONL file and separate usable records from records needing review.

Nothing is silently dropped: every line is either a usable Document, or appears in the
review output / report with a reason.

Run:  python -m src.ingestion.validate_dataset [--input PATH]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.rag.config import Settings
from src.rag.schema import Document

REQUIRED_FIELDS = ("document_id", "title", "source_url", "text")
OPTIONAL_FIELDS = ("publisher", "source", "licence", "publication_date", "last_updated_date", "date",
                   "category", "source_type", "section")


@dataclass
class ValidationResult:
    documents: list[Document] = field(default_factory=list)  # usable, ready to ingest
    review: list[dict] = field(default_factory=list)  # {line, reason, record}
    report: dict = field(default_factory=dict)


def validate_jsonl(path: str | Path) -> ValidationResult:
    """Parse and validate every line of `path`. The file itself is never modified."""
    path = Path(path)
    result = ValidationResult()
    total_lines = blank = 0
    invalid_json: list[dict] = []
    not_object: list[int] = []
    missing_required: Counter = Counter()
    key_sets: Counter = Counter()
    parsed: list[tuple[int, dict]] = []

    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            total_lines += 1
            if not line.strip():
                blank += 1
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_json.append({"line": line_no, "error": str(exc)})
                result.review.append({"line": line_no, "reason": "invalid_json", "detail": str(exc),
                                      "record": {"raw_line": line.rstrip("\n")}})
                continue
            if not isinstance(rec, dict):
                not_object.append(line_no)
                result.review.append({"line": line_no, "reason": "not_a_json_object", "record": {"raw_line": line.rstrip("\n")}})
                continue
            parsed.append((line_no, rec))
            key_sets[tuple(sorted(rec.keys()))] += 1

    # Convert, check required fields, group by id to find duplicates/conflicts.
    by_id: dict[str, list[tuple[int, dict, Document]]] = defaultdict(list)
    for line_no, rec in parsed:
        doc = Document.from_record(rec)
        problems = [f for f in REQUIRED_FIELDS if not isinstance(getattr(doc, f), str)]
        problems += [f for f in ("document_id", "title", "source_url") if isinstance(getattr(doc, f), str) and not getattr(doc, f).strip()]
        if problems:
            for f in problems:
                missing_required[f] += 1
            result.review.append({"line": line_no, "reason": "missing_required_field", "detail": sorted(set(problems)), "record": rec})
            continue
        by_id[doc.document_id].append((line_no, rec, doc))

    identical_dupes: list[dict] = []
    conflicts: list[dict] = []
    empty_text: list[str] = []
    for doc_id, items in by_id.items():
        hashes = {d.content_hash() for _, _, d in items}
        if len(hashes) > 1:  # same id, different content: unresolved -> review, not ingested
            conflicts.append({"document_id": doc_id, "lines": [ln for ln, _, _ in items]})
            for ln, rec, _ in items:
                result.review.append({"line": ln, "reason": "conflicting_versions_same_id", "record": rec})
            continue
        first_line, _, doc = items[0]
        if len(items) > 1:
            identical_dupes.append({"document_id": doc_id, "lines": [ln for ln, _, _ in items]})
        if not doc.text.strip():
            empty_text.append(doc_id)
            result.review.append({"line": first_line, "reason": "empty_text", "detail": doc.extraction_status,
                                  "record": items[0][1]})
            continue
        result.documents.append(doc)

    # Informational checks on usable documents.
    optional_missing = {f: sum(1 for d in result.documents if getattr(d, f) in (None, "")) for f in OPTIONAL_FIELDS}
    url_groups = Counter(d.source_url.rstrip("/").lower() for d in result.documents)
    flag_counts = Counter(f for d in result.documents for f in d.review_flags)

    result.report = {
        "input_file": str(path),
        "total_lines": total_lines,
        "blank_lines": blank,
        "valid_json_objects": len(parsed),
        "invalid_json_lines": invalid_json,
        "non_object_lines": not_object,
        "missing_required_fields": dict(missing_required),
        "duplicate_ids_identical": identical_dupes,
        "duplicate_ids_conflicting": conflicts,
        "empty_text_documents": empty_text,
        "usable_documents": len(result.documents),
        "sent_to_review": len(result.review),
        "distinct_key_sets": len(key_sets),
        "optional_fields_missing_in_usable_docs": optional_missing,
        "urls_shared_by_multiple_documents": sum(1 for v in url_groups.values() if v > 1),
        "existing_review_flags_on_usable_docs": dict(flag_counts),
        "note": "Usable documents with existing review_flags are still ingested; the flags travel with every chunk.",
    }
    return result


def write_outputs(result: ValidationResult, report_path: Path, review_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.report, indent=2, ensure_ascii=False), encoding="utf-8")
    with review_path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in result.review:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=settings.documents_path)
    ap.add_argument("--report", type=Path, default=settings.validation_report_path)
    ap.add_argument("--review", type=Path, default=settings.validation_review_path)
    args = ap.parse_args(argv)
    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}\nPlace documents_merged.jsonl in data/processed/ or pass --input.")
        return 2
    result = validate_jsonl(args.input)
    write_outputs(result, args.report, args.review)
    r = result.report
    print(f"Lines: {r['total_lines']}  valid objects: {r['valid_json_objects']}  invalid JSON: {len(r['invalid_json_lines'])}")
    print(f"Usable documents: {r['usable_documents']}  sent to review: {r['sent_to_review']}")
    print(f"Duplicate ids (identical): {len(r['duplicate_ids_identical'])}  conflicting: {len(r['duplicate_ids_conflicting'])}  empty text: {len(r['empty_text_documents'])}")
    print(f"Report: {args.report}\nReview: {args.review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
