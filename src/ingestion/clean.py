"""Data cleaning step: documents_merged.jsonl -> documents_clean.jsonl (+ cleaning_report.json).

What it does (and deliberately does NOT do):
* EXCLUDES documents that are mostly SVG/markup noise (found by counting <svg>/<path>/viewBox markers). Each exclusion is logged
  with its id, URL and reason; the source files are never modified.
* REPORTS (does not change) groups of documents with identical text, documents with missing licence/dates, and review flags.
  Identical-text groups are collapsed at *retrieval* time, so every original URL stays citable.
* Text is never rewritten, so the original meaning is preserved.

Run:  python -m src.ingestion.clean
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.ingestion.validate_dataset import validate_jsonl
from src.rag.config import PROJECT_ROOT, Settings
from src.rag.schema import sha256_text

MARKUP_MARKERS = ("<svg", "<path", "viewBox")


def markup_noise_count(text: str) -> int:
    return sum(text.count(m) for m in MARKUP_MARKERS)


def clean_documents(in_path: Path, out_path: Path, report_path: Path, noise_threshold: int = 2) -> dict:
    result = validate_jsonl(in_path)
    kept, excluded = [], []
    for doc in result.documents:
        n = markup_noise_count(doc.text)
        if n >= noise_threshold:
            excluded.append({"document_id": doc.document_id, "source_url": doc.source_url, "reason": f"markup_noise ({n} SVG/markup markers)"})
        else:
            kept.append(doc)
    groups: dict[str, list[str]] = defaultdict(list)
    for d in kept:
        groups[sha256_text(" ".join(d.text.split()))].append(d.document_id)
    dup_groups = [ids for ids in groups.values() if len(ids) > 1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for d in kept:
            fh.write(json.dumps({"doc_id": d.document_id, **{k: v for k, v in d.to_dict().items() if k not in ("document_id", "metadata")}, **d.metadata}, ensure_ascii=False) + "\n")
    tmp.replace(out_path)
    report = {
        "input": str(in_path), "output": str(out_path),
        "input_usable_documents": len(result.documents), "kept": len(kept), "excluded": excluded,
        "sent_to_validation_review": len(result.review),
        "identical_text_groups": len(dup_groups), "documents_in_identical_text_groups": sum(len(g) for g in dup_groups),
        "largest_identical_group": max((len(g) for g in dup_groups), default=0),
        "missing_licence": sum(1 for d in kept if not d.licence),
        "missing_all_dates": sum(1 for d in kept if not (d.publication_date or d.last_updated_date or d.date)),
        "review_flags": dict(Counter(f for d in kept for f in d.review_flags)),
        "note": "Text is unchanged. Duplicates are kept and collapsed at retrieval (Retriever(dedupe=True)).",
    }
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/processed/documents_merged.jsonl")
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/documents_clean.jsonl")
    ap.add_argument("--report", type=Path, default=PROJECT_ROOT / "data/processed/cleaning_report.json")
    args = ap.parse_args(argv)
    r = clean_documents(args.input, args.output, args.report)
    print(f"kept {r['kept']} of {r['input_usable_documents']}; excluded {len(r['excluded'])}: {[e['document_id'] for e in r['excluded']]}")
    print(f"identical-text groups: {r['identical_text_groups']} (largest {r['largest_identical_group']}); report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
