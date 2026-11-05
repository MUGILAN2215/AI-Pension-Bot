"""Structure-aware document chunking.

How it works (see README "Chunking" for the long version):
1. The text is parsed line by line into *units*: headings, paragraphs, list items,
   table rows. Heading context (e.g. "Eligibility > Your National Insurance record")
   is tracked for every unit.
2. Over-long units are split at sentence boundaries (then at word boundaries as a last resort).
3. Units are packed greedily into chunks of about `target_tokens`. A chunk never ends on a
   dangling heading. When a table is split, its header row is repeated so rows stay readable.
4. Each new chunk starts with up to `overlap_tokens` of the previous chunk's trailing units
   (never crossing a heading boundary).

Token counts are an *approximation* (words + punctuation marks); no tokenizer download is needed.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.rag.config import ChunkingConfig, Settings
from src.rag.schema import Chunk, Document, sha256_text

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(\S.*)$")
_LIST_RE = re.compile(r"^(?:[-*•]\s+|\d{1,3}[.)]\s+)")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9£\"“‘(\[])")
_TABLE_START, _TABLE_END = "[TABLE]", "[/TABLE]"


def count_tokens(text: str) -> int:
    """Approximate token count: words and punctuation marks."""
    return len(_TOKEN_RE.findall(text))


def normalise_whitespace(text: str) -> str:
    """Light cleanup only: unify line endings, NBSP/tabs to spaces, trim line ends, collapse spaces."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ").replace("\t", " ")
    return "\n".join(re.sub(r" {2,}", " ", ln).strip() for ln in text.split("\n"))


@dataclass
class _Unit:
    text: str
    kind: str  # heading | text | list | table_row | table_marker
    heading_path: str
    table_header: str | None = None
    tokens: int = 0

    def __post_init__(self) -> None:
        self.tokens = count_tokens(self.text)


def _split_long_text(text: str, limit: int) -> list[str]:
    """Split text longer than `limit` tokens at sentence boundaries, then word boundaries."""
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    sentences: list[str] = []
    for sentence in _SENTENCE_RE.split(text):
        if count_tokens(sentence) <= limit:
            sentences.append(sentence)
        else:  # a single huge "sentence": cut on words
            words = sentence.split(" ")
            buf: list[str] = []
            for w in words:
                buf.append(w)
                if count_tokens(" ".join(buf)) >= limit:
                    sentences.append(" ".join(buf))
                    buf = []
            if buf:
                sentences.append(" ".join(buf))
    for s in sentences:
        t = count_tokens(s)
        if current and current_tokens + t > limit:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(s)
        current_tokens += t
    if current:
        pieces.append(" ".join(current))
    return pieces


def _parse_units(text: str, limit: int) -> list[_Unit]:
    units: list[_Unit] = []
    stack: dict[int, str] = {}
    in_table = False
    table_header: str | None = None

    def path() -> str:
        return " > ".join(stack[k] for k in sorted(stack))

    def add(line: str, kind: str, header: str | None = None) -> None:
        if count_tokens(line) <= limit:
            units.append(_Unit(line, kind, path(), header))
        else:
            for piece in _split_long_text(line, limit):
                units.append(_Unit(piece, kind, path(), header))

    for line in normalise_whitespace(text).split("\n"):
        if not line:
            continue
        m = _HEADING_RE.match(line)
        if m:
            in_table, table_header = False, None
            level = len(m.group(1))
            for k in [k for k in stack if k >= level]:
                del stack[k]
            stack[level] = m.group(2).strip()
            units.append(_Unit(line, "heading", path()))
        elif line == _TABLE_START:
            in_table, table_header = True, None
            units.append(_Unit(line, "table_marker", path()))
        elif line == _TABLE_END:
            in_table, table_header = False, None
            units.append(_Unit(line, "table_marker", path()))
        elif " | " in line or line.startswith("|"):
            if table_header is None:
                # Only a short first row is a usable header; a huge "row" is not repeated.
                table_header = line if count_tokens(line) <= limit // 5 else ""
            in_table = True
            add(line, "table_row", table_header or None)
        else:
            in_table, table_header = False, None
            add(line, "list" if _LIST_RE.match(line) else "text")
    return units


def _chunk_tokens(units: list[_Unit]) -> int:
    total = sum(u.tokens for u in units)
    first = units[0] if units else None
    if first and first.kind == "table_row" and first.table_header and first.text != first.table_header:
        total += count_tokens(first.table_header)
    return total


def _render(units: list[_Unit]) -> str:
    parts: list[str] = []
    first = units[0]
    if first.kind == "table_row" and first.table_header and first.text != first.table_header:
        parts.append(first.table_header)  # repeat header so continued tables stay readable
    for i, u in enumerate(units):
        if u.kind == "heading" and (parts or i > 0):
            parts.append("")  # blank line before a heading for readability
        parts.append(u.text)
    return "\n".join(parts).strip()


def _overlap_tail(units: list[_Unit], overlap_tokens: int) -> list[_Unit]:
    tail: list[_Unit] = []
    total = 0
    for u in reversed(units):
        if u.kind in ("heading", "table_marker") or total + u.tokens > overlap_tokens:
            break
        tail.insert(0, u)
        total += u.tokens
    return tail


def chunk_document(doc: Document, cfg: ChunkingConfig | None = None) -> list[Chunk]:
    """Split one document into chunks. Returns [] for documents without usable text."""
    cfg = cfg or ChunkingConfig()
    units = _parse_units(doc.text or "", cfg.target_tokens)
    if not any(u.kind in ("text", "list", "table_row") for u in units):
        return []

    groups: list[tuple[list[_Unit], int]] = []  # (units, number of leading overlap units)
    cur: list[_Unit] = []
    n_overlap = 0
    for u in units:
        is_close = u.kind == "table_marker" and u.text == _TABLE_END
        if cur and len(cur) > n_overlap and not is_close and _chunk_tokens(cur + [u]) > cfg.target_tokens:
            trailing: list[_Unit] = []
            while len(cur) > n_overlap + 1 and cur[-1].kind == "heading":
                trailing.insert(0, cur.pop())  # never end a chunk on a heading
            groups.append((cur, n_overlap))
            tail = _overlap_tail(cur, cfg.overlap_tokens)
            while tail and _chunk_tokens(tail + trailing + [u]) > cfg.target_tokens:
                tail.pop(0)
            cur = tail + trailing
            n_overlap = len(tail)
        cur.append(u)
    if len(cur) > n_overlap:
        groups.append((cur, n_overlap))

    doc_hash = doc.content_hash()
    chunks: list[Chunk] = []
    for idx, (g_units, n_ov) in enumerate(groups):
        new_units = g_units[n_ov:]
        anchor = next((u for u in new_units if u.kind != "heading"), new_units[0])
        path = anchor.heading_path
        text = _render(g_units)
        chunks.append(Chunk(
            chunk_id=f"{doc.document_id}#{idx:05d}",
            document_id=doc.document_id,
            chunk_index=idx,
            chunk_count=len(groups),
            text=text,
            heading_path=path,
            heading=path.split(" > ")[-1] if path else None,
            title=doc.title,
            source_url=doc.source_url,
            token_count=count_tokens(text),
            overlap_tokens=sum(u.tokens for u in g_units[:n_ov]),
            content_hash=sha256_text(text),
            document_hash=doc_hash,
            parent_source_url=doc.parent_source_url,
            section=doc.section,
            publisher=doc.publisher,
            source=doc.source,
            source_type=doc.source_type,
            category=doc.category,
            original_format=doc.original_format,
            jurisdiction=doc.jurisdiction,
            language=doc.language,
            publication_date=doc.publication_date,
            last_updated_date=doc.last_updated_date,
            date=doc.date,
            licence=doc.licence,
            review_flags=list(doc.review_flags),
            provenance=list(doc.provenance),
            chunking={"target_tokens": cfg.target_tokens, "overlap_tokens": cfg.overlap_tokens,
                      "version": cfg.version, "fingerprint": cfg.fingerprint()},
        ))
    return chunks


# ---- chunks.jsonl helpers -------------------------------------------------------------

def read_chunks_file(path: Path) -> dict[str, list[dict]]:
    """Load chunks.jsonl grouped by document_id (empty dict if the file is missing)."""
    out: dict[str, list[dict]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    out.setdefault(rec["document_id"], []).append(rec)
    return out


def write_chunks_file(path: Path, by_doc: dict[str, list[dict]]) -> int:
    """Atomically write chunks.jsonl (sorted by document id then chunk index). Returns line count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for doc_id in sorted(by_doc):
            for rec in sorted(by_doc[doc_id], key=lambda r: r["chunk_index"]):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    tmp.replace(path)
    return n


def main(argv: list[str] | None = None) -> int:
    from src.ingestion.validate_dataset import validate_jsonl

    settings = Settings.from_env()
    ap = argparse.ArgumentParser(description="Generate chunks.jsonl from the documents file.")
    ap.add_argument("--input", type=Path, default=settings.documents_path)
    ap.add_argument("--output", type=Path, default=settings.chunks_path)
    ap.add_argument("--target-tokens", type=int, default=settings.chunking.target_tokens)
    ap.add_argument("--overlap-tokens", type=int, default=settings.chunking.overlap_tokens)
    args = ap.parse_args(argv)
    cfg = ChunkingConfig(target_tokens=args.target_tokens, overlap_tokens=args.overlap_tokens)
    result = validate_jsonl(args.input)
    by_doc = {d.document_id: [c.to_dict() for c in chunk_document(d, cfg)] for d in result.documents}
    n = write_chunks_file(args.output, {k: v for k, v in by_doc.items() if v})
    sizes = sorted(c["token_count"] for v in by_doc.values() for c in v)
    no_chunks = [k for k, v in by_doc.items() if not v]
    print(f"Documents: {len(result.documents)}  chunks written: {n}  documents without chunks: {len(no_chunks)}")
    if sizes:
        print(f"Chunk tokens (approx) min/median/max: {sizes[0]}/{sizes[len(sizes)//2]}/{sizes[-1]}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
