"""Turn an uploaded file into a Document that the normal ingestion pipeline can index. Admin-only (enforced in main.py).

Accepted: .txt, .md, .pdf (text-based), .docx. Scanned PDFs have no text layer and are rejected with a clear message (no OCR).
"""
from __future__ import annotations

import hashlib
import io
from datetime import date

from src.rag.schema import Document

MAX_BYTES = 10 * 1024 * 1024
ALLOWED = {".txt", ".md", ".pdf", ".docx"}


class UploadError(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    ext = "." + filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in ALLOWED:
        raise UploadError(f"Unsupported file type '{ext or filename}'. Allowed: {', '.join(sorted(ALLOWED))}")
    if len(data) > MAX_BYTES:
        raise UploadError("File is larger than 10 MB")
    if ext in (".txt", ".md"):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UploadError("Text file is not valid UTF-8") from exc
    if ext == ".pdf":
        from pypdf import PdfReader
        try:
            pages = [(p.extract_text() or "").strip() for p in PdfReader(io.BytesIO(data)).pages]
        except Exception as exc:
            raise UploadError(f"Could not read the PDF: {exc}") from exc
        return "\n\n".join(p for p in pages if p)
    from docx import Document as Docx
    try:
        doc = Docx(io.BytesIO(data))
    except Exception as exc:
        raise UploadError(f"Could not read the Word file: {exc}") from exc
    lines: list[str] = []
    for para in doc.paragraphs:
        t = para.text.strip()
        if not t:
            continue
        style = para.style.name if para.style is not None else ""
        if style.startswith("Heading ") and style[8:].isdigit():
            lines.append("#" * min(int(style[8:]), 6) + " " + t)
        else:
            lines.append(t)
    for table in doc.tables:
        for row in table.rows:
            lines.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(lines)


def build_document(filename: str, data: bytes, title: str, source_url: str | None, publisher: str | None, licence: str | None,
                   doc_date: str | None) -> Document:
    title = (title or "").strip()
    if not title:
        raise UploadError("A title is required")
    text = extract_text(filename, data)
    if not text.strip():
        raise UploadError("No readable text found in the file (a scanned PDF needs OCR first)")
    digest = hashlib.sha256((title + "\n" + text).encode("utf-8")).hexdigest()[:12]
    return Document(document_id=f"upload_{digest}", title=title, text=text,
                    source_url=(source_url or "").strip() or f"uploaded-file:{filename}",
                    publisher=(publisher or "").strip() or "Admin upload", licence=(licence or "").strip() or None,
                    date=(doc_date or "").strip() or date.today().isoformat(), source="admin_upload", source_type="admin_upload",
                    original_format=filename.rsplit(".", 1)[-1].lower())
