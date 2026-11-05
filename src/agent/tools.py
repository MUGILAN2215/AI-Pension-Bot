"""Tools the agent can use. Each tool is plain Python: it is tested and never "imagined" by a language model.

1. search_pension_docs  - the RAG retriever
2. ni_years_calculator  - deterministic rule check for the new State Pension qualifying years
3. get_source_document  - fetch the full text of a cited document
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.rag.retriever import Retriever
from src.rag.schema import RetrievalResult

# Rules below were checked against the indexed GOV.UK text ("you'll need 10 qualifying years ... to get any new State Pension",
# "If your National Insurance record started after April 2016 you will need 35 qualifying years to get the full rate").
MIN_YEARS_ANY_NEW_STATE_PENSION = 10
FULL_NEW_STATE_PENSION_YEARS = 35


def search_pension_docs(retriever: Retriever, query: str, top_k: int | None = None) -> RetrievalResult:
    return retriever.retrieve(query, top_k)


@dataclass
class NIYearsResult:
    qualifying_years: int
    gets_any_new_state_pension: bool
    gets_full_new_state_pension: bool
    years_still_needed_for_any: int
    years_still_needed_for_full: int
    message: str


def ni_years_calculator(qualifying_years: int) -> NIYearsResult:
    """Compare a number of qualifying years with the new State Pension thresholds.

    Deliberately limited: it assumes the person's National Insurance record started after April 2016 and does NOT compute a pound
    amount (rates change, and older records have a "starting amount" with transitional rules). Use the official State Pension forecast.
    """
    if not isinstance(qualifying_years, int) or isinstance(qualifying_years, bool) or not 0 <= qualifying_years <= 60:
        raise ValueError("qualifying_years must be a whole number between 0 and 60")
    any_ = qualifying_years >= MIN_YEARS_ANY_NEW_STATE_PENSION
    full = qualifying_years >= FULL_NEW_STATE_PENSION_YEARS
    need_any = max(0, MIN_YEARS_ANY_NEW_STATE_PENSION - qualifying_years)
    need_full = max(0, FULL_NEW_STATE_PENSION_YEARS - qualifying_years)
    if full:
        msg = f"With {qualifying_years} qualifying years you meet the {FULL_NEW_STATE_PENSION_YEARS}-year requirement for the full rate of new State Pension (if your National Insurance record started after April 2016)."
    elif any_:
        msg = (f"With {qualifying_years} qualifying years you meet the {MIN_YEARS_ANY_NEW_STATE_PENSION}-year minimum for some new State Pension, "
               f"but you would need {need_full} more for the full rate ({FULL_NEW_STATE_PENSION_YEARS}). Your exact amount depends on your record; check your State Pension forecast.")
    else:
        msg = (f"With {qualifying_years} qualifying years you do not yet meet the {MIN_YEARS_ANY_NEW_STATE_PENSION}-year minimum for any new State Pension; "
               f"you would need {need_any} more.")
    return NIYearsResult(qualifying_years, any_, full, need_any, need_full, msg + " This is a general rule check, not a forecast of your actual pension.")


_YEARS_RE = re.compile(r"\b(\d{1,2})\s*(?:qualifying\s+)?years?\b", re.I)


def extract_years_question(text: str) -> int | None:
    """Return N when the question asks about N qualifying years / National Insurance years, else None."""
    low = text.lower()
    # 1. Natural conversational phrasing: "I have 20 years, how many more do I need?", "with 25 years"
    m_direct = re.search(r"\b(?:i have|i've got|with|paid|got|built up|completed)\s+(\d{1,2})\s*years?\b", low)
    if m_direct and any(k in low for k in ("need", "more", "left", "pension", "retire", "get", "eligible", "qualif", "state", "full")):
        return int(m_direct.group(1))

    # 2. Formal phrasing: contains "qualifying", "national insurance", or "ni"
    if "qualifying" in low or "national insurance" in low or re.search(r"\bni\b", low):
        # Questions about rules, part-time work, earnings thresholds, or whether something counts are knowledge questions, not calculator:
        if any(k in low for k in ("does ", "do ", "how ", "what counts", "part-time", "part time", "low earnings", "credits", "voluntary", "gaps", "can i", "is it possible")):
            if not re.search(r"\b(?:i have|i've got|with|paid|got)\s+\d+", low):
                return None
        m = _YEARS_RE.search(text)
        return int(m.group(1)) if m else None

    return None


class DocumentLibrary:
    """Read-only access to the full text of documents (loaded lazily from the cleaned documents file)."""

    def __init__(self, path: Path):
        self.path, self._docs = path, None

    def _load(self) -> dict:
        if self._docs is None:
            self._docs = {}
            if self.path.exists():
                with self.path.open(encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            r = json.loads(line)
                            self._docs[r["doc_id"]] = r
        return self._docs

    def get(self, document_id: str, max_chars: int = 6000) -> dict | None:
        r = self._load().get(document_id)
        if not r:
            return None
        return {"document_id": document_id, "title": r.get("title"), "source_url": r.get("source_url"), "licence": r.get("licence"),
                "last_updated_date": r.get("last_updated_date") or r.get("date"), "text": r.get("text", "")[:max_chars],
                "truncated": len(r.get("text", "")) > max_chars}


def get_source_document(library: DocumentLibrary, document_id: str) -> dict | None:
    return library.get(document_id)
