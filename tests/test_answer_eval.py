from scripts.answer_eval import evaluate_answers
from src.rag.answer_generation import LLMProvider
from src.rag.ingest import ingest_documents
from src.rag.retriever import Retriever
from tests.conftest import STATE_PENSION, PENSION_CREDIT, make_doc

SNIP = "You need 10 qualifying years on your National Insurance record to get any new State Pension."


class Scripted(LLMProvider):
    def __init__(self, reply): self.reply = reply
    def generate(self, s, u): return self.reply


def eval_set(doc_id="sp"):
    return {"questions": [
        {"id": "q1", "question": "How many qualifying years of National Insurance for the new State Pension?", "type": "answerable",
         "evidence_snippet": SNIP, "relevant_document_ids": [doc_id]},
        {"id": "u1", "question": "What is the best recipe for sourdough bread?", "type": "unanswerable", "absent_terms": ["sourdough"]}]}


def run(settings, embedder, store, reply):
    ingest_documents([STATE_PENSION, PENSION_CREDIT], settings, embedder, store)
    return evaluate_answers(Scripted(reply), Retriever(embedder, store, top_k=3, mode="hybrid"), eval_set(), min_score=0.2)


def test_good_cited_answer_scores_well(settings, embedder, store):
    r = run(settings, embedder, store, "You need 10 qualifying years [1].")
    m = r["metrics_on_fair_questions"]
    assert r["counts"]["answerable_where_evidence_had_the_answer"] == 1
    assert m["answered_not_refused"] == m["has_citation"] == m["citations_valid"] == m["cites_right_document"] == m["no_invented_numbers"] == 1.0
    assert r["refusal_on_unanswerable"] == {"refused": 1, "of": 1}  # off-topic stopped by the score gate


def test_bad_answers_are_caught(settings, embedder, store):
    no_cite = run(settings, embedder, store, "You need 10 qualifying years.")["metrics_on_fair_questions"]
    assert no_cite["has_citation"] == 0.0 and no_cite["reply_rejected_for_bad_citations"] == 1.0
    invented = run(settings, embedder, store, "You need 35 qualifying years [1].")["metrics_on_fair_questions"]
    assert invented["no_invented_numbers"] == 0.0 and invented["has_citation"] == 1.0
    refusing = run(settings, embedder, store, "The evidence does not contain this.")["metrics_on_fair_questions"]
    assert refusing["answered_not_refused"] == 0.0


def test_key_numbers_review_sheet_and_compare(settings, embedder, store, tmp_path):
    from scripts.answer_eval import _key_numbers_present, compare, write_review_sheet
    assert _key_numbers_present(SNIP, "You need 10 years [1].") is True and _key_numbers_present(SNIP, "You need ten years [1].") is False
    assert _key_numbers_present("No figures in this sentence.", "anything") is None
    good = run(settings, embedder, store, "You need 10 qualifying years [1].")
    bad = run(settings, embedder, store, "You need 12 qualifying years [1].")
    assert good["metrics_on_fair_questions"]["key_numbers_present"] == 1.0 and bad["metrics_on_fair_questions"]["key_numbers_present"] == 0.0
    sheet = tmp_path / "sheet.md"
    write_review_sheet(good, sheet, 4)
    text = sheet.read_text(encoding="utf-8")
    assert "Verdict: [ ] correct" in text and "sourdough" in text and "You need 10 qualifying years [1]." in text
    table = compare(good, bad, "good", "bad")
    assert "key_numbers_present" in table and "refused unanswerable" in table
