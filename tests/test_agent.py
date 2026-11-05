import pytest

from src.agent.graph import PensionAgent
from src.agent.tools import DocumentLibrary, extract_years_question, get_source_document, ni_years_calculator
from src.rag.answer_generation import AnswerGenerator, LLMProvider
from src.rag.ingest import ingest_documents
from src.rag.retriever import Retriever
from tests.conftest import write_jsonl


class Fixed(LLMProvider):
    def __init__(self, reply): self.reply, self.seen = reply, []
    def generate(self, s, u):
        self.seen.append(u)
        return self.reply


def test_calculator_thresholds():
    assert not ni_years_calculator(5).gets_any_new_state_pension and ni_years_calculator(5).years_still_needed_for_any == 5
    r = ni_years_calculator(10)
    assert r.gets_any_new_state_pension and not r.gets_full_new_state_pension and r.years_still_needed_for_full == 25
    assert ni_years_calculator(35).gets_full_new_state_pension and ni_years_calculator(40).years_still_needed_for_full == 0
    assert "not a forecast" in r.message and "£" not in r.message  # never invents pound amounts


@pytest.mark.parametrize("bad", [-1, 61, 3.5, "10", True])
def test_calculator_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        ni_years_calculator(bad)


def test_years_extraction():
    assert extract_years_question("I have 12 qualifying years, do I get a pension?") == 12
    assert extract_years_question("I paid National Insurance for 8 years") == 8
    assert extract_years_question("I am 12 years old") is None and extract_years_question("How do I claim Pension Credit?") is None
    assert extract_years_question("Does working part-time or having low earnings still count towards my 35 qualifying years?") is None


def test_document_library(tmp_path):
    p = tmp_path / "d.jsonl"
    write_jsonl(p, [{"doc_id": "a", "title": "A", "source_url": "u", "text": "x" * 10, "licence": "OGL"}])
    lib = DocumentLibrary(p)
    assert get_source_document(lib, "a")["title"] == "A" and get_source_document(lib, "zzz") is None
    assert DocumentLibrary(tmp_path / "missing.jsonl").get("a") is None


@pytest.fixture
def agent_parts(settings, embedder, store, docs):
    ingest_documents(docs, settings, embedder, store)
    return Retriever(embedder, store, top_k=3, mode="hybrid"), settings


def test_agent_search_route_and_citations(agent_parts):
    retriever, s = agent_parts
    res = PensionAgent(retriever, AnswerGenerator(min_score=0.05)).ask("How do I apply for Pension Credit?")
    assert res.route == "search" and res.answer.status == "ok" and res.answer.citations and res.calculator is None
    assert res.steps[-1].startswith("answer: ok")


def test_agent_runs_calculator_and_still_cites(agent_parts):
    retriever, _ = agent_parts
    res = PensionAgent(retriever, AnswerGenerator(min_score=0.05)).ask("I have 12 qualifying years National Insurance, what State Pension do I get?")
    assert res.route == "calculator" and res.calculator["gets_any_new_state_pension"] is True
    assert res.answer.text.startswith("Calculation:") and res.answer.citations


def test_agent_rewrites_follow_ups_with_history(agent_parts):
    retriever, _ = agent_parts
    rewriter = Fixed("Can my spouse claim Pension Credit?")
    history = [{"role": "user", "content": "How does Pension Credit work?"}, {"role": "assistant", "content": "It tops up low income."}]
    res = PensionAgent(retriever, AnswerGenerator(min_score=0.05), rewriter).ask("What about my spouse?", history)
    assert res.rewritten_question == "Can my spouse claim Pension Credit?" and "Pension Credit work" in rewriter.seen[0]


def test_agent_rewrite_failures_fall_back(agent_parts):
    retriever, _ = agent_parts
    history = [{"role": "user", "content": "hi"}]
    class Down(LLMProvider):
        def generate(self, s, u): raise TimeoutError()
    assert PensionAgent(retriever, AnswerGenerator(), Down()).ask("State Pension age", history).rewritten_question == "State Pension age"
    assert PensionAgent(retriever, AnswerGenerator(), Fixed("x" * 400)).ask("State Pension age", history).rewritten_question == "State Pension age"
    assert PensionAgent(retriever, AnswerGenerator()).ask("State Pension age", history).rewritten_question == "State Pension age"  # no rewriter


def test_agent_refuses_unsupported_question(agent_parts):
    retriever, _ = agent_parts
    res = PensionAgent(retriever, AnswerGenerator(min_score=0.2)).ask("What is the best recipe for sourdough bread?")
    assert res.answer.status == "insufficient_evidence" and res.answer.citations == []
