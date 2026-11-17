from src.rag.answer_generation import AnswerGenerator, LLMProvider, ungrounded_numbers
from src.rag.schema import Evidence, RetrievalResult


def ev(text, score=0.9):
    return Evidence(rank=1, score=score, chunk_id="c#0", document_id="d", title="T", source_url="https://x", heading_path="", text=text)


class Fixed(LLMProvider):
    def __init__(self, reply): self.reply = reply
    def generate(self, s, u): return self.reply


def test_ungrounded_numbers_detects_invented_figures():
    e = [ev("You need 10 qualifying years. The full rate is £230.25 a week and rises by 4.1%.")]
    assert ungrounded_numbers("You need 10 qualifying years [1].", e) == []
    assert ungrounded_numbers("The rate is £230.25 a week and rises 4.1% [1].", e) == []
    assert ungrounded_numbers("You need 12 qualifying years [1].", e) == ["12"]
    assert ungrounded_numbers("It is £999 a week [1].", e) == ["£999"]
    assert ungrounded_numbers("See source [1] and point 3.", e) == []  # markers and single digits ignored


def test_generated_answer_with_invented_number_is_rejected():
    r = RetrievalResult("q", "ok", [ev("You need 10 qualifying years.")])
    ans = AnswerGenerator(Fixed("You need 35 qualifying years [1]."), min_score=0.1).answer("q", r)
    assert ans.status == "ungrounded" and "35" in " ".join(ans.notes) and ans.mode == "extractive"
    ok = AnswerGenerator(Fixed("You need 10 qualifying years [1]."), min_score=0.1).answer("q", r)
    assert ok.status == "ok" and ok.mode == "generated"


def test_number_check_can_be_switched_off():
    r = RetrievalResult("q", "ok", [ev("You need 10 qualifying years.")])
    assert AnswerGenerator(Fixed("You need 35 years [1]."), min_score=0.1, check_numbers=False).answer("q", r).status == "ok"


def test_relevance_judge_can_force_a_refusal():
    r = RetrievalResult("q", "ok", [ev("Pension Credit text", 0.9)])
    ans = AnswerGenerator(Fixed("x [1]"), min_score=0.1, judge=Fixed("NO")).answer("q", r)
    assert ans.status == "insufficient_evidence" and ans.citations == []
    assert AnswerGenerator(Fixed("x [1]"), min_score=0.1, judge=Fixed("YES")).answer("q", r).status == "ok"


def test_judge_failure_falls_back_to_threshold():
    class Down(LLMProvider):
        def generate(self, s, u): raise TimeoutError()
    r = RetrievalResult("q", "ok", [ev("text", 0.9)])
    assert AnswerGenerator(None, min_score=0.1, judge=Down()).answer("q", r).status == "ok"


def test_honest_model_refusal_is_accepted_not_replaced():
    r = RetrievalResult("q", "ok", [ev("Unrelated pension text.")])
    ans = AnswerGenerator(Fixed("The evidence does not contain this, so I can't answer."), min_score=0.1).answer("q", r)
    assert ans.status == "insufficient_evidence" and ans.mode == "generated" and ans.citations == []
    assert "does not contain" in ans.text
    still_bad = AnswerGenerator(Fixed("You need 10 years."), min_score=0.1).answer("q", r)  # no citation, not a refusal
    assert still_bad.status == "ungrounded"


def test_model_refusal_without_citations_is_kept_not_rejected():
    r = RetrievalResult("q", "ok", [ev("Pension Credit text")])
    ans = AnswerGenerator(Fixed("The evidence provided does not say what this will be, so I can't answer it."), min_score=0.1).answer("q", r)
    assert ans.status == "insufficient_evidence" and ans.citations == [] and ans.mode == "generated"
    assert "does not say" in ans.text and "https://x" not in ans.text  # the refusal is shown, not quoted passages
    assert ans.raw_reply.startswith("The evidence provided")


def test_long_uncited_reply_with_a_refusal_phrase_is_still_rejected_and_kept_for_inspection():
    r = RetrievalResult("q", "ok", [ev("Pension Credit text")])
    long_reply = "You need ten years. " * 40 + "I could not find more."
    ans = AnswerGenerator(Fixed(long_reply), min_score=0.1).answer("q", r)
    assert ans.status == "ungrounded" and ans.raw_reply == long_reply and "did not pass our citation check" in " ".join(ans.notes)


def test_fallback_header_and_disclaimer_are_honest():
    r = RetrievalResult("q", "ok", [ev("Some passage")])
    assert "no language model was used" in AnswerGenerator(None, min_score=0.1).answer("q", r).text
    rejected = AnswerGenerator(Fixed("no citations here"), min_score=0.1).answer("q", r)
    assert rejected.status == "ungrounded" and "no language model" not in rejected.text
    own = "You need 10 years [1]. Note: this is general information, not personal financial advice."
    text = AnswerGenerator(Fixed(own), min_score=0.1).answer("q", RetrievalResult("q", "ok", [ev("You need 10 years.")])).text
    assert text.count("general information") == 1  # the model's own disclaimer is not repeated


def test_phantom_citation_stripped_preserving_valid_answer():
    # Model cites real passage [1], but also hallucinates phantom [6] on note
    r = RetrievalResult("q", "ok", [ev("You need 10 years.")])  # only 1 passage
    reply = "You need 10 years [1]. Note: this is general information [6]."
    ans = AnswerGenerator(Fixed(reply), min_score=0.1).answer("q", r)
    assert ans.status == "ok" and ans.mode == "generated"
    assert "[1]" in ans.text and "[6]" not in ans.text
    assert len(ans.citations) == 1

