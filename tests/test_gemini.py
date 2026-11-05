import math
from types import SimpleNamespace

import pytest

from src.rag.config import EmbeddingConfig
from src.rag.gemini import (CallBudget, GeminiEmbedder, GeminiError, GeminiLLM, GeminiSettings, LLMReranker, call_with_retry)
from src.rag.schema import Evidence


def settings(tmp_path, **kw):
    base = dict(api_key="k", model="m", embedding_model="emb-x", embed_dim=None, min_interval=0.0, daily_cap=100, cache_path=tmp_path / "c.sqlite")
    base.update(kw)
    return GeminiSettings(**base)


class FakeModels:
    def __init__(self):
        self.embed_calls, self.gen_calls = [], []
        self.reply = "An answer [1]."
    def embed_content(self, *, model, contents, config):
        self.embed_calls.append((model, list(contents), config.task_type))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[float(len(c) % 7 + 1), 2.0, 3.0, 4.0]) for c in contents])
    def generate_content(self, *, model, contents, config):
        self.gen_calls.append((model, contents, config.system_instruction))
        return SimpleNamespace(text=self.reply)


def fake_client():
    return SimpleNamespace(models=FakeModels())


def test_embedder_task_types_dimension_and_normalisation(tmp_path):
    c = fake_client()
    e = GeminiEmbedder(EmbeddingConfig(provider="gemini"), settings(tmp_path), c)
    assert e.dim == 4
    docs = e.embed_documents(["State Pension age", "Pension Credit"])
    q = e.embed_query("State Pension age")
    assert all(abs(math.sqrt(sum(x * x for x in v)) - 1) < 1e-9 for v in docs + [q])
    kinds = {call[2] for call in c.models.embed_calls}
    assert {"RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"} <= kinds
    assert all(call[0] == "emb-x" for call in c.models.embed_calls)


def test_embedding_cache_avoids_repeat_calls(tmp_path):
    c = fake_client()
    e = GeminiEmbedder(EmbeddingConfig(provider="gemini"), settings(tmp_path), c)
    e.embed_documents(["same text"])
    n = len(c.models.embed_calls)
    e.embed_documents(["same text"])
    assert len(c.models.embed_calls) == n


def test_embedder_batches_requests(tmp_path):
    c = fake_client()
    e = GeminiEmbedder(EmbeddingConfig(provider="gemini"), settings(tmp_path), c, batch=3)
    before = len(c.models.embed_calls)
    e.embed_documents([f"text number {i}" for i in range(7)], batch_size=100)
    assert len(c.models.embed_calls) - before == 3  # 7 texts in batches of 3


def test_fingerprint_depends_on_model(tmp_path):
    a = GeminiEmbedder(EmbeddingConfig(provider="gemini"), settings(tmp_path, embedding_model="m1"), fake_client(), use_cache=False)
    b = GeminiEmbedder(EmbeddingConfig(provider="gemini"), settings(tmp_path, embedding_model="m2"), fake_client(), use_cache=False)
    assert a.fingerprint() != b.fingerprint()


def test_missing_key_and_model_give_clear_errors(tmp_path):
    with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
        GeminiEmbedder(EmbeddingConfig(provider="gemini"), settings(tmp_path, api_key=None, embed_dim=4))
    with pytest.raises(GeminiError, match="GEMINI_MODEL"):
        GeminiLLM(settings(tmp_path, model=None), client=fake_client())


def test_llm_generate_passes_system_prompt(tmp_path):
    c = fake_client()
    assert GeminiLLM(settings(tmp_path), c).generate("be strict", "question?") == "An answer [1]."
    assert c.models.gen_calls[0][2] == "be strict"
    c.models.reply = ""
    with pytest.raises(GeminiError):
        GeminiLLM(settings(tmp_path), c).generate("s", "u")


def test_retry_on_rate_limit_then_succeeds():
    class Err(Exception):
        code = 429
    attempts, waits = [], []
    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise Err("slow down")
        return "ok"
    assert call_with_retry(flaky, CallBudget(0, 100), sleep=waits.append) == "ok" and len(attempts) == 3 and waits == [10.0, 20.0]


def test_non_retryable_errors_and_daily_cap():
    class Bad(Exception):
        code = 400
    with pytest.raises(GeminiError):
        call_with_retry(lambda: (_ for _ in ()).throw(Bad("bad request")), CallBudget(0, 100), sleep=lambda s: None)
    b = CallBudget(0, 2)
    b.wait(); b.wait()
    with pytest.raises(GeminiError, match="daily cap"):
        b.wait()


def _ev(i):
    return Evidence(rank=i, score=0.5, chunk_id=f"c{i}", document_id=f"d{i}", title=f"T{i}", source_url="u", heading_path="", text=f"text {i}")


def test_llm_reranker_reorders_and_falls_back():
    class L:
        def __init__(self, reply): self.reply = reply
        def generate(self, s, u): return self.reply
    ev = [_ev(1), _ev(2), _ev(3)]
    assert [e.chunk_id for e in LLMReranker(L('{"scores": [1, 9, 5]}')).rerank("q", ev)] == ["c2", "c3", "c1"]
    for bad in ("not json", '{"scores": [1]}', '{"scores": ["a","b","c"]}'):
        assert LLMReranker(L(bad)).rerank("q", ev) == ev


def test_daily_cap_survives_restart_and_resets_next_day(tmp_path):
    f = tmp_path / "usage.json"
    day = {"d": "2026-10-27"}
    mk = lambda: CallBudget(0, 2, usage_path=f, today=lambda: day["d"])
    a = mk(); a.wait(); a.wait()
    with pytest.raises(GeminiError, match="daily cap"):
        mk().wait()                      # a NEW run (new object) still sees the 2 calls already used today
    day["d"] = "2026-10-28"
    mk().wait()                          # next day: allowed again


def test_vector_size_is_remembered_so_restart_needs_no_api_call(tmp_path):
    c1 = fake_client()
    GeminiEmbedder(EmbeddingConfig(provider="gemini"), settings(tmp_path), c1)
    assert len(c1.models.embed_calls) == 1                      # first start learns the size (one call)
    c2 = fake_client()
    e = GeminiEmbedder(EmbeddingConfig(provider="gemini"), settings(tmp_path), c2)
    assert e.dim == 4 and c2.models.embed_calls == []           # second start: no call


def test_embedding_cache_thread_safety(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from src.rag.gemini import EmbeddingCache
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    cache.put("k1", [1.0, 2.0])

    def worker(i):
        cache.put(f"worker_{i}", [float(i)])
        return cache.get("k1")

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(worker, range(20)))
    assert all(r == [1.0, 2.0] for r in results)
    assert cache.get("worker_5") == [5.0]

