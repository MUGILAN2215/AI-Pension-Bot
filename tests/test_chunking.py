from src.rag.chunking import chunk_document, count_tokens
from src.rag.config import ChunkingConfig
from tests.conftest import make_doc

CFG = ChunkingConfig(target_tokens=60, overlap_tokens=10)


def long_doc(n=40):
    body = "\n".join(f"Sentence number {i} explains a rule about pensions and National Insurance." for i in range(n))
    return make_doc("long", "Long doc", "# Long doc\n\n## Rules\n\n" + body + "\n\n## Payments\n\nPayments are made every four weeks.")


def test_chunk_sizes_and_overlap():  # 7
    chunks = chunk_document(long_doc(), CFG)
    assert len(chunks) > 3
    assert all(c.token_count <= CFG.target_tokens for c in chunks)
    assert chunks[0].overlap_tokens == 0
    for prev, nxt in zip(chunks, chunks[1:]):
        if nxt.overlap_tokens:
            assert nxt.overlap_tokens <= CFG.overlap_tokens
            first_line = nxt.text.split("\n")[0]
            assert first_line in prev.text  # overlap text really comes from the previous chunk


def test_stable_unique_chunk_ids():  # 8
    a = chunk_document(long_doc(), CFG)
    b = chunk_document(long_doc(), CFG)
    ids = [c.chunk_id for c in a]
    assert ids == [c.chunk_id for c in b] and len(set(ids)) == len(ids)
    assert ids[0] == "long#00000" and all(c.document_id == "long" for c in a)


def test_metadata_preserved_in_every_chunk():  # 9
    doc = make_doc("m", "Meta", "A paragraph of text. " * 30, publication_date="2020-01-02", last_updated_date="2024-05-06",
                   parent_source_url="https://p", review_flags=["x"], provenance=[{"file": "f", "line": 1}])
    for c in chunk_document(doc, CFG):
        assert (c.title, c.source_url, c.licence, c.publisher) == ("Meta", doc.source_url, "OGL v3.0", "GOV.UK")
        assert (c.publication_date, c.last_updated_date, c.parent_source_url) == ("2020-01-02", "2024-05-06", "https://p")
        assert c.review_flags == ["x"] and c.provenance == [{"file": "f", "line": 1}]
        assert c.document_hash == doc.content_hash() and c.chunk_count == len(chunk_document(doc, CFG))


def test_chunking_is_idempotent_and_does_not_mutate_document():  # 10
    doc = long_doc()
    snapshot = doc.to_dict()
    first = [c.to_dict() for c in chunk_document(doc, CFG)]
    second = [c.to_dict() for c in chunk_document(doc, CFG)]
    assert first == second and doc.to_dict() == snapshot


def test_headings_kept_as_context_and_never_last_line():
    chunks = chunk_document(long_doc(), CFG)
    assert any(c.heading_path == "Long doc > Rules" for c in chunks)
    # heading_path describes where a chunk STARTS; a later heading inside the chunk stays in its text
    assert "## Payments" in chunks[-1].text
    for c in chunks:
        assert not c.text.rstrip().split("\n")[-1].startswith("#")


def test_table_split_repeats_header_row():
    rows = "\n".join(f"{2000 + i} to {2001 + i} | £{1000 + i}" for i in range(80))
    doc = make_doc("t", "Table doc", f"# Rates\n\nTax year | Amount\n{rows}")
    chunks = chunk_document(doc, CFG)
    assert len(chunks) > 1
    assert all(c.text.split("\n")[0].startswith("Tax year | Amount") or c.chunk_index == 0 and "Tax year | Amount" in c.text for c in chunks)


def test_short_empty_unicode_and_odd_whitespace():
    short = chunk_document(make_doc("s", "Short", "Just one line."), CFG)
    assert len(short) == 1 and short[0].chunk_count == 1
    assert chunk_document(make_doc("e", "Empty", "  \n\t \n"), CFG) == []
    assert chunk_document(make_doc("h", "Only heading", "# Only a heading"), CFG) == []
    odd = chunk_document(make_doc("u", "Unicode", "Pensioner\u00a0income\u00a0\u2013 £1,073,100 \u201cquoted\u201d  text\t\twith   spaces\r\nnext line"), CFG)
    assert "£1,073,100" in odd[0].text and "\u201cquoted\u201d" in odd[0].text
    assert "  " not in odd[0].text and "\r" not in odd[0].text


def test_very_long_unbroken_line_is_split():
    doc = make_doc("v", "Long line", "word " * 500)
    chunks = chunk_document(doc, CFG)
    assert len(chunks) > 5 and all(c.token_count <= CFG.target_tokens for c in chunks)
    assert count_tokens(chunks[0].text) > 0
