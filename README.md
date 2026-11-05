# UK Pension AI

A question-answering system for UK pensions that answers **only from retrieved official sources** (GOV.UK, HMRC), cites them, and says when the evidence is not enough.

```
React + TypeScript  ->  FastAPI  ->  LangGraph agent  ->  RAG retriever (Qdrant) + answer writer  ->  cited answer
                                          |-> pension calculator tool
```

## 1. What is in this folder, and what is proven

| Part | Where | Status |
|---|---|---|
| Data validation + cleaning | `src/ingestion/` | **Run on the real data**; tested |
| Chunking, embeddings (hashing), Qdrant, incremental ingestion | `src/rag/` | **Run on the real data**; tested |
| Hybrid search (dense + BM25), duplicate collapsing | `src/rag/retriever.py`, `bm25.py` | **Measured** on the evaluation set (section 8) |
| Evaluation set (51 questions) + scripts | `data/eval/`, `src/rag/evaluate.py`, `scripts/` | Run; labels **not yet human-reviewed** |
| Gemini embedder / answer writer / reranker | `src/rag/gemini.py` | Written against the SDK; tested with a **fake** client only. **Never run against real Gemini** (the build sandbox could not reach Google) |
| SLM training-data scripts | `src/slm/data/` | Tested with a fake model; Gemini steps **not run for real** |
| Colab QLoRA notebook | `src/slm/train/train_qlora.ipynb` | **Not run** (no GPU in the build sandbox) |
| Ollama connector | `src/slm/ollama_llm.py` | Tested against a fake local server; **not run with real Ollama** |
| LangGraph agent + 3 tools | `src/agent/` | Tested (fake/extractive answers) |
| FastAPI backend (login, chat, streaming, feedback, admin upload) | `src/api/` | Tested, and smoke-tested live against the real index |
| React + TypeScript UI | `frontend/` | Type-checks and builds; **never opened in a browser by the author** |
| The trained SLM | - | **Does not exist yet.** You create it on Colab |

Automated tests: `python -m pytest` (119 pass; they never touch the real index or any paid service).

## 2. First-time setup (laptop, VS Code)

```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                   # then edit .env (never commit it)
python -m pytest                       # sanity check
python -m src.rag.ingest               # builds data/qdrant (about 1 minute with the hashing embedder)
python -m src.rag.retriever "How do I claim Pension Credit?"
```
The Qdrant index is **not** in the zip (it is generated). `data/processed/` already holds the cleaned documents and chunks.

## 3. Run the application

```bash
# terminal 1 - backend (ONE worker: local Qdrant allows one process)
uvicorn src.api.main:create_app --factory --port 8000
# terminal 2 - frontend
cd frontend && cp .env.example .env && npm install && npm run dev      # http://localhost:5173
```
Set `ADMIN_USERNAME`, `ADMIN_PASSWORD` and a long `API_SECRET_KEY` in `.env` first, then log in as that admin. Admins get a **Documents** tab to upload files. Normal users can only ask questions (create users by setting `ALLOW_REGISTRATION=true`, or add them yourself in the database).

* Without a model (`LLM_PROVIDER=none`) answers are **quoted source passages with numbered citations** - safe and fully working.
* `LLM_PROVIDER=gemini` or `ollama` makes the answer writer a model; the citation check, number check and refusal rules stay on.

## 4. Adding documents: no model training needed

Documents are **knowledge**; the fine-tuned model only learns **behaviour** (cite, refuse, ignore hidden instructions).
Admin uploads (TXT, MD, text-based PDF, DOCX) are validated, chunked, embedded and searchable immediately, labelled "Admin-uploaded" in answers, and appended to `data/processed/documents_uploads.jsonl`.
Uploaded text becomes evidence users will trust, so **only admins can upload**. Scanned PDFs (no text layer) are rejected (no OCR). To bulk-load files: `python -m src.rag.ingest --input data/processed/documents_uploads.jsonl` (never add `--delete-missing` for partial files).
Deleting documents from the UI is not implemented; deletion needs an explicit full-snapshot ingest (`--delete-missing --full-snapshot`).

## 5. Switch to Gemini embeddings (expected quality gain - UNPROVEN until you measure it)

1. In `.env`: `GEMINI_API_KEY=...`, `RAG_EMBEDDING_PROVIDER=gemini` (check `GEMINI_EMBEDDING_MODEL` is current in Google AI Studio).
2. `python -m src.rag.ingest --rebuild-collection` (drops and rebuilds **only** our collection; embeddings are cached in `data/cache/`, so you never pay twice for the same text; calls are rate-limited and capped per day).
3. `python -m src.rag.evaluate` and `python -m scripts.compare_retrieval` - compare with the numbers in section 8.
4. Recalibrate `RAG_MIN_SCORE` (the 0.2 default was set for the hashing embedder).

## 6. SLM: training data -> Colab -> Ollama

Run in this order (laptop, inside this folder; needs `GEMINI_API_KEY` and `GEMINI_MODEL` for steps 2 and 4). Every step saves progress and can be re-run.

```bash
python -m src.slm.data.split_documents                 # 1 split by DOCUMENT first (eval documents are held out)
python -m src.slm.data.generate_questions --limit 100  # 2 Gemini writes a question per chunk      (pilot: 100)
python -m src.slm.data.build_inputs                    # 3 real retriever -> the evidence the SLM will see (+off-topic + injection cases)
python -m src.slm.data.draft_answers --limit 100       # 4 Gemini drafts cited answers / refusals
python -m src.slm.data.auto_check                      # 5 automatic flags (bad citations, invented figures, ...)
python -m src.slm.data.review                          # 6 YOU approve / edit / reject every example (cannot be skipped)
python -m src.slm.data.export_chat_format              # 7 writes data/slm/train.jsonl, val.jsonl, test.jsonl
```
Then open `src/slm/train/train_qlora.ipynb` in Colab (GPU runtime), upload `train.jsonl` and `val.jsonl`, choose a small instruct model (check its licence), train, save the adapter to Drive. Load it into Ollama (adapter or merged GGUF - follow the current Ollama docs), then set `LLM_PROVIDER=ollama` and `OLLAMA_MODEL=<name>`.
Rules: never train on `eval_set.json` questions (export drops them); do a 100-example pilot before scaling; check the drafting model's terms of use for training other models on its outputs; compare base vs fine-tuned on the test set before trusting it.

## 7. Where each thing lives

```
src/ingestion/   validate_dataset.py (checks every line), clean.py (drops markup-noise docs, reports duplicates)
src/rag/         config, schema, chunking, embeddings, gemini, bm25, vector_store, ingest, retriever, answer_generation, evaluate
src/slm/         data/ (7 training-data scripts), train/ (Colab notebook), ollama_llm.py
src/agent/       tools.py (search, NI-years calculator, fetch source), graph.py (LangGraph: rewrite -> route -> retrieve -> answer)
src/api/         main.py (routes), service.py, security.py, db.py, uploads.py
frontend/        React + TypeScript (login, chat, citation drawer, admin upload)
data/raw/        your original JSONL files (never modified)
data/processed/  documents_merged.jsonl, documents_clean.jsonl, chunks.jsonl, manifest, validation + cleaning reports
data/eval/       eval_set.json (51 questions) and measured results
data/slm/        training-data working files (generated) + offtopic_questions.txt (edit freely)
tests/           pytest suite; scripts/ comparison + chunk-size experiment + incremental demo
```

## 8. Measured results (hashing embedder, real data, 39 answerable + 12 unanswerable questions)

| Retrieval variant | Success@5 | MRR |
|---|---|---|
| dense (vectors only, the old baseline) | 0.64 | 0.387 |
| keywords only (BM25) | 0.77 | 0.532 |
| **hybrid (default)** | **0.85** | **0.624** |
| hybrid + max 2 chunks per document (default) | 0.85 | 0.627 |

Chunk size (hybrid): 250 tokens S@5 0.74 / MRR 0.550; 350 -> 0.74 / 0.557; **500 -> 0.85 / 0.627** (kept).
Abstention: all 7 clearly off-topic questions were refused, 0 of 39 answerable questions were wrongly refused, but **0 of 5 pension-sounding unsupported questions** (for example a 2045 State Pension amount) were refused. The optional relevance judge (`AnswerGenerator(judge=...)`) is meant to catch these; it is tested with a fake model only.
**Caveats:** Success@k = a relevant document is in the top k. The questions were written from verbatim source sentences, which favours keyword matching, and labels are unreviewed, so absolute numbers are optimistic. Use them to compare variants, not as a promise of accuracy. Not yet measured: Gemini embeddings, reranking, any generated answers. Known weakness: for "How do I claim Pension Credit?" news articles can outrank the main guide.

## 9. Known limits (read before a demo)

* Data is a **snapshot of unknown date**: rates and thresholds may be out of date. Say "based on the collected sources".
* 15 documents have conflicting versions awaiting review (flagged; answers disclose it). 135 documents have no licence, most have no date.
* The calculator (`src/agent/tools.py`) only compares qualifying years with the 10-year and 35-year thresholds found in the sources. It does **not** compute pound amounts or State Pension age (transition rules not verified).
* The "streaming" endpoint sends the finished answer in small pieces; it is not model-token streaming.
* Login/security is demo-grade: use HTTPS, a strong `API_SECRET_KEY`, and review before real users. Tokens are kept in the browser's sessionStorage.
* Local Qdrant allows one process; for several API workers run a Qdrant server and set `RAG_QDRANT_URL`.
* Pension answers are general information, not personal financial, tax or legal advice (shown in every answer and in the UI).

## 10. Answer evaluation (does the model write good, cited, honest answers?)

Retrieval evaluation asks "did search find the right page?". Answer evaluation asks "did the writer use it properly?". Run it once per model and compare:
```bash
python -m scripts.answer_eval --provider gemini --tag gemini              # needs GEMINI_API_KEY + GEMINI_MODEL; ~51 embedding + ~51 text calls
OLLAMA_MODEL=<base-small-model> python -m scripts.answer_eval --provider ollama --tag base-slm
OLLAMA_MODEL=pension-slm        python -m scripts.answer_eval --provider ollama --tag tuned-slm
python -m scripts.answer_eval --compare gemini base-slm                   # side by side, no model calls
```
It writes `data/eval/answer_eval_<tag>.json` (every answer) and `answer_review_<tag>.md` (20 answers for YOU to mark correct/partly/wrong - automatic checks cannot tell whether a sentence is truly supported).
Automatic metrics (on questions whose retrieved passages really contained the answer): answered-not-refused, citations present and valid, cites a labelled-correct document, key figures present, no invented figures; plus refusal rate on unanswerable questions. The safety nets are switched off in this script so you see the model itself.
**Use it to decide whether the SLM is worth it:** fine-tune only if a base model with our prompt fails on citations/refusals, and keep the tuned model only if it beats the base model here.

## 11. Changes in this version
* Honest refusals are kept: a short reply with no citations that says the sources do not answer the question is shown as a refusal (it used to be thrown away and replaced by quoted passages).
* Rejected replies are saved (`raw_reply`; shown only when `DEBUG_RAW_REPLIES=true`), messages no longer claim "no language model" when one was used, and the disclaimer is not repeated.
* Gemini: daily cap now persists per Pacific day, vector size is remembered (no API call on start), embedding batches of 10, no `temperature`.
* API dates are shown as YYYY-MM-DD; optional `RAG_USE_JUDGE` and `RAG_RERANK` switches (off by default; NOT measured yet).
* UI: sources grouped per page, one calm message for refusals, "How this answer was made" panel, notes collapsed.
* Not fixed: pension-sounding questions with no support can still be answered unless the judge is on and measured; conflicting document versions need a human decision.

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `collection_missing` | run `python -m src.rag.ingest` |
| `ConfigMismatchError` | embedding settings changed: `python -m src.rag.ingest --rebuild-collection` |
| `Storage folder ... already accessed` | only one process may open local Qdrant; stop the other (e.g. the API) |
| `GEMINI_API_KEY is not set` / `GEMINI_MODEL is not set` | add them to `.env` (look up the current model name in AI Studio) |
| Gemini 429 / daily cap | scripts wait and retry; the daily cap stops the run safely; rerun tomorrow |
| Login stops working after restart | set a fixed `API_SECRET_KEY` (>= 32 chars) in `.env` |
| Frontend cannot reach backend | check `VITE_API_URL` and `CORS_ORIGINS` |
