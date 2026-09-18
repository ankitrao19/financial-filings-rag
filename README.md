# Financial-filings RAG

Answers questions about SEC 10-K / 10-Q filings for 15 companies, and measures how well it does against 45 hand-checked questions.

**Current best (run 5): 64.4% of answers fully correct, 88.9% at least partly correct.** Full history and analysis: [docs/METRICS.md](docs/METRICS.md).

## What's where

```
agentic/
├── README.md                  ← you are here
├── .env / .env.example        ← API keys (HF_TOKEN, OPENAI_API_KEY); .env is never committed
│
├── pipeline/                  ← all the code
│   ├── paths.py               ← every file location in one place
│   ├── build_chunks.py        ← step 1: fetch filings, chunk them, embed into the vector store
│   ├── filing_inference.py    ← works out which filing a question is about ("fiscal 2024" → that 10-K)
│   └── evaluate_rag.py        ← step 2: retrieve → answer → grade all 45 questions → metrics
│
├── app/                       ← serving layer (wraps pipeline/, no changes to it)
│   ├── api.py                 ← FastAPI: POST /ask (retrieve → generate), /feedback, /health, /tickers
│   ├── tracing.py             ← Langfuse client + traced OpenAI client
│   └── ui.py                  ← Gradio UI, calls the API over HTTP
│
├── tests/
│   └── test_filing_inference.py
│
├── data/                      ← inputs (the pipeline reads these)
│   ├── corpus/                ← 90 filings from SEC EDGAR, one JSON per filing
│   ├── chromadb/              ← vector store: one collection per embedding model (MiniLM, nomic)
│   ├── ground-truth-qna.json  ← the 45 evaluation questions + expected answers
│   └── filing-periods.json    ← lookup: filing → fiscal year / quarter (built from corpus)
│
├── results/                   ← outputs (the pipeline writes these)
│   ├── latest/                ← most recent eval run: metrics.json, results.json, answers.json
│   └── runs/                  ← saved runs, numbered to match docs/METRICS.md §7
│       ├── 0-baseline/
│       ├── 1-table-aware/
│       ├── 2-filing-filter-oracle/
│       ├── 3-prompt-v2-reverted/
│       ├── 4-nomic-oracle/
│       ├── 4b-nomic-no-filter/
│       └── 5-nomic-inferred/  ← current best
│
├── docs/
│   ├── METRICS.md             ← results, metric definitions, every run explained
│   └── process.md             ← learning notes
│
└── venv/                      ← Python environment
```

## How to run

Run everything from this folder with the venv's Python (scripts also work from any other folder).

```bash
source venv/bin/activate

# 1. Build data (only when the corpus or chunking changes)
python pipeline/build_chunks.py --fetch                                  # download filings → data/corpus
python pipeline/build_chunks.py --index --reset --embed-model nomic      # embed → data/chromadb (~30 min)
python pipeline/filing_inference.py --build                              # filing → fiscal period lookup

# 2. Evaluate (current best setup, ~5 min, uses OpenAI)
python pipeline/evaluate_rag.py --embed-model nomic --filing inferred

# Quick checks
python pipeline/build_chunks.py --stats --embed-model nomic              # how many chunks are indexed
python pipeline/filing_inference.py --check                              # which filing each question resolves to
python tests/test_filing_inference.py                                    # unit tests
python pipeline/evaluate_rag.py --embed-model nomic --filing inferred --limit 3   # smoke test on 3 questions
```

Useful `evaluate_rag.py` flags: `--embed-model minilm|nomic`, `--filing none|oracle|inferred`, `--prompt v1|v2`, `--k 15`, `--limit N`.

## Run the API + UI

```bash
venv/bin/pip install -r requirements.txt
venv/bin/uvicorn app.api:app --port 8000     # API, docs at http://localhost:8000/docs
venv/bin/python app/ui.py                     # UI at http://localhost:7860
```

Live demo: <https://hereicome-filings-rag.hf.space> (UI at `/`, API docs at `/rag/docs`). Deploy with `venv/bin/python deploy/deploy_space.py --watch`.

## Observability

Every query is traced end-to-end (retrieval → generation) with token/cost/latency per step via Langfuse. [Langfuse trace of one query](docs/images/langfuselogs.png)

```
rag-query                  chain       question + ticker -> answer          1.79s  $0.000752
├── filing-inference       span        question text -> filing_date filter  0.00s
├── retrieval              retriever   top-k chunks + distances             0.22s
└── generate-answer        span                                             1.56s
    └── OpenAI-generation  generation  gpt-4o-mini, 5,395 -> 33 tokens      1.56s  $0.000752
```

Tracing turns on when `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are set in `.env` (or in the Space secrets); without them the API runs unchanged with tracing disabled. Traces carry the run config as tags (`embed:nomic`, `filing:inferred`, `prompt:v1`), a session id per UI session, and 👍/👎 from the UI as a `user-feedback` score — so bad answers can be filtered out and folded back into the eval set.

## Recording a new experiment

1. Change **one** thing.
2. Run the eval; outputs land in `results/latest/`.
3. Keep it: copy `results/latest/` to `results/runs/<next number>-<short-name>/`.
4. Add a row to the log table in [docs/METRICS.md](docs/METRICS.md) §7 and a short section explaining the change and the result.
