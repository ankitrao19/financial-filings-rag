"""
paths.py — every file and folder the pipeline reads or writes, in one place.

All paths are anchored to the project root (the folder above pipeline/), so the
scripts work no matter which directory you run them from.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = ROOT / ".env"

# inputs
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"                     # filings pulled from SEC EDGAR, one JSON per filing
CHROMA_PATH = DATA_DIR / "chromadb"                  # vector store (one collection per embedding model)
GROUND_TRUTH = DATA_DIR / "ground-truth-qna.json"    # the 45 evaluation questions + expected answers
PERIOD_MAP_PATH = DATA_DIR / "filing-periods.json"   # filing -> fiscal year/quarter lookup (filing_inference.py --build)

# outputs of the most recent evaluate_rag.py run
LATEST_DIR = ROOT / "results" / "latest"
ANSWERS_OUT = LATEST_DIR / "answers.json"            # question + generated answer
RESULTS_OUT = LATEST_DIR / "results.json"            # per-question detail: verdict, retrieval trace, inference
METRICS_OUT = LATEST_DIR / "metrics.json"            # aggregate metrics

# frozen copies of past runs, numbered to match docs/METRICS.md
RUNS_DIR = ROOT / "results" / "runs"
