"""
tracing.py — Langfuse setup for the API, in one place.

Tracing is optional: with LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY unset the Langfuse
client runs in no-op mode, every span call below does nothing, and the API still answers.

What gets traced (see docs/LANGFUSE.md for the full picture):
    rag-query                    trace + root span  (one per POST /ask)
    ├── filing-inference         span       question text -> filing_date filter
    ├── retrieval                retriever  chroma query, top-k chunks + distances
    └── generate-answer          span
        └── OpenAI-generation    generation auto-captured by langfuse.openai (model, tokens, cost)
"""

import os

from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes  # noqa: F401  (re-exported for api.py)

from paths import ENV_FILE

# must run before get_client() and before langfuse.openai is imported: both read keys from env
load_dotenv(ENV_FILE)

# drop-in replacement for openai.OpenAI — every chat.completions.create becomes a Langfuse
# generation nested under whatever span is current
from langfuse.openai import OpenAI  # noqa: E402

langfuse = get_client()


def tracing_enabled():
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def trace_url(trace_id):
    """Deep link to the trace in the Langfuse UI, or None when tracing is off."""
    if not (tracing_enabled() and trace_id):
        return None
    try:
        return langfuse.get_trace_url(trace_id=trace_id)
    except Exception:
        return None
