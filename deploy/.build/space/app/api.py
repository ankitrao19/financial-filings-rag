"""
api.py — FastAPI wrapper around retrieve -> generate, traced with Langfuse.

Reuses the eval harness functions as-is (retrieve, generate_answer, infer_filing), so the API
answers exactly the way the measured pipeline does. Defaults match the current best run (run 5):
nomic embeddings, filing inferred from the question text, prompt v1, k=15.

Endpoints:
    GET  /health     — models loaded? tracing on?
    GET  /tickers    — companies in the corpus
    POST /ask        — question (+ ticker) -> answer, sources, trace id
    POST /feedback   — thumbs up/down on an answer, stored as a Langfuse score

Run (from the project root):
    venv/bin/uvicorn app.api:app --port 8000
    open http://localhost:8000/docs
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# pipeline modules import each other by bare name (from paths import ...)
APP_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(APP_DIR), str(APP_DIR.parent / "pipeline")]

from tracing import OpenAI, langfuse, propagate_attributes, trace_url, tracing_enabled  # noqa: E402  (loads .env first)

from build_chunks import EMBED_MODELS, load_embedder  # noqa: E402
from evaluate_rag import ANSWER_MODEL, ANSWER_PROMPTS, TOP_K, generate_answer, looks_like_refusal, retrieve  # noqa: E402
from filing_inference import infer_filing, load_period_map  # noqa: E402
from paths import CHROMA_PATH  # noqa: E402

log = logging.getLogger("rag.api")

EMBED_MODEL = "nomic"
SNIPPET_CHARS = 400

state = {}


def load_state():
    """Load models + index once (the nomic model takes a few seconds and ~0.5 GB, far too slow per request).
    Called by the lifespan, or directly when this app is mounted inside another one (lifespans of
    mounted sub-apps don't run — see serve.py)."""
    if state:
        return
    embed_cfg = EMBED_MODELS[EMBED_MODEL]
    collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(embed_cfg["collection"])
    period_map = load_period_map()
    state.update(
        embed_cfg=embed_cfg,
        collection=collection,
        # EMBED_DEVICE=cpu on HF ZeroGPU: it reports cuda as available outside @spaces.GPU functions
        embedder=load_embedder(embed_cfg, device=os.getenv("EMBED_DEVICE")),
        period_map=period_map,
        tickers=sorted(period_map["fiscal_year_end_month"]),
        llm=OpenAI(),
    )


@asynccontextmanager
async def lifespan(_app):
    load_state()
    yield
    # spans are exported in a background batch; flush so nothing is lost on shutdown
    langfuse.flush()


app = FastAPI(title="Financial-filings RAG", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=3, examples=["What was Costco's total revenue in fiscal 2024?"])
    ticker: Optional[str] = Field(None, examples=["COST"], description="restrict retrieval to one company")
    k: int = Field(TOP_K, ge=1, le=50)
    filing: Literal["none", "inferred"] = "inferred"
    prompt: Literal["v1", "v2"] = "v1"
    session_id: Optional[str] = Field(None, description="groups traces from one UI session in Langfuse")
    user_id: Optional[str] = None


class Source(BaseModel):
    rank: int
    ticker: Optional[str]
    form: Optional[str]
    filing_date: Optional[str]
    distance: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    filing_filter: Optional[str]
    inference_reason: Optional[str]
    sources: list[Source]
    latency_ms: int
    trace_id: Optional[str]
    trace_url: Optional[str]


class FeedbackRequest(BaseModel):
    trace_id: str
    value: Literal[0, 1] = Field(description="1 = helpful, 0 = not helpful")
    comment: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok" if state else "loading",
        "embed_model": state["embed_cfg"]["model"] if state else None,
        "chunks_indexed": state["collection"].count() if state else None,
        "tracing": tracing_enabled(),
    }


@app.get("/tickers")
def tickers():
    return state["tickers"]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    ticker = req.ticker.upper() if req.ticker else None
    if ticker and ticker not in state["tickers"]:
        raise HTTPException(422, f"unknown ticker {ticker}; see GET /tickers")
    started = time.perf_counter()
    config = {"embed_model": EMBED_MODEL, "k": req.k, "filing": req.filing,
              "prompt": req.prompt, "answer_model": ANSWER_MODEL}

    # root span = the trace. propagate_attributes stamps session/user/tags/metadata onto it
    # and every child span, so they are filterable in the Langfuse UI.
    with langfuse.start_as_current_observation(
        name="rag-query", as_type="chain", input={"question": req.question, "ticker": ticker},
    ) as root, propagate_attributes(
        trace_name="rag-query",
        session_id=req.session_id,
        user_id=req.user_id,
        tags=["api", f"embed:{EMBED_MODEL}", f"filing:{req.filing}", f"prompt:{req.prompt}"],
        metadata={k: str(v) for k, v in config.items()},
    ):
        # 1. which filing is this question about? (never raises; a miss leaves ticker-only filtering)
        filing_date, reason = None, None
        with langfuse.start_as_current_observation(
            name="filing-inference", as_type="span", input={"ticker": ticker, "question": req.question},
        ) as span:
            if req.filing == "inferred" and ticker:
                inferred = infer_filing(state["period_map"], ticker, req.question)
                filing_date, reason = inferred["filing_date"], inferred["reason"]
                span.update(output=inferred)
            else:
                reason = "skipped_no_ticker" if req.filing == "inferred" else "disabled"
                span.update(output={"reason": reason})

        # 2. retrieve
        with langfuse.start_as_current_observation(
            name="retrieval", as_type="retriever",
            input={"question": req.question, "ticker": ticker, "filing_date": filing_date, "k": req.k},
            metadata={"collection": state["embed_cfg"]["collection"]},
        ) as span:
            docs, metas, dists = retrieve(
                state["collection"], state["embedder"], req.question,
                ticker=ticker, filing_date=filing_date, k=req.k, embed_cfg=state["embed_cfg"],
            )
            sources = [
                Source(rank=i, ticker=m.get("ticker"), form=m.get("form"), filing_date=m.get("filing_date"),
                       distance=round(d, 4), snippet=doc[:SNIPPET_CHARS])
                for i, (doc, m, d) in enumerate(zip(docs, metas, dists), start=1)
            ]
            span.update(output=[s.model_dump() for s in sources])

        # 3. generate — the OpenAI call inside is auto-traced as a generation under this span
        with langfuse.start_as_current_observation(
            name="generate-answer", as_type="span", input={"prompt_version": req.prompt, "chunks": len(docs)},
        ) as span:
            try:
                answer = generate_answer(state["llm"], "\n\n---\n\n".join(docs), req.question, req.prompt)
            except Exception as e:
                # full error goes to logs + Langfuse only: provider errors can echo parts of the API key
                log.exception("answer model failed")
                span.update(level="ERROR", status_message=str(e))
                root.update(level="ERROR", status_message=str(e))
                raise HTTPException(502, "answer model failed — see server logs")
            span.update(output=answer)

        refused = looks_like_refusal(answer)
        root.update(output={"answer": answer, "refused": refused}, metadata={"inference_reason": reason})
        trace_id = langfuse.get_current_trace_id()

    return AskResponse(
        answer=answer,
        refused=refused,
        filing_filter=filing_date,
        inference_reason=reason,
        sources=sources,
        latency_ms=int((time.perf_counter() - started) * 1000),
        trace_id=trace_id,
        trace_url=trace_url(trace_id),
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if not tracing_enabled():
        raise HTTPException(503, "tracing disabled — set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY")
    langfuse.create_score(
        trace_id=req.trace_id, name="user-feedback", value=req.value,
        data_type="BOOLEAN", comment=req.comment,
    )
    return {"ok": True}


# sanity: every prompt version the schema allows must exist in the harness
assert set(AskRequest.model_fields["prompt"].annotation.__args__) <= set(ANSWER_PROMPTS)
