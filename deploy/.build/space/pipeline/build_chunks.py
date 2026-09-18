"""
build_chunks.py — Stage 1 of the financial-filings RAG pipeline.

Everything that PRODUCES chunks lives here: pulling filings from SEC EDGAR,
splitting them into overlapping chunks, and loading them into chromadb.
Retrieval and evaluation live in evaluate_rag.py.

Usage:
    python pipeline/build_chunks.py --fetch            # pull filings from EDGAR into data/corpus
    python pipeline/build_chunks.py --index            # embed data/corpus and load into data/chromadb
    python pipeline/build_chunks.py --fetch --index    # both
    python pipeline/build_chunks.py --index --reset    # wipe the collection first, then reload
    python pipeline/build_chunks.py --stats            # just report what's already there
    python pipeline/build_chunks.py --index --embed-model nomic   # index into the nomic collection instead
"""

import argparse
import json
import os
import re
import time

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from paths import CHROMA_PATH, CORPUS_DIR, ENV_FILE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEC_IDENTITY = "Ankit Rao seeankitrao@gmail.com"  # SEC requires a real name + email

TICKERS = [
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    # Financials
    "JPM", "GS", "BAC", "MS", "V",
    # Consumer / retail
    "WMT", "KO", "PG", "COST", "NKE",
]

FORMS = ["10-K", "10-Q"]
FILINGS_PER_FORM = 3

CHUNK_SIZE = 1200      # characters (~300 tokens)
CHUNK_OVERLAP = 200    # characters of overlap between chunks
MAX_TABLE_CHARS = 6000 # tables above this are split on row boundaries, header repeated
CAPTION_CHARS = 300    # prose immediately above a table, prepended so the table chunk says what it is

# Each embedding model gets its own collection, so switching models never overwrites
# another model's index and earlier runs stay reproducible.
EMBED_MODELS = {
    "minilm": {
        "model": "all-MiniLM-L6-v2",       # 384-dim, reads only the first 256 tokens
        "collection": "edgar_corpus_collection",
        "doc_prefix": "",
        "query_prefix": "",
        "trust_remote_code": False,
        "normalize": False,                # already has a Normalize layer
        "batch": 64,
    },
    "nomic": {
        "model": "nomic-ai/nomic-embed-text-v1.5",   # 768-dim, 8192-token window
        "collection": "edgar_corpus_nomic",
        # nomic is trained with task prefixes; without them retrieval quality drops
        "doc_prefix": "search_document: ",
        "query_prefix": "search_query: ",
        "trust_remote_code": True,
        "normalize": True,                 # chroma's default L2 then ranks like cosine
        "batch": 8,                        # long table chunks (up to ~2.8k tokens) are memory-hungry
    },
}
DEFAULT_EMBED = "minilm"
ADD_BATCH = 500


def load_embedder(cfg, device=None):
    """device=None lets sentence-transformers pick (cuda/mps/cpu); pass "cpu" to pin it."""
    return SentenceTransformer(cfg["model"], trust_remote_code=cfg["trust_remote_code"], device=device)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
TABLE_SPACES_RE = re.compile(r"[ \t]{2,}")


def split_blocks(text):
    """Group lines into ("table", lines) / ("prose", lines) runs. A table is a run of
    consecutive lines containing '|' (edgar wraps long cells onto continuation lines)."""
    blocks = []
    for line in text.split("\n"):
        kind = "table" if "|" in line else "prose"
        if blocks and blocks[-1][0] == kind:
            blocks[-1][1].append(line)
        else:
            blocks.append((kind, [line]))
    return blocks


def slide_window(text):
    """Original fixed-size sliding window, used for prose."""
    pieces = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(text), step):
        pieces.append(text[i : i + CHUNK_SIZE].strip())
    return pieces


def table_pieces(rows, caption):
    """A whole table as one piece. Only tables over MAX_TABLE_CHARS are split, and then
    only between rows, with the caption + first two (header) rows repeated on each part."""
    # edgar pads cells with long runs of spaces; collapsing them is what lets most tables fit whole
    rows = [TABLE_SPACES_RE.sub(" ", r).strip() for r in rows if r.strip()]
    whole = "\n".join(([caption] if caption else []) + rows)
    if len(whole) <= MAX_TABLE_CHARS:
        return [whole]

    header = ([caption] if caption else []) + rows[:2]
    head_len = sum(len(h) + 1 for h in header)
    pieces, part, size = [], [], head_len
    for row in rows[2:]:
        if part and size + len(row) + 1 > MAX_TABLE_CHARS:
            pieces.append("\n".join(header + part))
            part, size = [], head_len
        part.append(row)
        size += len(row) + 1
    if part:
        pieces.append("\n".join(header + part))
    return pieces


def chunk_text(text, meta):
    """Split text into chunks, each carrying full metadata. Markdown tables are never
    cut mid-table: each becomes its own chunk. Prose keeps the sliding window."""
    pieces, prev_prose = [], ""
    for kind, lines in split_blocks(text):
        if kind == "prose":
            prev_prose = "\n".join(lines)
            pieces.extend(slide_window(prev_prose))
        else:
            # the last lines of prose above a table are usually its title ("CONSOLIDATED STATEMENTS OF ...")
            caption = prev_prose[-CAPTION_CHARS:].strip()
            pieces.extend(table_pieces(lines, caption))

    chunks = []
    for piece in pieces:
        if len(piece) < 50:  # skip tiny fragments
            continue
        chunks.append({
            "text": piece,
            "metadata": {**meta, "chunk_index": len(chunks)},
        })
    return chunks


def rebuild_text(chunks):
    """Reassemble a filing's full text from its stored overlapping chunks, so the corpus
    can be re-chunked without re-fetching (which could pull different filings)."""
    text = chunks[0]["text"] if chunks else ""
    for chunk in chunks[1:]:
        nxt = chunk["text"]
        # earliest tail position whose remainder is a prefix of the next chunk = the overlap
        for pos in range(max(0, len(text) - CHUNK_OVERLAP - 100), len(text)):
            if nxt.startswith(text[pos:]):
                text = text[:pos] + nxt
                break
        else:  # overlap was pure whitespace and got stripped away
            text += "\n" + nxt
    return text


# ---------------------------------------------------------------------------
# Stage 1a: fetch filings from EDGAR
# ---------------------------------------------------------------------------
def fetch_corpus():
    from edgar import Company, set_identity

    set_identity(SEC_IDENTITY)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        try:
            company = Company(ticker)
        except Exception as e:
            print(f"[skip] {ticker}: could not resolve company ({e})")
            continue

        for form in FORMS:
            try:
                filings = company.get_filings(form=form).latest(FILINGS_PER_FORM)
            except Exception as e:
                print(f"[skip] {ticker} {form}: {e}")
                continue

            if not isinstance(filings, list):
                filings = [filings]

            for filing in filings:
                try:
                    # .markdown() keeps heading/table structure; .text() is the fallback
                    try:
                        body = filing.markdown()
                    except Exception:
                        body = filing.text()

                    meta = {
                        "ticker": ticker,
                        "form": form,
                        "filing_date": str(getattr(filing, "filing_date", "")),
                        "accession": str(getattr(filing, "accession_no", "")),
                        "company": company.name,
                    }

                    chunks = chunk_text(body, meta)
                    fname = f"{ticker}_{form.replace('-', '')}_{meta['filing_date']}.json"
                    (CORPUS_DIR / fname).write_text(
                        json.dumps(chunks, indent=2, ensure_ascii=False)
                    )
                    print(f"[ok]  {ticker} {form} {meta['filing_date']}: {len(chunks)} chunks")

                    time.sleep(0.2)  # be polite to sec.gov (<10 req/s)
                except Exception as e:
                    print(f"[warn] {ticker} {form} filing failed: {e}")

    files = list(CORPUS_DIR.glob("*.json"))
    total = sum(len(json.loads(f.read_text())) for f in files)
    print(f"\nFetched {len(files)} filings, {total} chunks into {CORPUS_DIR}")


# ---------------------------------------------------------------------------
# Stage 1b: embed the corpus and load it into chromadb
# ---------------------------------------------------------------------------
def get_collection(cfg, reset=False):
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    if reset:
        try:
            client.delete_collection(name=cfg["collection"])
            print(f"[reset] dropped collection {cfg['collection']}")
        except Exception:
            pass
    return client.get_or_create_collection(name=cfg["collection"])


def index_corpus(cfg, reset=False):
    load_dotenv(ENV_FILE)
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN missing — copy .env.example to .env and fill it in")

    files = sorted(CORPUS_DIR.glob("*.json"))
    if not files:
        raise RuntimeError(f"no corpus files in {CORPUS_DIR} — run with --fetch first")

    documents, metadatas, ids = [], [], []
    for f in files:
        stored = json.loads(f.read_text(encoding="utf-8"))
        if not stored:
            continue
        # corpus files hold chunks cut at fetch time; re-chunk them with the current chunk_text()
        base_meta = {k: v for k, v in stored[0]["metadata"].items() if k != "chunk_index"}
        for chunk in chunk_text(rebuild_text(stored), base_meta):
            meta = chunk["metadata"]
            documents.append(chunk["text"])
            metadatas.append(meta)
            ids.append(f"{meta['accession']}_{meta['chunk_index']}")  # unique per chunk

    print(f"{len(files)} filings -> {len(documents)} chunks to index")

    model = load_embedder(cfg)
    embeddings = model.encode(
        [cfg["doc_prefix"] + d for d in documents],
        batch_size=cfg["batch"],
        normalize_embeddings=cfg["normalize"],
        show_progress_bar=True,
    ).tolist()

    collection = get_collection(cfg, reset=reset)
    for i in range(0, len(documents), ADD_BATCH):
        # upsert (not add) so re-running is idempotent instead of erroring on dupes
        collection.upsert(
            ids=ids[i : i + ADD_BATCH],
            embeddings=embeddings[i : i + ADD_BATCH],
            metadatas=metadatas[i : i + ADD_BATCH],
            documents=documents[i : i + ADD_BATCH],
        )
    print(f"Indexed {len(documents)} chunks. Collection now holds {collection.count()}.")


def stats(cfg):
    files = sorted(CORPUS_DIR.glob("*.json"))
    corpus_chunks = sum(len(json.loads(f.read_text())) for f in files) if files else 0
    collection = get_collection(cfg)
    print(f"corpus:     {len(files)} filings, {corpus_chunks} chunks on disk")
    print(f"chromadb:   {collection.count()} chunks in '{cfg['collection']}' ({cfg['model']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true", help="pull filings from SEC EDGAR")
    ap.add_argument("--index", action="store_true", help="embed corpus into chromadb")
    ap.add_argument("--reset", action="store_true", help="drop the collection before indexing")
    ap.add_argument("--stats", action="store_true", help="report corpus / collection size")
    ap.add_argument("--embed-model", choices=sorted(EMBED_MODELS), default=DEFAULT_EMBED,
                    help="embedding model; each one indexes into its own collection")
    args = ap.parse_args()

    if args.fetch:
        fetch_corpus()
    if args.index:
        index_corpus(EMBED_MODELS[args.embed_model], reset=args.reset)
    if args.stats or not (args.fetch or args.index):
        stats(EMBED_MODELS[args.embed_model])


if __name__ == "__main__":
    main()
