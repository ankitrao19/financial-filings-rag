"""
evaluate_rag.py — Stage 2 of the financial-filings RAG pipeline.

Retrieval + generation + grading against the ground-truth set. Chunk building
lives in build_chunks.py; nothing here writes to the vector store.

For every question in data/ground-truth-qna.json it:
  1. retrieves top-k chunks (filtered by ticker),
  2. scores RETRIEVAL — did the tagged filing come back, and at what rank,
  3. generates an answer from the retrieved context,
  4. scores the ANSWER two ways — an LLM judge and a provider-free numeric check,
  5. aggregates accuracy / coverage / refusal / hit-rate / MRR metrics.

Outputs (results/latest/, see paths.py):
    answers.json  — question + generated answer (per question)
    results.json  — full per-question detail incl. retrieval trace
    metrics.json  — aggregate metrics, incl. success percent
Copy results/latest/ into results/runs/<n>-<name>/ to keep a run.

Usage:
    python pipeline/evaluate_rag.py              # run the whole set
    python pipeline/evaluate_rag.py --limit 5    # smoke-test on the first 5 questions
    python pipeline/evaluate_rag.py --k 25       # retrieve 25 chunks instead of 15
    python pipeline/evaluate_rag.py --no-filter  # drop the ticker filter (to measure its effect)
    python pipeline/evaluate_rag.py --filing-filter  # also restrict to the tagged filing_date (oracle; same as --filing oracle)
    python pipeline/evaluate_rag.py --filing inferred  # restrict to the filing inferred from the question text
    python pipeline/evaluate_rag.py --prompt v2      # hardened answer prompt (see ANSWER_PROMPTS)
    python pipeline/evaluate_rag.py --embed-model nomic   # query the nomic collection (index it first)

    Current best (run 5): python pipeline/evaluate_rag.py --embed-model nomic --filing inferred
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from build_chunks import DEFAULT_EMBED, EMBED_MODELS, load_embedder
from filing_inference import infer_filing, load_period_map, mismatch_kind
from paths import ANSWERS_OUT, CHROMA_PATH, ENV_FILE, GROUND_TRUTH, LATEST_DIR, METRICS_OUT, RESULTS_OUT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ANSWER_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"
TOP_K = 15

REFUSAL_PATTERNS = [
    "i don't know", "i do not know", "not provided in the context",
    "not contain", "does not include", "not specified in the context",
]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup(embed_cfg):
    load_dotenv(ENV_FILE)
    for key in ("HF_TOKEN", "OPENAI_API_KEY"):
        if not os.getenv(key):
            raise RuntimeError(f"{key} missing — copy .env.example to .env and fill it in")

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = chroma_client.get_or_create_collection(name=embed_cfg["collection"])
    if collection.count() == 0:
        raise RuntimeError(f"collection {embed_cfg['collection']} is empty — index it with build_chunks.py first")
    embedder = load_embedder(embed_cfg)
    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return collection, embedder, llm


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(collection, embedder, question, ticker=None, filing_date=None, k=TOP_K,
             embed_cfg=EMBED_MODELS[DEFAULT_EMBED]):
    qvec = embedder.encode(
        embed_cfg["query_prefix"] + question, normalize_embeddings=embed_cfg["normalize"]
    ).tolist()
    clauses = []
    if ticker:
        clauses.append({"ticker": ticker})
    if filing_date:
        clauses.append({"filing_date": filing_date})
    # chroma wants $and for more than one condition
    where = {"$and": clauses} if len(clauses) > 1 else (clauses[0] if clauses else None)
    res = collection.query(
        query_embeddings=[qvec],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return res["documents"][0], res["metadatas"][0], res["distances"][0]


def score_retrieval(metas, expected_ticker, expected_filing_date):
    """Filing-level hit + rank. Weak proxy: right filing can still be the wrong
    section, and cross-year questions may legitimately be answered from a
    different filing than the one tagged in the ground truth."""
    rank = None
    for i, m in enumerate(metas, start=1):
        if m.get("ticker") == expected_ticker and m.get("filing_date") == expected_filing_date:
            rank = i
            break
    return {
        "hit": rank is not None,
        "rank": rank,
        "reciprocal_rank": (1.0 / rank) if rank else 0.0,
        "retrieved_filings": sorted({
            f"{m.get('ticker')}|{m.get('form')}|{m.get('filing_date')}" for m in metas
        }),
    }


def filing_is_indexed(collection, ticker, filing_date):
    """Is the filing the ground truth points at actually in the vector store?
    If not, the question is unanswerable and accuracy shouldn't be blamed on the model."""
    got = collection.get(
        where={"$and": [{"ticker": ticker}, {"filing_date": filing_date}]},
        limit=1,
        include=["metadatas"],
    )
    return len(got["ids"]) > 0


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
ANSWER_PROMPTS = {
    # v1: baseline prompt, runs 0-2
    "v1": """Answer using ONLY the context below. If the answer isn't in the context, say you don't know. Cite the ticker and filing date.

Context:
{context}

Question: {question}""",
    # v2: hardened against picking the wrong row/period out of similar-looking tables
    "v2": """You answer questions about SEC filings using ONLY the context below.

Rules for picking the figure:
1. Line item: use the row whose label matches the line item named in the question. Do not substitute a similarly named row (e.g. "other operating expense" is not "other income (expense), net").
2. Scope: use the consolidated, company-wide total unless the question names a segment, business line or product. Ignore segment-level and geographic breakdowns otherwise.
3. Period: match the exact period asked. A quarter means the "three months ended" column, not "six/nine months ended" or year-to-date. A fiscal year means the full-year column for that fiscal year. Check the column header before reading a number.
4. Comparisons: when the question compares periods, give the figure for each period, then the change.
5. Report figures exactly as they appear in the table, with their units (e.g. "$15,745 million"). Do not round or recompute unless asked.

If no row in the context satisfies rules 1-3, say you don't know rather than using a near match.

Answer in one or two sentences, then cite the ticker and filing date.

Context:
{context}

Question: {question}""",
}


def generate_answer(llm, context, question, prompt_version="v1"):
    prompt = ANSWER_PROMPTS[prompt_version].format(context=context, question=question)
    resp = llm.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Grading — two independent signals
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are grading a financial-QA system. Compare the generated answer against the expected answer.

Question: {question}
Expected answer: {expected}
Generated answer: {generated}

Grade with one verdict:
- "correct": every key figure/fact in the expected answer is present and stated accurately (unit rewrites are fine, e.g. $61,761 million == $61.8 billion).
- "partial": some key figures/facts are right but others are missing or wrong.
- "incorrect": the answer asserts something that contradicts the expected answer.
- "refused": the answer declines to answer (says it doesn't know / not in context) without giving the figures.

Reply with JSON only: {{"verdict": "...", "reason": "one short sentence"}}"""


def judge_answer(llm, question, expected, generated):
    resp = llm.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                question=question, expected=expected, generated=generated
            ),
        }],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        out = json.loads(resp.choices[0].message.content)
        verdict = str(out.get("verdict", "")).lower().strip()
        if verdict not in {"correct", "partial", "incorrect", "refused"}:
            verdict = "unparsed"
        return verdict, out.get("reason", "")
    except (json.JSONDecodeError, TypeError):
        return "unparsed", "judge returned non-JSON"


NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def figures(text):
    """Numbers that carry meaning — drop bare years, which otherwise inflate matches."""
    out = set()
    for raw in NUM_RE.findall(text or ""):
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        if val.is_integer() and 1900 <= val <= 2100:  # a year, not a figure
            continue
        out.add(val)
    return out


def numeric_recall(expected, generated):
    """Provider-free check: what share of the expected figures show up verbatim?
    Blind to unit rewrites ($61,761M vs $61.8B), so it under-counts — it's a
    sanity check on the judge, not a replacement for it."""
    want = figures(expected)
    if not want:
        return None, 0, 0
    got = figures(generated)
    matched = len(want & got)
    return matched / len(want), matched, len(want)


def score_passage(docs, expected):
    """Passage-level retrieval: rank of the first chunk holding at least half of the
    expected figures. Unlike score_retrieval, a filing_date filter can't make this
    trivially perfect. Undefined (None) when the expected answer has no figures."""
    want = figures(expected)
    if not want:
        return {"hit": None, "rank": None, "reciprocal_rank": None}
    need = max(1, (len(want) + 1) // 2)
    for i, doc in enumerate(docs, start=1):
        if len(want & figures(doc)) >= need:
            return {"hit": True, "rank": i, "reciprocal_rank": 1.0 / i}
    return {"hit": False, "rank": None, "reciprocal_rank": 0.0}


def looks_like_refusal(text):
    low = (text or "").lower()
    return any(p in low for p in REFUSAL_PATTERNS)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def pct(numerator, denominator):
    return round(100.0 * numerator / denominator, 1) if denominator else 0.0


def inference_summary(rows):
    """How often the filing inferred from the question text equals the ground-truth filing.
    The ground truth is only used here, for scoring — never to pick the filter."""
    inferred = [r for r in rows if r.get("filing_inference")]
    if not inferred:
        return None
    by_reason = defaultdict(lambda: {"questions": 0, "matches_ground_truth": 0, "correct_answers": 0})
    by_mismatch = defaultdict(list)
    for r in inferred:
        fi = r["filing_inference"]
        b = by_reason[fi["reason"]]
        b["questions"] += 1
        b["matches_ground_truth"] += fi["matches_ground_truth"]
        b["correct_answers"] += r["verdict"] == "correct"
        if fi["mismatch"] != "match":
            by_mismatch[fi["mismatch"]].append(r["id"])
    resolved = [r for r in inferred if r["filing_inference"]["filing_date"]]
    return {
        "resolved": len(resolved),
        "unresolved_fell_back_to_ticker_filter": len(inferred) - len(resolved),
        "matches_ground_truth": sum(r["filing_inference"]["matches_ground_truth"] for r in inferred),
        "match_percent": pct(sum(r["filing_inference"]["matches_ground_truth"] for r in inferred), len(inferred)),
        "by_reason": dict(by_reason),
        # why inference disagreed with the ground truth: parse miss (fallback_*),
        # wrong_year, wrong_period (quarter vs annual / wrong quarter), or both
        "mismatch_kinds": {k: {"count": len(v), "ids": v} for k, v in sorted(by_mismatch.items())},
        "mismatched": [
            {"id": r["id"], "inferred": r["filing_inference"]["filing_date"],
             "key": r["filing_inference"]["key"], "reason": r["filing_inference"]["reason"],
             "mismatch": r["filing_inference"]["mismatch"], "verdict": r["verdict"]}
            for r in inferred if not r["filing_inference"]["matches_ground_truth"]
        ],
    }


def aggregate(rows, pairs, k, ticker_filter, filing_mode, prompt_version, embed_cfg):
    total = len(rows)
    verdicts = defaultdict(int)
    for r in rows:
        verdicts[r["verdict"]] += 1

    correct = verdicts["correct"]
    partial = verdicts["partial"]
    answerable = [r for r in rows if r["filing_indexed"]]
    hits = sum(1 for r in rows if r["retrieval"]["hit"])
    mrr = sum(r["retrieval"]["reciprocal_rank"] for r in rows) / total if total else 0.0
    recalls = [r["numeric_recall"] for r in rows if r["numeric_recall"] is not None]
    passage_rows = [r for r in rows if r["passage"]["hit"] is not None]
    passage_hits = sum(1 for r in passage_rows if r["passage"]["hit"])
    passage_mrr = (
        sum(r["passage"]["reciprocal_rank"] for r in passage_rows) / len(passage_rows)
        if passage_rows else 0.0
    )

    def bucket(key):
        out = {}
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        for name, group in sorted(groups.items()):
            n = len(group)
            c = sum(1 for r in group if r["verdict"] == "correct")
            p = sum(1 for r in group if r["verdict"] == "partial")
            h = sum(1 for r in group if r["retrieval"]["hit"])
            out[name] = {
                "questions": n,
                "correct": c,
                "accuracy_percent": pct(c, n),
                "lenient_accuracy_percent": pct(c + p, n),
                "retrieval_hit_percent": pct(h, n),
            }
        return out

    # ground-truth health: is the eval set itself complete?
    missing_expected = [o["id"] for o in pairs if not str(o.get("expected_answer", "")).strip()]
    unverified = [o["id"] for o in pairs if not o.get("verified")]
    not_indexed = [r["id"] for r in rows if not r["filing_indexed"]]

    return {
        "config": {
            "answer_model": ANSWER_MODEL,
            "judge_model": JUDGE_MODEL,
            "embedding_model": embed_cfg["model"],
            "collection": embed_cfg["collection"],
            "top_k": k,
            "ticker_filter": ticker_filter,
            "filing_filter": filing_mode,
            "answer_prompt": prompt_version,
        },
        "headline": {
            # the "success percent" — strict judge-verified correctness
            "success_percent": pct(correct, total),
            "lenient_success_percent": pct(correct + partial, total),
            "questions_scored": total,
            "questions_in_ground_truth": len(pairs),
            "coverage_percent": pct(total, len(pairs)),
        },
        "answer_metrics": {
            "verdicts": dict(verdicts),
            "correct": correct,
            "partial": partial,
            "incorrect": verdicts["incorrect"],
            "refused": verdicts["refused"],
            "refusal_percent": pct(verdicts["refused"], total),
            "refusal_phrase_percent": pct(
                sum(1 for r in rows if r["refusal_phrase"]), total
            ),
            "mean_numeric_recall": round(sum(recalls) / len(recalls), 3) if recalls else None,
        },
        "retrieval_metrics": {
            f"hit_rate_at_{k}_percent": pct(hits, total),
            "mrr": round(mrr, 3),
            f"passage_hit_rate_at_{k}_percent": pct(passage_hits, len(passage_rows)),
            "passage_mrr": round(passage_mrr, 3),
            "passage_questions": len(passage_rows),
            "note": (
                "Filing-level hit: did any retrieved chunk come from the filing the "
                "ground truth tags? Right filing, wrong section still counts as a hit, "
                "so this is an upper bound on true retrieval quality."
            ),
        },
        "by_difficulty": bucket("difficulty"),
        "by_ticker": bucket("ticker"),
        "filing_inference": inference_summary(rows),
        "ground_truth_health": {
            "entries_missing_expected_answer": missing_expected,
            "entries_not_marked_verified": unverified,
            "tagged_filing_not_in_vector_store": not_indexed,
            "answerable_questions": len(answerable),
            "accuracy_on_answerable_percent": pct(
                sum(1 for r in answerable if r["verdict"] == "correct"), len(answerable)
            ),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=TOP_K, help="chunks to retrieve per question")
    ap.add_argument("--limit", type=int, help="only run the first N questions")
    ap.add_argument("--no-filter", action="store_true", help="disable the ticker metadata filter")
    ap.add_argument("--prompt", choices=sorted(ANSWER_PROMPTS), default="v1", help="answer prompt version")
    ap.add_argument("--embed-model", choices=sorted(EMBED_MODELS), default=DEFAULT_EMBED,
                    help="embedding model / collection to retrieve from")
    ap.add_argument("--filing", choices=["none", "oracle", "inferred"], default="none",
                    help="filing-level filter: none, the ground-truth filing (oracle), or one inferred from the question")
    ap.add_argument("--filing-filter", action="store_true",
                    help="also filter on the ground-truth filing_date (makes filing-level hit/MRR trivially perfect)")
    args = ap.parse_args()

    embed_cfg = EMBED_MODELS[args.embed_model]
    collection, embedder, llm = setup(embed_cfg)
    filing_mode = "oracle" if args.filing_filter else args.filing
    period_map = load_period_map() if filing_mode == "inferred" else None
    use_filter = not args.no_filter

    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(GROUND_TRUTH) as f:
        pairs = json.load(f)["qa_pairs"]
    todo = pairs[: args.limit] if args.limit else pairs

    rows, answers = [], []
    for i, o in enumerate(todo, start=1):
        question, expected = o["question"], o.get("expected_answer", "")
        ticker, filing_date = o.get("ticker"), o.get("filing_date")

        inferred = None
        if filing_mode == "oracle":
            filter_date = filing_date
        elif filing_mode == "inferred":
            # question text only; filing_date from the ground truth is used for scoring, not here.
            # a miss returns None, which leaves the ticker filter alone
            inferred = infer_filing(period_map, ticker, question)
            filter_date = inferred["filing_date"]
        else:
            filter_date = None

        docs, metas, dists = retrieve(
            collection, embedder, question,
            ticker=ticker if use_filter else None,
            filing_date=filter_date,
            k=args.k,
            embed_cfg=embed_cfg,
        )
        retrieval = score_retrieval(metas, ticker, filing_date)
        passage = score_passage(docs, expected)
        inference = None
        if inferred:
            inference = {
                **inferred,
                "matches_ground_truth": filter_date == filing_date,
                "mismatch": mismatch_kind(period_map, ticker, inferred, filing_date),
            }
        context = "\n\n---\n\n".join(docs)

        try:
            answer = generate_answer(llm, context, question, args.prompt)
        except Exception as e:
            print(f"  [gen error] {o['id']}: {e}")
            continue
        try:
            verdict, reason = judge_answer(llm, question, expected, answer)
        except Exception as e:
            print(f"  [judge error] {o['id']}: {e}")
            verdict, reason = "unparsed", str(e)

        recall, matched, wanted = numeric_recall(expected, answer)
        rows.append({
            "id": o["id"],
            "ticker": ticker,
            "difficulty": o.get("difficulty", "unknown"),
            "question": question,
            "expected_answer": expected,
            "answer": answer,
            "verdict": verdict,
            "judge_reason": reason,
            "numeric_recall": recall,
            "figures_matched": matched,
            "figures_expected": wanted,
            "refusal_phrase": looks_like_refusal(answer),
            "filing_indexed": filing_is_indexed(collection, ticker, filing_date),
            "retrieval": retrieval,
            "passage": passage,
            "filing_inference": inference,
        })
        answers.append({"question": question, "answer": answer})

        flag = "hit" if retrieval["hit"] else "MISS"
        print(f"{i}/{len(todo)} {o['id']:<10} {verdict:<9} retrieval={flag:<4} rank={retrieval['rank']}")

        with open(RESULTS_OUT, "w") as f:
            json.dump(rows, f, indent=2)
        with open(ANSWERS_OUT, "w") as f:
            json.dump(answers, f, indent=2)
        time.sleep(0.2)

    metrics = aggregate(rows, pairs, args.k, use_filter, filing_mode, args.prompt, embed_cfg)
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    h, a, r = metrics["headline"], metrics["answer_metrics"], metrics["retrieval_metrics"]
    print("\n" + "=" * 60)
    print(f"success (strict)     : {h['success_percent']}%  ({a['correct']}/{h['questions_scored']})")
    print(f"success (+partial)   : {h['lenient_success_percent']}%")
    print(f"refused              : {a['refusal_percent']}%")
    print(f"retrieval hit@{args.k}     : {r[f'hit_rate_at_{args.k}_percent']}%   MRR {r['mrr']}")
    print(f"passage hit@{args.k}       : {r[f'passage_hit_rate_at_{args.k}_percent']}%   MRR {r['passage_mrr']}   (n={r['passage_questions']})")
    if metrics["filing_inference"]:
        fi = metrics["filing_inference"]
        print(f"filing inference     : {fi['match_percent']}% match ground truth  "
              f"({fi['resolved']} resolved, {fi['unresolved_fell_back_to_ticker_filter']} fell back to ticker filter)")
    print(f"coverage of GT set   : {h['coverage_percent']}%  ({h['questions_scored']}/{h['questions_in_ground_truth']})")
    print(f"\nby difficulty:")
    for name, d in metrics["by_difficulty"].items():
        print(f"  {name:<14} {d['accuracy_percent']:>5}% acc   {d['retrieval_hit_percent']:>5}% hit   (n={d['questions']})")
    print(f"\nwrote {METRICS_OUT}, {RESULTS_OUT}, {ANSWERS_OUT}")


if __name__ == "__main__":
    main()
