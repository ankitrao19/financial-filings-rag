"""
filing_inference.py — work out which filing a question is about, from the question text alone.

Two independent halves that meet in the middle:

  Half A — period map (data side). Built once from the corpus and saved to data/filing-periods.json.
      Every filing's cover page states the period it covers ("For the fiscal year ended
      September 28, 2024"); from that date + the company's fiscal year-end we derive
      fiscal year and quarter, and index each filing under a key:
          "TICKER|FISCAL_YEAR|annual"   -> 10-K filing_date
          "TICKER|FISCAL_YEAR|Q1..Q3"   -> 10-Q filing_date

  Half B — question parser (query side). Pure text -> {year, quarter, period, date}.
      Knows nothing about the corpus; a stated date ("quarter ended June 30, 2026") is turned
      into a fiscal quarter with the company's fiscal calendar, supplied by Half A.

  Meeting point — infer_filing(): parse the question, build the key, look it up.
      A miss never raises: it returns filing_date=None with a reason, and retrieval falls back
      to the ticker filter. Reasons are logged so parser fixes can target them.

Fiscal conventions (hold for every company in the corpus):
  - fiscal year = calendar year in which the fiscal year ENDS (NVDA FY2026 ends Jan 25, 2026);
  - a period ending in the first week of a month belongs to the previous month
    (Costco FY2024 ends Sep 1, 2024; Coca-Cola Q1 ends Apr 3, 2026);
  - fiscal Q1-Q3 are 10-Qs; Q4 has no 10-Q, so it is covered by the annual 10-K.

Usage:
    python pipeline/filing_inference.py --build    # (re)build data/filing-periods.json from data/corpus
    python pipeline/filing_inference.py --show     # print the map
    python pipeline/filing_inference.py --check    # resolve every ground-truth question, report misses and why
"""

import argparse
import calendar
import json
import re
from datetime import date, timedelta
from pathlib import Path

from paths import CORPUS_DIR, GROUND_TRUTH, PERIOD_MAP_PATH

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTH_ALT = "|".join(MONTHS)


# ---------------------------------------------------------------------------
# Shared fiscal-calendar arithmetic (used by both halves)
# ---------------------------------------------------------------------------
def month_key(d):
    """Normalise a period-end date to its (year, month); first-week dates roll back a month."""
    if d.day <= 7:
        d = d.replace(day=1) - timedelta(days=1)
    return d.year, d.month


def fiscal_period(period_end, fy_end_month):
    """(fiscal_year, quarter) for a period ending on period_end. quarter is 1-3, or None
    when the period ends on the fiscal year-end (i.e. the annual report)."""
    year, month = month_key(period_end)
    fiscal_year = year if month <= fy_end_month else year + 1
    months_in = (month - fy_end_month) % 12  # 0 at year-end, 3/6/9 at quarter-ends
    quarter = round(months_in / 3)
    return fiscal_year, (quarter if quarter in (1, 2, 3) else None)


def period_label(quarter):
    return f"Q{quarter}" if quarter else "annual"


def make_key(ticker, fiscal_year, quarter):
    return f"{ticker}|{fiscal_year}|{period_label(quarter)}"


# ---------------------------------------------------------------------------
# Half A — period map from the corpus
# ---------------------------------------------------------------------------
# the date sits anywhere from right after "ended" to a few table cells later on messy cover pages
COVER_RE = re.compile(
    r"(fiscal year|year|quarterly period|period) ended\W{0,40}?(?:[^|\n]{0,40}\|\W*){0,6}?"
    r"(" + MONTH_ALT + r")\.?\s+(\d{1,2})\s*,\s*(\d{4})",
    re.IGNORECASE,
)


def quarter_end_before(d, min_gap_days=20):
    """Latest calendar quarter-end at least min_gap_days before d."""
    d = d - timedelta(days=min_gap_days)
    q_month = ((d.month - 1) // 3) * 3
    if q_month == 0:
        return date(d.year - 1, 12, 31)
    return date(d.year, q_month, calendar.monthrange(d.year, q_month)[1])


def read_period_end(chunks, filed):
    """Period end stated on the cover page, or (fallback) the last quarter-end before filing."""
    head = " ".join(c["text"] for c in chunks[:3])
    m = COVER_RE.search(head)
    if m:
        try:
            period = date(int(m.group(4)), MONTHS[m.group(2).lower()], int(m.group(3)))
            if filed - timedelta(days=200) <= period < filed:
                return period, "cover"
        except ValueError:
            pass
    # exact for calendar-year filers, which is every company this fires for in the corpus
    return quarter_end_before(filed), "filing_date"


def build_period_map(corpus_dir=CORPUS_DIR):
    filings = []
    for f in sorted(corpus_dir.glob("*.json")):
        chunks = json.loads(f.read_text(encoding="utf-8"))
        if not chunks:
            continue
        meta = chunks[0]["metadata"]
        period, source = read_period_end(chunks, date.fromisoformat(meta["filing_date"]))
        filings.append({
            "ticker": meta["ticker"], "form": meta["form"], "filing_date": meta["filing_date"],
            "period_end": period.isoformat(), "period_source": source,
        })

    # each company's fiscal year-end month comes from its most recent 10-K
    fy_end_month = {}
    for x in sorted(filings, key=lambda x: x["filing_date"]):
        if x["form"] == "10-K":
            fy_end_month[x["ticker"]] = month_key(date.fromisoformat(x["period_end"]))[1]

    index, problems = {}, []
    for x in filings:
        fy, quarter = fiscal_period(date.fromisoformat(x["period_end"]), fy_end_month.get(x["ticker"], 12))
        x["fiscal_year"], x["quarter"] = fy, quarter
        if (x["form"] == "10-K") != (quarter is None):
            problems.append(f"{x['ticker']} {x['form']} {x['filing_date']}: period {x['period_end']} maps to {period_label(quarter)}")
        key = make_key(x["ticker"], fy, quarter)
        if key in index:
            problems.append(f"duplicate key {key}: {index[key]} and {x['filing_date']}")
        index[key] = x["filing_date"]

    return {
        "source": "data/corpus",
        "fiscal_year_end_month": fy_end_month,
        "index": dict(sorted(index.items())),
        "filings": sorted(filings, key=lambda x: (x["ticker"], x["filing_date"])),
        "problems": problems,
    }


def save_period_map(period_map, path=PERIOD_MAP_PATH):
    Path(path).write_text(json.dumps(period_map, indent=2))


def load_period_map(path=PERIOD_MAP_PATH):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"{path} missing — build it with: python pipeline/filing_inference.py --build")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Half B — question parser
# ---------------------------------------------------------------------------
DATE_RE = re.compile(r"\b(" + MONTH_ALT + r")\.?\s+(\d{1,2})\s*,\s*(\d{4})\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(?:FY\s?)?(20\d{2})\b|\bFY\s?(\d{2})\b", re.IGNORECASE)
QUARTER_NUM_RE = re.compile(
    r"\bQ([1-4])\b|\b(first|second|third|fourth)\s+(?:fiscal\s+)?quarter\b", re.IGNORECASE
)
QUARTER_CUE_RE = re.compile(r"\bquarter|\bthree months\b|\bQ[1-4]\b", re.IGNORECASE)
ANNUAL_CUE_RE = re.compile(r"\bfiscal year\b|\bannual\b|\bfull[- ]year\b|\byear ended\b", re.IGNORECASE)
QUARTER_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def parse_question(question, fy_end_month=12):
    """Question text -> period cues. Every field may be None.

    period  : "quarter" | "annual" | None
    year    : fiscal year; the LATEST year named, since a comparison ("2022 to 2024") is answered
              by the later filing, which carries earlier years as comparative columns
    quarter : 1-3 for a 10-Q period; None for annual (Q4 is folded into annual)
    date    : explicit date in the question, if any
    """
    dates = []
    for m in DATE_RE.finditer(question):
        try:
            dates.append(date(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2))))
        except ValueError:
            pass
    stated = max(dates) if dates else None

    years = {int(m.group(1)) if m.group(1) else 2000 + int(m.group(2)) for m in YEAR_RE.finditer(question)}

    quarter = None
    m = QUARTER_NUM_RE.search(question)
    if m:
        quarter = int(m.group(1)) if m.group(1) else QUARTER_WORDS[m.group(2).lower()]

    if stated:
        # a stated date pins year and quarter through the company's own fiscal calendar
        fiscal_year, date_quarter = fiscal_period(stated, fy_end_month)
        return {"period": "quarter" if date_quarter else "annual", "year": fiscal_year,
                "quarter": date_quarter, "date": stated.isoformat()}

    if QUARTER_CUE_RE.search(question) and quarter != 4:
        period = "quarter"
    elif years or ANNUAL_CUE_RE.search(question) or quarter == 4:
        period, quarter = "annual", None
    else:
        period = None

    return {"period": period, "year": max(years) if years else None,
            "quarter": quarter if period == "quarter" else None, "date": None}


# ---------------------------------------------------------------------------
# Meeting point — question -> filing_date
# ---------------------------------------------------------------------------
def infer_filing(period_map, ticker, question):
    """Returns {"filing_date", "key", "reason", "parsed"}. filing_date is None on any miss;
    the caller then filters on ticker alone. reason is one of:
      resolved                 key found in the map
      quarter_latest_year      "Q1 ..." with no year -> most recent Q1 filing
      miss_no_period           question names no year, quarter or date (parse miss)
      miss_quarter_without_num "quarter" cue but no quarter number and no year
      miss_not_in_corpus       key parsed fine but that filing isn't indexed
      miss_unknown_ticker      ticker has no filings in the map
    """
    fy_end_month = period_map["fiscal_year_end_month"].get(ticker)
    if fy_end_month is None:
        return {"filing_date": None, "key": None, "reason": "miss_unknown_ticker", "parsed": None}

    parsed = parse_question(question, fy_end_month)
    index = period_map["index"]
    out = {"filing_date": None, "key": None, "reason": None, "parsed": parsed}

    if parsed["period"] is None:
        out["reason"] = "miss_no_period"
        return out

    if parsed["year"] is None:
        if parsed["period"] == "quarter" and parsed["quarter"]:
            suffix = f"|{period_label(parsed['quarter'])}"
            hits = sorted(k for k in index if k.startswith(f"{ticker}|") and k.endswith(suffix))
            if hits:
                out.update(key=hits[-1], filing_date=index[hits[-1]], reason="quarter_latest_year")
                return out
        out["reason"] = "miss_quarter_without_num" if parsed["period"] == "quarter" else "miss_no_period"
        return out

    key = make_key(ticker, parsed["year"], parsed["quarter"])
    out["key"] = key
    if key in index:
        out.update(filing_date=index[key], reason="resolved")
    else:
        out["reason"] = "miss_not_in_corpus"
    return out


def key_for_filing(period_map, ticker, filing_date):
    """Reverse lookup: the key a filing is indexed under (used to explain mismatches)."""
    for key, fd in period_map["index"].items():
        if fd == filing_date and key.startswith(f"{ticker}|"):
            return key
    return None


def mismatch_kind(period_map, ticker, inferred, expected_filing_date):
    """Why an inferred filing differs from the ground-truth one:
    match | fallback_<miss reason> | wrong_year | wrong_period (quarter vs annual, or wrong quarter)."""
    if inferred["filing_date"] == expected_filing_date:
        return "match"
    if inferred["filing_date"] is None:
        return f"fallback_{inferred['reason']}"
    want = key_for_filing(period_map, ticker, expected_filing_date)
    got = inferred["key"] or key_for_filing(period_map, ticker, inferred["filing_date"])
    if not want or not got:
        return "expected_filing_not_in_map"
    _, want_year, want_period = want.split("|")
    _, got_year, got_period = got.split("|")
    return "wrong_period" if want_year == got_year and want_period != got_period else (
        "wrong_year" if want_period == got_period else "wrong_year_and_period")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="build data/filing-periods.json from data/corpus")
    ap.add_argument("--show", action="store_true", help="print the period map")
    ap.add_argument("--check", action="store_true", help="resolve every ground-truth question")
    args = ap.parse_args()

    if args.build:
        pm = build_period_map()
        save_period_map(pm)
        sources = {}
        for x in pm["filings"]:
            sources[x["period_source"]] = sources.get(x["period_source"], 0) + 1
        print(f"wrote {PERIOD_MAP_PATH}: {len(pm['filings'])} filings, {len(pm['index'])} keys, period source {sources}")
        for p in pm["problems"]:
            print(f"  [problem] {p}")

    pm = load_period_map()
    if args.show:
        for x in pm["filings"]:
            print(f"{x['ticker']:5} {x['form']} filed {x['filing_date']}  period {x['period_end']} "
                  f"({x['period_source']:11})  -> {make_key(x['ticker'], x['fiscal_year'], x['quarter'])}")

    if args.check:
        pairs = json.load(open(GROUND_TRUTH))["qa_pairs"]
        kinds = {}
        for o in pairs:
            inf = infer_filing(pm, o["ticker"], o["question"])
            kind = mismatch_kind(pm, o["ticker"], inf, o["filing_date"])
            kinds[kind] = kinds.get(kind, 0) + 1
            if kind != "match":
                print(f"{o['id']:10} {kind:28} inferred {inf['filing_date']} ({inf['key']})  "
                      f"gt {o['filing_date']} ({key_for_filing(pm, o['ticker'], o['filing_date'])})  | {o['question']}")
        print(kinds)


if __name__ == "__main__":
    main()
