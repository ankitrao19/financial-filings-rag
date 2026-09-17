"""
Tests for filing_inference.py, one section per half so each can be checked on its own.

    python tests/test_filing_inference.py      # no dependencies
    pytest tests/                              # also works if pytest is installed

Half A tests read the saved data/filing-periods.json (run `python pipeline/filing_inference.py --build` first).
Half B and meeting-point tests are pure: fixed strings and a tiny hand-made map, no corpus.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from filing_inference import (  # noqa: E402
    fiscal_period, infer_filing, load_period_map, make_key, mismatch_kind, parse_question,
)

# fiscal year-end months used below
DEC, SEP, AUG, JAN, JUN = 12, 9, 8, 1, 6


# ---------------------------------------------------------------------------
# Shared arithmetic
# ---------------------------------------------------------------------------
def test_fiscal_period_calendar_year():
    assert fiscal_period(date(2026, 3, 31), DEC) == (2026, 1)
    assert fiscal_period(date(2026, 6, 30), DEC) == (2026, 2)
    assert fiscal_period(date(2025, 12, 31), DEC) == (2025, None)


def test_fiscal_period_offset_calendars():
    # NVIDIA: year ends late January -> April quarter is Q1 of the NEXT fiscal year
    assert fiscal_period(date(2026, 4, 26), JAN) == (2027, 1)
    assert fiscal_period(date(2026, 1, 25), JAN) == (2026, None)
    # Apple: year ends late September -> December quarter is Q1
    assert fiscal_period(date(2025, 12, 27), SEP) == (2026, 1)
    # Costco: year ends ~Aug 31 / first days of Sep -> both normalise to August
    assert fiscal_period(date(2024, 9, 1), AUG) == (2024, None)
    assert fiscal_period(date(2026, 5, 10), AUG) == (2026, 3)


def test_first_week_dates_roll_back_a_month():
    # Coca-Cola Q1 ends Apr 3 -> treated as the March quarter-end
    assert fiscal_period(date(2026, 4, 3), DEC) == (2026, 1)


# ---------------------------------------------------------------------------
# Half A — period map (reads filing-periods.json)
# ---------------------------------------------------------------------------
def test_map_is_complete_and_consistent():
    pm = load_period_map()
    assert pm["problems"] == []
    assert len(pm["filings"]) == len(pm["index"]) == 90  # one unique key per filing
    for x in pm["filings"]:
        assert (x["form"] == "10-K") == (x["quarter"] is None), x


def test_map_known_filings():
    idx = load_period_map()["index"]
    assert idx["COST|2024|annual"] == "2024-10-09"
    assert idx["NVDA|2026|annual"] == "2026-02-25"
    assert idx["NVDA|2027|Q1"] == "2026-05-20"
    assert idx["AAPL|2026|Q1"] == "2026-01-30"
    assert idx["KO|2026|Q1"] == "2026-04-30"
    assert idx["GS|2026|Q1"] == "2026-05-01"   # cover date unreadable -> filing-date fallback
    assert idx["JPM|2024|annual"] == "2025-02-14"


def test_map_fiscal_year_end_months():
    fy = load_period_map()["fiscal_year_end_month"]
    assert (fy["AAPL"], fy["COST"], fy["NVDA"], fy["MSFT"], fy["NKE"], fy["GS"]) == (9, 8, 1, 6, 5, 12)


# ---------------------------------------------------------------------------
# Half B — question parser (pure text)
# ---------------------------------------------------------------------------
def test_parse_fiscal_year():
    assert parse_question("What was NVIDIA's revenue in fiscal 2026?", JAN) == \
        {"period": "annual", "year": 2026, "quarter": None, "date": None}
    assert parse_question("What were Amazon's total net sales in 2023?")["year"] == 2023
    assert parse_question("Revenue for FY24?")["year"] == 2024


def test_parse_comparison_takes_latest_year():
    p = parse_question("How did Costco's net income change from fiscal 2022 to fiscal 2024?", AUG)
    assert (p["period"], p["year"]) == ("annual", 2024)


def test_parse_quarter_with_year():
    for q in ["net interest income in Q1 2026", "total net revenues in the first quarter of 2026"]:
        assert parse_question(q, DEC) == {"period": "quarter", "year": 2026, "quarter": 1, "date": None}


def test_parse_quarter_without_year():
    p = parse_question("How did Q1 total net revenues compare to the prior-year quarter?", DEC)
    assert (p["period"], p["year"], p["quarter"]) == ("quarter", None, 1)


def test_parse_stated_date_uses_company_calendar():
    q = "What was net income in the quarter ended June 30, 2026?"
    assert parse_question(q, SEP) == {"period": "quarter", "year": 2026, "quarter": 3, "date": "2026-06-30"}   # Visa
    assert parse_question(q, DEC) == {"period": "quarter", "year": 2026, "quarter": 2, "date": "2026-06-30"}   # calendar
    assert parse_question(q, JUN)["period"] == "annual"                                                       # Microsoft year-end


def test_parse_stated_date_on_year_end_is_annual():
    p = parse_question("How many warehouses did Costco operate as of September 1, 2024?", AUG)
    assert (p["period"], p["year"]) == ("annual", 2024)


def test_parse_q4_folds_into_annual():
    assert parse_question("Revenue in Q4 2025?", DEC) == {"period": "annual", "year": 2025, "quarter": None, "date": None}


def test_parse_no_period():
    assert parse_question("What risks does Amazon disclose about competition?")["period"] is None


# ---------------------------------------------------------------------------
# Meeting point — lookup + miss reasons (tiny hand-made map)
# ---------------------------------------------------------------------------
TOY = {
    "fiscal_year_end_month": {"ACME": 12},
    "index": {
        make_key("ACME", 2025, None): "2026-02-20",
        make_key("ACME", 2025, 1): "2025-05-01",
        make_key("ACME", 2026, 1): "2026-05-01",
    },
}


def test_infer_resolved():
    r = infer_filing(TOY, "ACME", "Net income in fiscal 2025?")
    assert (r["filing_date"], r["key"], r["reason"]) == ("2026-02-20", "ACME|2025|annual", "resolved")


def test_infer_quarter_latest_when_year_missing():
    r = infer_filing(TOY, "ACME", "Q1 revenue versus the prior-year quarter?")
    assert (r["filing_date"], r["reason"]) == ("2026-05-01", "quarter_latest_year")


def test_infer_misses_fall_back_with_reason():
    assert infer_filing(TOY, "ACME", "What risks does ACME disclose?")["reason"] == "miss_no_period"
    assert infer_filing(TOY, "ACME", "Net income in fiscal 2019?")["reason"] == "miss_not_in_corpus"
    assert infer_filing(TOY, "NOPE", "Net income in fiscal 2025?")["reason"] == "miss_unknown_ticker"
    for q in ["What risks does ACME disclose?", "Net income in fiscal 2019?"]:
        assert infer_filing(TOY, "ACME", q)["filing_date"] is None


def test_mismatch_kinds():
    annual25 = infer_filing(TOY, "ACME", "Net income in fiscal 2025?")
    assert mismatch_kind(TOY, "ACME", annual25, "2026-02-20") == "match"
    assert mismatch_kind(TOY, "ACME", annual25, "2025-05-01") == "wrong_period"            # same year, Q1 vs annual
    q1_26 = infer_filing(TOY, "ACME", "Revenue in Q1 2026?")
    assert mismatch_kind(TOY, "ACME", q1_26, "2025-05-01") == "wrong_year"                 # Q1 both, year differs
    assert mismatch_kind(TOY, "ACME", annual25, "2026-05-01") == "wrong_year_and_period"
    miss = infer_filing(TOY, "ACME", "What risks does ACME disclose?")
    assert mismatch_kind(TOY, "ACME", miss, "2026-02-20") == "fallback_miss_no_period"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
