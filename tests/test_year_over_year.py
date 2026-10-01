"""Offline check: year-over-year evolution of ONE training type.

Script-style like its siblings: run it directly, it asserts and prints.
The point of this test is the HONESTY RULES, not the numbers — that no drawn
point sits in a cell below the floor, that no delta exists unless both of its
years clear it, that the loader's own blank-label test is the one being used,
and that the view refuses to claim a year-over-year WATTS comparison that the
measurements cannot support.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)

import pandas as pd

from core.data import load_data, BLANK_TYPE_TOKENS
from core.theme import MAIN_TYPES
from ml import year_over_year as yoy

df_all = load_data()
s = yoy.prep(df_all)
print(f"sessions: {len(df_all):,} loaded, {len(s):,} carry a training type, "
      f"{len(df_all) - len(s):,} unlabelled and excluded")

# ── the blank-label test must be the LOADER's, not a second copy ────────────
_lab = df_all["training_type"].astype(object)
_core_keep = int((_lab.notna() & ~_lab.astype(str).str.strip().isin(BLANK_TYPE_TOKENS)).sum())
assert len(s) == _core_keep, (
    f"prep kept {len(s)} but core.data's predicate keeps {_core_keep} — the two "
    f"blank-label tests have drifted apart")
print(f"blank-label parity with core.data: {len(s)} == {_core_keep}  OK")

# ── coverage: what the picker offers, and in what order ────────────────────
scope = yoy.types_in_scope(s, MAIN_TYPES)
print("\n=== types in scope, most comparable first ===")
print(scope[["training_type", "n_sessions", "n_drawable", "years",
             "comparable"]].to_string(index=False))
assert set(scope["training_type"]) == set(MAIN_TYPES), \
    "every first-class type with sessions must be offered, and only those"
# the ordering claim is the point: more comparable years first
_ord = scope[scope["comparable"]]
assert list(_ord["years"]) == sorted(_ord["years"], reverse=True), \
    "types are not ordered by comparable years"
print("ordering by comparable years: OK")

# ── the floor ──────────────────────────────────────────────────────────────
print(f"\n=== the {yoy.MIN_CELL_N}-session floor, checked on every type x metric ===")
n_ev = 0
for tt in MAIN_TYPES:
    for metric in yoy.METRIC_KEYS:
        ev = yoy.evolution(s, tt, metric)
        if not len(ev):
            continue
        n_ev += len(ev)
        bad = ev[ev["n"] < yoy.MIN_CELL_N]
        assert not len(bad), f"{tt}/{metric}: drew a point on n<{yoy.MIN_CELL_N}"
        bad_m = ev[ev["n_metric"] < yoy.MIN_CELL_N]
        assert not len(bad_m), (
            f"{tt}/{metric}: drew a point where only {bad_m['n_metric'].min()} "
            f"of {yoy.MIN_CELL_N} sessions carry the measure")
        # one row per (duration class, year) — never two, never a mix
        assert not ev.duplicated(["dur_class", "year"]).any(), \
            f"{tt}/{metric}: duplicate cell rows"
        assert set(ev["dur_class"].astype(str)) <= set(yoy.DUR_ORDER), \
            f"{tt}/{metric}: a duration class outside the declared set"
print(f"{n_ev:,} drawn points across every type and metric, all on cells that "
      f"cleared the floor: OK")

# ── deltas need BOTH years, and must state the gap ──────────────────────────
print("\n=== 'vs previous year' deltas ===")
n_del = 0
for tt in MAIN_TYPES:
    t = yoy.yoy_table(s, tt, "power_avg")
    if not len(t):
        continue
    d = t[t["delta"].notna()]
    n_del += len(d)
    for r in d.itertuples():
        assert r.n >= yoy.MIN_CELL_N and r.prev_n >= yoy.MIN_CELL_N, (
            f"{tt}: a delta exists on a thin side (n={r.n}, prev n={r.prev_n})")
        assert r.gap_years == r.year - r.prev_year, \
            f"{tt}: stated gap does not match the two years compared"
        assert r.gap_years >= 1, f"{tt}: delta against a non-earlier year"
print(f"{n_del:,} deltas, each with both years above the floor and the year gap "
      f"stated correctly: OK")

# BILLAT end to end, the type the athlete reclassified to enable this comparison
print("\n=== BILLAT (the reclassified type) ===")
cov = yoy.coverage(s, "BILLAT")
print(cov.pivot(index="dur_class", columns="year", values="n").to_string())
t = yoy.yoy_table(s, "BILLAT", "if_score")
t = t[t["drawable"]]
print("\nIntensity factor, 90-150 min, year over year:")
for r in t[t["dur_class"].astype(str) == "90-150 min"].itertuples():
    prev = "—" if pd.isna(r.prev_year) else f"{int(r.prev_year)} (n={int(r.prev_n)})"
    print(f"  {int(r.year)}  n={int(r.n):2}  IF {yoy.fmt_value('if_score', r.value)}"
          f"   vs {prev:<16} change {yoy.fmt_delta('if_score', r.delta)}")
assert len(t[t["dur_class"].astype(str) == "90-150 min"]) >= 4, \
    "BILLAT lost its year-over-year row"

# ── exact durations, minutes whole, seconds always shown ───────────────────
print("\n=== duration formatting ===")
for secs, want in [(0, "0:00:00"), (59, "0:00:59"), (90, "0:01:30"),
                   (3600, "1:00:00"), (8439, "2:20:39"), (25239, "7:00:39")]:
    got = yoy.fmt_dur(secs)
    assert got == want, f"fmt_dur({secs}) = {got}, expected {want}"
print("h:mm:ss exact, seconds never rounded away: OK")

# ── thousands separators on the big numbers ────────────────────────────────
assert yoy.fmt_value("tss", 1234567.4) == "1,234,567.4", yoy.fmt_value("tss", 1234567.4)
# watts carry no decimals, so a delta rounds to whole watts
assert yoy.fmt_delta("power_avg", -12345.6) == "-12,346", yoy.fmt_delta("power_avg", -12345.6)
assert yoy.fmt_delta("if_score", 0.0584) == "+0.058"
assert yoy.fmt_value("if_score", float("nan")) == "—"
print("thousands separators and blank placeholders: OK")

# ── the outcome wall ───────────────────────────────────────────────────────
print("\n=== threshold watts: what may NOT be claimed ===")
g = yoy.watts_gate(s)
print(f"  measured readings : {g['n_measured']:,}")
print(f"  by year           : {g['by_year']}")
print(f"  months per year   : {g['by_year_months']}")
print(f"  window            : {g['first_month']} -> {g['last_month']} "
      f"({g['n_months']} months)")
print(f"  years with none   : {g['years_without']}")
print(f"  comparable years  : {g['comparable_years']}")
print(f"  can compare years : {g['can_compare_years']}")
assert not g["can_compare_years"], \
    "the gate claimed a year-over-year watts comparison is possible"
assert 2025 not in g["comparable_years"], \
    "2025 was called comparable on 3 readings inside one December week"
assert g["first_month"] == "2025-12", f"window starts {g['first_month']}"
assert g["n_months"] <= 10, "the threshold window has grown beyond Dec 2025"
print("gate refuses the comparison and explains why: OK")

print("\nAll year-over-year evolution checks passed.")
