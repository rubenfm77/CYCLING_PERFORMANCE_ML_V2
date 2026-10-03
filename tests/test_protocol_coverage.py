"""Every selected set must reach the bars or the session line — no silent drop.

The complaint behind this file: on the Intervals page, "FTP · 20 min" said
4 sessions in the picker and then showed ONE — 17 Apr 2026 — with no other
dates on the axis, while Evolution showed all four. The cause was in
`run_protocol_view`: it walked back to the raw interval rows the detector
segmented, and a set with no member row (intervals.icu's peak-power meter
reports a held window outside the interval list) fell out of BOTH the bars
and the session frame.

Rules pinned here:

  1. a single-effort set with no member rows becomes ONE bar — its watts and
     its length are the measurement itself, and nothing is invented for it;
  2. a set of several reps with no member rows gets NO bars (one bar per rep
     would be an invention) but its DAY still rides on the session line at
     its own average watts, with the interval count blank rather than guessed;
  3. one row per date, ever — the session mean of a date equals the plain
     mean of that date's bars (the invariant test_protocol_reps asserts);
  4. the source of every bar is carried and printed, so the detector's rows
     and the meter's readings are never mistaken for one instrument.

Two halves: synthetic (no files, no credentials) and this athlete's own
cache, skipped cleanly when data/interval_cache.csv is absent (gitignored).

Run:  python tests/test_protocol_coverage.py     (exit 0 = pass, 1 = fail)
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
from pathlib import Path

from ml.set_evolution import DETECTED_SRC, PEAK_SRC, SET_COLUMNS
from ml.protocol_reps import run_protocol_view

print("=" * 74)
print("1. the rules, on synthetic sets and rows")
print("=" * 74)


def _iv(rows):
    """A detector-shaped interval frame: WORK rows only, no session context
    (temp/TSB/ride name live on the SETS table and are merged in)."""
    base = {c: np.nan for c in
            ("secs", "avg_w", "np_w", "hr_avg", "intensity", "load",
             "cad_avg", "group_id")}
    out = pd.DataFrame([{**base, **r} for r in rows])
    out["iv_type"] = "WORK"
    out["date"] = pd.to_datetime(out["date"])
    for c in ("secs", "avg_w", "np_w", "hr_avg", "intensity", "load",
              "cad_avg"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["seq"] = np.arange(1, len(out) * 2, 2, dtype=float)
    return out


def _sets(rows):
    """A sets-shaped frame: one row per (activity, length)."""
    f = pd.DataFrame([{**{c: np.nan for c in SET_COLUMNS}, **r}
                      for r in rows])
    f["date"] = pd.to_datetime(f["date"])
    if "act_start" in f.columns:
        f["act_start"] = pd.to_datetime(f["act_start"])
    if "db_type" in f.columns:
        f["db_type"] = f["db_type"].fillna("").astype(str)
    if "name" in f.columns:
        f["name"] = f["name"].fillna("").astype(str)
    f["family"] = "FTP"
    f["dur_b"] = 20.0
    return f


# act A: the detector segmented the 20-minute block into three reps
# act B: the detector cut the ride into shorter pieces, and the peak meter
#        holds the 20:00 window — one held effort, no member rows
iv = _iv([
    {"activity_id": "A", "date": "2026-04-17", "group_id": "g1",
     "secs": 1206, "avg_w": 209.0, "np_w": 214.0, "hr_avg": 151.0,
     "intensity": 89.0, "cad_avg": 88.0},
    {"activity_id": "A", "date": "2026-04-17", "group_id": "g1",
     "secs": 1204, "avg_w": 213.0, "np_w": 218.0, "hr_avg": 156.0,
     "intensity": 91.0, "cad_avg": 87.0},
    {"activity_id": "A", "date": "2026-04-17", "group_id": "g1",
     "secs": 1207, "avg_w": 216.0, "np_w": 221.0, "hr_avg": 159.0,
     "intensity": 92.0, "cad_avg": 86.0},
    {"activity_id": "B", "date": "2026-09-15", "group_id": "266s@209w91rpm",
     "secs": 266, "avg_w": 209.0, "np_w": 212.0, "hr_avg": 148.0,
     "intensity": 88.0, "cad_avg": 91.0},
    {"activity_id": "B", "date": "2026-09-15", "group_id": "310s@201w90rpm",
     "secs": 310, "avg_w": 201.0, "np_w": 205.0, "hr_avg": 150.0,
     "intensity": 85.0, "cad_avg": 90.0},
])
sets = _sets([
    {"activity_id": "A", "date": "2026-04-17", "_key": "g1", "reps": 3,
     "rep_secs": 1206.0, "set_w": 212.666667, "intensity": 90.666667,
     "Source": DETECTED_SRC, "name": "sweet spot",
     "temp": 19.4, "tsb": 5.0},
    {"activity_id": "B", "date": "2026-09-15", "_key": "peak:B:1200s",
     "reps": 1, "rep_secs": 1200.0, "set_w": 231.0, "intensity": np.nan,
     "Source": PEAK_SRC, "name": "FTP test", "temp": 17.4, "tsb": -3.0},
])

pv = run_protocol_view(iv, sets, "FTP", 20.0)
assert pv["ok"], pv.get("reason")
bars, sess = pv["bars"], pv["sessions"]

assert len(sess) == 2, f"two selected sets -> {len(sess)} sessions"
assert sorted(pd.to_datetime(sess["date"]).dt.strftime("%Y-%m-%d")) == \
    ["2026-04-17", "2026-09-15"], "a date is missing from the session line"
assert pv["n_sessions"] == 2 and pv["n_reps"] == 4, \
    f"{pv['n_sessions']} sessions / {pv['n_reps']} intervals"
print("   PASS  the peak-meter session reached the session line and the axis")

# one bar for the held window, three for the detected block — no invention
assert len(bars) == 4, f"{len(bars)} bars, expected 3 + 1"
per_day = bars.groupby(pd.to_datetime(bars["date"]).dt.strftime("%Y-%m-%d"))
assert per_day.size().to_dict() == {"2026-04-17": 3, "2026-09-15": 1}
peak_bar = bars[bars["src"] == PEAK_SRC]
assert len(peak_bar) == 1, "the meter reading is not one bar"
assert float(peak_bar["secs"].iloc[0]) == 1200.0
assert float(peak_bar["avg_w"].iloc[0]) == 231.0
assert set(bars["src"]) == {DETECTED_SRC, PEAK_SRC}, sorted(set(bars["src"]))
print("   PASS  one bar per held window, three for the block, sources kept")

# the invariant test_protocol_reps asserts: a session mean IS the plain mean
# of that day's bars, and no date appears twice
assert not sess["date"].duplicated().any(), "two rows for one date"
chk = bars.groupby("date")["avg_w"].mean().round(6)
ref = sess.set_index("date")["w"].round(6)
d = (chk - ref.reindex(chk.index)).abs().max()
assert d < 1e-6, f"session mean mismatch {d}"
print("   PASS  one row per date, session mean == mean of that day's bars")

# the trend now spans both days, and the carried reading counts as one rep
assert int(sess["reps"].sum()) == 4, "rep count lost a bar"
assert pd.Timestamp(sess["date"].iloc[0]) == pd.Timestamp("2026-04-17")
assert pd.Timestamp(sess["date"].iloc[-1]) == pd.Timestamp("2026-09-15")
assert pv["trend"]["first"] == 212.666667 or abs(pv["trend"]["first"] -
                                                  212.666667) < 1e-6
assert abs(pv["trend"]["last"] - 231.0) < 1e-6, pv["trend"]["last"]
print("   PASS  first and last session are the real ends of the series")

# ── a set of several reps whose rows are gone: no bars, but no lost day ────
iv_pruned = iv[iv["activity_id"] == "A"]
sets2 = _sets([
    {"activity_id": "A", "date": "2026-04-17", "_key": "g1", "reps": 3,
     "rep_secs": 1206.0, "set_w": 212.666667, "intensity": 90.666667,
     "Source": DETECTED_SRC, "name": "sweet spot",
     "temp": 19.4, "tsb": 5.0},
    # rows pruned: three reps, average known, no member rows to draw from
    {"activity_id": "C", "date": "2026-07-23", "_key": "g3", "reps": 3,
     "rep_secs": 602.0, "set_w": 250.0, "intensity": np.nan,
     "Source": DETECTED_SRC, "name": "threshold", "temp": 21.0, "tsb": 2.0},
])
pv2 = run_protocol_view(iv_pruned, sets2, "FTP", 20.0)
assert pv2["ok"]
s2, b2 = pv2["sessions"], pv2["bars"]
dates2 = sorted(pd.to_datetime(s2["date"]).dt.strftime("%Y-%m-%d"))
assert dates2 == ["2026-04-17", "2026-07-23"], dates2
carried = s2[pd.to_datetime(s2["date"]) == pd.Timestamp("2026-07-23")].iloc[0]
assert abs(float(carried["w"]) - 250.0) < 1e-9, "the day was not carried"
assert np.isnan(float(carried["reps"])), \
    "the interval count must stay blank — it was never observed"
assert len(b2) == 3, f"{len(b2)} bars: one per rep would be an invention"
chk2 = b2.groupby("date")["avg_w"].mean().round(6)
ref2 = s2.set_index("date")["w"].round(6)
assert (chk2 - ref2.reindex(chk2.index)).abs().max() < 1e-6
assert not s2["date"].duplicated().any()
print("   PASS  a set with no rows: carried at its own watts, zero fake bars")

# ── a series with no member rows at all: bars still drawn, no crash ───────
peak_only = _sets([
    {"activity_id": "B", "date": "2026-09-15", "_key": "peak:B:1200s",
     "reps": 1, "rep_secs": 1200.0, "set_w": 231.0, "intensity": np.nan,
     "Source": PEAK_SRC, "name": "FTP test", "temp": 17.4, "tsb": -3.0},
    {"activity_id": "D", "date": "2026-09-22", "_key": "peak:D:1200s",
     "reps": 1, "rep_secs": 1200.0, "set_w": 241.0, "intensity": np.nan,
     "Source": PEAK_SRC, "name": "FTP test", "temp": 18.0, "tsb": -1.0},
])
pv3 = run_protocol_view(iv, peak_only, "FTP", 20.0)
assert pv3["ok"]
assert pv3["n_sessions"] == 2 and pv3["n_reps"] == 2, pv3
assert len(pv3["bars"]) == 2 and "src" in pv3["bars"].columns
got3 = sorted(pd.to_datetime(pv3["sessions"]["date"]).dt.strftime("%Y-%m-%d"))
assert got3 == ["2026-09-15", "2026-09-22"], got3
print("   PASS  a meter-only series draws its bars instead of nothing")

print()
print("=" * 74)
print("2. on this athlete's own cache")
print("=" * 74)

CACHE = Path("data/interval_cache.csv")
CSV = Path("data/combined_training_data.csv")
if not CACHE.exists() or not CSV.exists():
    print("   SKIP  the interval cache is not in this checkout (gitignored)")
else:
    from core.data import load_data
    from core.interval_data import read_intervals
    from ml.set_evolution import (add_signatures, build_sets,
                                  fill_peak_efforts)
    from ml.type_comparison import dur_class_label, prep_types

    iv = read_intervals("2025-01-01T00:00:00")
    df = load_data()
    sets, n_peak = fill_peak_efforts(build_sets(iv, None, df), df)
    s, _ = add_signatures(sets)
    s = prep_types(s)
    s = s.assign(lab=[dur_class_label(b) for b in s["dur_b"]])

    bad = dropped_days = dup = inv = unknown_src = 0
    for (fam, lab), g in s.groupby(["family", "lab"]):
        for db in g["dur_b"].unique():
            sub = g[np.isclose(g["dur_b"].astype(float), float(db))]
            pv = run_protocol_view(iv, s, fam, float(db))
            if not pv["ok"]:
                continue
            want = pd.to_datetime(sub["date"]).dt.normalize().nunique()
            if pv["n_sessions"] != want:
                bad += 1
                print(f"   {fam} {lab} dur_b={db}: {pv['n_sessions']} sessions "
                      f"vs {want} set dates")
            ss = pv["sessions"]
            if ss["date"].duplicated().any():
                dup += 1
                print(f"   {fam} {lab} dur_b={db}: a date appears twice")
            b = pv["bars"]
            if len(b):
                chk = b.groupby("date")["avg_w"].mean().round(6)
                ref = ss.set_index("date")["w"].round(6)
                if (chk - ref.reindex(chk.index)).abs().max() > 1e-6:
                    inv += 1
                    print(f"   {fam} {lab} dur_b={db}: session mean != bar "
                          f"mean for a day")
                if "src" not in b.columns or \
                        set(b["src"]) - {DETECTED_SRC, PEAK_SRC}:
                    unknown_src += 1
                    print(f"   {fam} {lab} dur_b={db}: a bar does not state "
                          f"which instrument reported it")
            lost = set(pd.to_datetime(sub["date"]).dt.normalize()) - \
                set(pd.to_datetime(ss["date"]).dt.normalize())
            if lost:
                dropped_days += len(lost)
                print(f"   {fam} {lab} dur_b={db}: "
                      f"{len(lost)} day(s) with no session row")

    n_series = sum(len(g["dur_b"].unique()) for _, g in s.groupby(["family",
                                                                  "lab"]))
    print(f"   {n_series} series checked · {n_peak} peak readings in the "
          f"sets table")
    assert bad == 0, f"{bad} series lost a session"
    assert dropped_days == 0, f"{dropped_days} days dropped from the line"
    assert dup == 0, "a date got two session rows"
    assert inv == 0, "a session mean no longer equals its bars' mean"

    # the reported case: the 20-minute FTP sessions Evolution shows
    g = s[(s["family"] == "FTP") & (s["lab"] == "20 min")]
    if len(g):
        db = float(g["dur_b"].iloc[0])
        pv = run_protocol_view(iv, s, "FTP", db)
        assert pv["ok"] and pv["n_sessions"] == \
            pd.to_datetime(g[np.isclose(g["dur_b"].astype(float), db)]
                           ["date"]).dt.normalize().nunique()
        assert len(pv["sessions"]) > 1, \
            "the 20-minute FTP sessions are still collapsed to one"
        print(f"   PASS  FTP · 20 min: {pv['n_sessions']} sessions, "
              f"{pv['n_reps']} intervals, all dates on the axis")

    # every bar states where it came from, and the meter's rows are labelled
    assert unknown_src == 0, f"{unknown_src} series with unlabelled bars"
    print("   PASS  every series: all sessions, one row per date, means exact")

print()
print("ALL CHECKS PASS")
