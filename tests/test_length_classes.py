"""The Intervals detail and the Evolution day chart must pick the SAME days.

The complaint behind this file: on the Intervals page "FTP · 20 min" offered
four sessions (17 Apr, 23 Jul, 15 Sep, 22 Sep) while Evolution's
"20-30 min" chart showed every one of the athlete's 20-minute efforts —
including 30 Sep 2026, where two 20-minute intervals were ridden. Nothing
had been dropped by accident: the detail grouped by WHOLE MINUTE, so the
same effort read as 21:00, 22:00 or 24:00 (the peak meter's window for a
20-minute effort) was filed under its own one-session option and read as
lost, while Evolution's rep-length band "20-30 min" held all of them.

Rules pinned here:

  1. the detail's duration options are Evolution's rep-length classes
     (ml.interval_watts.REP_ORDER) — one criterion, both pages;
  2. `dur_cls` on the sets table is exactly rep_class(rep_secs): the class
     is never computed a second way in the second place;
  3. every day Evolution charts for a (type × class) appears in the detail
     for that same (type × class);
  4. whole-minute selection still works for the comparison table, and the
     one selector is never a blend of the two;
  5. a class outside REP_ORDER still gets an option — no silent drop.

Two halves: synthetic (no files, no credentials) and this athlete's own
cache, skipped cleanly when data/interval_cache.csv is absent (gitignored).

Run:  python tests/test_length_classes.py     (exit 0 = pass, 1 = fail)
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

import numpy as np
import pandas as pd
from pathlib import Path

from ml.interval_watts import REP_ORDER, rep_class
from ml.protocol_reps import run_protocol_view
from ml.set_evolution import DETECTED_SRC, PEAK_SRC, SET_COLUMNS
from ml.type_comparison import (duration_options, dur_bucket,
                                dur_class_label, series_sel, series_stats)

print("=" * 74)
print("1. the rule, on synthetic sets and rows")
print("=" * 74)


def _iv(rows):
    """A detector-shaped interval frame: WORK rows only."""
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
    """A sets-shaped frame, grouped the way the view hands it over."""
    f = pd.DataFrame([{**{c: np.nan for c in SET_COLUMNS}, **r}
                      for r in rows])
    f["date"] = pd.to_datetime(f["date"])
    f["db_type"] = f.get("db_type", pd.Series("", index=f.index)) \
        .fillna("").astype(str)
    f["name"] = f["name"].fillna("").astype(str)
    f["family"] = "FTP"
    f["dur_b"] = [dur_bucket(x) for x in f["rep_secs"]]   # whole minutes
    f["dur_cls"] = [rep_class(x) for x in f["rep_secs"]]  # Evolution's bands
    return f


# Four 20-minute efforts of the same training type, each read at a slightly
# different length — 20:06 detected, 21:00 / 22:00 / 26:00 from the meter.
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
])
sets = _sets([
    {"activity_id": "A", "date": "2026-04-17", "_key": "g1", "reps": 3,
     "rep_secs": 1206.0, "set_w": 212.666667, "intensity": 90.666667,
     "Source": DETECTED_SRC, "name": "sweet spot", "temp": 19.4,
     "tsb": 5.0},
    {"activity_id": "B", "date": "2026-07-04", "_key": "peak:B",
     "reps": 1, "rep_secs": 1260.0, "set_w": 224.0, "intensity": np.nan,
     "Source": PEAK_SRC, "name": "FTP test", "temp": 21.0, "tsb": 2.0},
    {"activity_id": "C", "date": "2026-09-30", "_key": "peak:C",
     "reps": 1, "rep_secs": 1320.0, "set_w": 244.0, "intensity": np.nan,
     "Source": PEAK_SRC, "name": "FTP test", "temp": 18.0, "tsb": -1.0},
    {"activity_id": "D", "date": "2026-07-09", "_key": "peak:D",
     "reps": 1, "rep_secs": 1560.0, "set_w": 236.0, "intensity": np.nan,
     "Source": PEAK_SRC, "name": "FTP test", "temp": 22.5, "tsb": -4.0},
    {"activity_id": "E", "date": "2026-03-03", "_key": "k:E",
     "reps": 4, "rep_secs": 240.0, "set_w": 281.0, "intensity": 112.0,
     "Source": DETECTED_SRC, "name": "4 x 4", "temp": 12.0, "tsb": 8.0},
    {"activity_id": "F", "date": "2026-03-05", "_key": "k:F",
     "reps": 1, "rep_secs": np.nan, "set_w": 200.0, "intensity": np.nan,
     "Source": DETECTED_SRC, "name": "unparsed", "temp": 11.0, "tsb": 0.0},
])

# 1 — the old whole-minute grouping: four one-session options for one job
mins = [dur_class_label(d) for d in
        sets[sets["dur_cls"] == "20-30 min"]["dur_b"]]
assert mins == ["20 min", "21 min", "22 min", "26 min"], mins
print(f"   whole minutes would have split this work into {mins}")

# 2 — the picker's options are Evolution's classes, in order, with counts
opts = duration_options(sets)
assert list(opts.values()) == ["90s-5min", "20-30 min", "unknown"], opts
lab = [k for k, v in opts.items() if v == "20-30 min"][0]
assert lab == "20-30 min · 4 sessions · 4 sets", lab
assert lab.split(" · ")[0] in REP_ORDER
assert "unknown" in opts.values(), \
    "a set with an unparseable length must keep its own option"
print(f"   PASS  options: {list(opts)}")

# 3 — one class, every session in it, whatever its whole-minute duration
pv = run_protocol_view(iv, sets, "FTP", None, "20-30 min")
assert pv["ok"], pv.get("reason")
got = sorted(pd.to_datetime(pv["sessions"]["date"]).dt.strftime("%Y-%m-%d"))
assert got == ["2026-04-17", "2026-07-04", "2026-07-09", "2026-09-30"], got
assert not pv["sessions"]["date"].duplicated().any(), "a date got two rows"
bars = pv["bars"]
if len(bars):
    chk = bars.groupby("date")["avg_w"].mean().round(6)
    ref = pv["sessions"].set_index("date")["w"].round(6)
    assert (chk - ref.reindex(chk.index)).abs().max() < 1e-6, \
        "session mean != mean of that day's bars"
print(f"   PASS  class '20-30 min' keeps all {pv['n_sessions']} sessions "
      f"({pv['n_reps']} intervals)")

# 4 — the whole-minute selector still works, and is never mixed with it
pv20 = run_protocol_view(iv, sets, "FTP", 20.0)
assert pv20["ok"] and pv20["n_sessions"] == 1, pv20
st20 = series_stats(sets, "FTP", 20.0)
stcls = series_stats(sets, "FTP", dur_cls="20-30 min")
assert st20["n"] == 1 and stcls["n"] == 4, (st20["n"], stcls["n"])
sel = series_sel(sets, "FTP", dur_cls="20-30 min")
assert len(sel) == 4 and not np.isclose(sel["dur_b"], 20.0).all()
try:
    series_sel(sets, "FTP")
    raise SystemExit("FAIL  series_sel accepted no selector at all")
except ValueError:
    pass
print("   PASS  whole-minute and class selectors stay separate")

# the 4-minute set is in its own class, never averaged with the long work
assert set(series_sel(sets, "FTP", dur_cls="90s-5min")["activity_id"]) == {"E"}

print()
print("=" * 74)
print("2. on this athlete's own cache")
print("=" * 74)

CACHE = Path("data/interval_cache.csv")
CSV = Path("data/combined_training_data.csv")
if not CACHE.exists() or not CSV.exists():
    print("   SKIP  the interval cache is not in this checkout (gitignored)")
else:
    import ml.interval_watts as iw
    from core.data import load_data
    from core.interval_data import read_intervals
    from ml.set_evolution import add_signatures, build_sets, fill_peak_efforts
    from ml.type_comparison import prep_types

    iv = read_intervals("2025-01-01T00:00:00")
    df = load_data()
    sets, _ = fill_peak_efforts(build_sets(iv, None, df), df)
    s, _ = add_signatures(sets)
    s = prep_types(s)
    assert "dur_cls" in s.columns, "prep_types did not file a length class"
    same = s["dur_cls"] == s["rep_secs"].map(rep_class)
    assert bool(same.all()), f"{int((~same).sum())} sets classed two ways"

    # the reported case: every day Evolution charts for FTP / 20-30 min must
    # be a session of the detail's FTP / 20-30 min
    cls = "20-30 min"
    evo = iw.day_series(df, "FTP", cls)
    pv = run_protocol_view(iv, s, "FTP", None, cls)
    assert pv["ok"], pv.get("reason")
    detail = set(pd.to_datetime(pv["sessions"]["date"]).dt.normalize())
    charted = set(pd.to_datetime(evo["day"]).dt.normalize())
    lost = sorted(str(d.date()) for d in charted - detail)
    assert not lost, f"Evolution charts {lost}, the detail does not"
    n_min = len(set(pd.to_datetime(
        run_protocol_view(iv, s, "FTP", 20.0)["sessions"]["date"]
    ).dt.normalize()))
    band = s[(s["family"] == "FTP") & (s["dur_cls"] == cls)]
    print(f"   {cls}: {pv['n_sessions']} sessions / {pv['n_reps']} intervals "
          f"in the detail · {len(charted)} days in Evolution · the old "
          f"whole-minute '20 min' option had {n_min}")
    print(f"   the band spans {sorted(set(band['dur_label']))} "
          f"({len(band)} sets)")
    assert pv["n_sessions"] >= n_min, "the class lost a session"
    assert len(charted) <= pv["n_sessions"], \
        "Evolution charts more days than the detail holds"

    # every option of the picker is a real class, and no set is unoffered
    opts = duration_options(s[s["family"] == "FTP"])
    offered = set(opts.values())
    assert offered == set(s.loc[s["family"] == "FTP", "dur_cls"]), \
        "a class of sets is not offered by the picker"
    assert list(opts) == sorted(
        list(opts), key=lambda k: REP_ORDER.index(k.split(" · ")[0])
        if k.split(" · ")[0] in REP_ORDER else len(REP_ORDER)
    ), "the options are not in class order"
    print(f"   PASS  FTP offers {len(opts)} classes, every one of them "
          f"reached by at least one set")

print()
print("ALL CHECKS PASS")
