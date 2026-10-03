"""The two-source rule on the Intervals page: peak meter fills what the
detector missed, and never describes the same effort twice.

The complaint behind this file: "no trace of 20 min intervals that are
perfectly detected in the evolution section". The detector segments a ride by
power and cadence, so it cuts sustained work into pieces — for 15 Sep 2026 its
longest row on that FTP session was 485 s while Intervals.icu's peak meter
holds 231 W for 1200 s. `fill_peak_efforts` brings the meter's reading in,
under rules that stop it ever counting an effort the detector already
reported.

Two halves:

  1. synthetic — the rules themselves, no files, no credentials;
  2. the real cache — the fill on this athlete's own data, skipped cleanly
     when `data/interval_cache.csv` is not present (it is gitignored).

Run:  python tests/test_peak_fill.py     (exit 0 = pass, 1 = fail)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from ml.set_evolution import (DETECTED_SRC, PEAK_SRC, SET_COLUMNS,
                              _dur_label, build_sets, fill_peak_efforts,
                              quality_gates)

print("=" * 74)
print("1. the rules, on synthetic sets")
print("=" * 74)


def _sets(rows):
    """A detector-shaped frame: one row per (activity, length)."""
    f = pd.DataFrame([{**{c: np.nan for c in SET_COLUMNS}, **r}
                      for r in rows])
    if "date" in f.columns:
        f["date"] = pd.to_datetime(f["date"])
    if "act_start" in f.columns:
        # build_sets hands this a real timestamp, so the dtype has to match —
        # otherwise concat promotes it and the comparison below is meaningless
        f["act_start"] = pd.to_datetime(f["act_start"])
    if "db_type" in f.columns:
        # build_sets leaves this a string column (blank when unlabelled)
        f["db_type"] = f["db_type"].fillna("").astype(str)
    if "name" in f.columns:
        f["name"] = f["name"].fillna("").astype(str)
    return f


def _peaks(rows):
    return pd.DataFrame(rows)


# activity a: the detector found the SAME effort (1190 s vs a 1200 s reading)
base = _sets([
    {"activity_id": "a", "date": "2026-09-15", "reps": 2,
     "rep_secs": 1190.0, "set_w": 230.0},
    # activity b: detector found a 12-minute block, the meter says 20 min
    {"activity_id": "b", "date": "2026-09-22", "reps": 1,
     "rep_secs": 742.0, "set_w": 244.0},
    # activity c: a short class only — the detector never cuts these
    {"activity_id": "c", "date": "2026-09-19", "reps": 3,
     "rep_secs": 300.0, "set_w": 300.0},
])
peaks = _peaks([
    {"id": "a", "date": "2026-09-15", "training_type": "FTP",
     "icu_pm_ftp_watts": 231.0, "icu_pm_ftp_secs": 1200.0},
    {"id": "b", "date": "2026-09-22", "training_type": "FTP",
     "icu_pm_ftp_watts": 241.0, "icu_pm_ftp_secs": 1200.0},
    {"id": "c", "date": "2026-09-19", "training_type": "FTP",
     "icu_pm_ftp_watts": 223.0, "icu_pm_ftp_secs": 1200.0},
    # d: nothing detected at all, and the reading is under the 10-min floor
    {"id": "d", "date": "2026-09-20", "training_type": "FTP",
     "icu_pm_ftp_watts": 350.0, "icu_pm_ftp_secs": 480.0},
])

out, n = fill_peak_efforts(base.copy(), peaks)
added = out[out["Source"] == PEAK_SRC]
print(f"   added {n} of {len(peaks)} readings; rows {len(base)} -> {len(out)}")

assert n == 2, f"expected b and c to be filled, a already found: got {n}"
assert set(added["activity_id"]) == {"b", "c"}, sorted(added["activity_id"])
assert len(out) == len(base) + n, "the frame did not grow by exactly n"
print("   PASS  same effort (1190 s) refused; the two gaps filled")

# every added row: long, single-rep, tagged, carrying the athlete's own label
assert (pd.to_numeric(added["rep_secs"]) >= 600).all(), "a short effort added"
assert (added["reps"] == 1).all(), "added rows are not single-rep readings"
assert set(added["Source"]) == {PEAK_SRC}
assert set(added["db_type"]) == {"FTP"}, sorted(set(added["db_type"]))
# the detector rows survive the fill word for word: same identity, same
# measurements, same label (the extra context columns only gain NaNs)
_keep = ["activity_id", "date", "reps", "rep_secs", "set_w", "rest",
         "db_type", "Source"]
assert (out[out["Source"] == DETECTED_SRC][_keep].reset_index(drop=True)
        .equals(base.assign(Source=DETECTED_SRC)[_keep]
                .reset_index(drop=True))), \
    "an existing detector row was modified"
print("   PASS  detector rows untouched, added rows tagged and labelled")

# the detector set of activity b (742 s) is still there, unchanged
b = out[(out["activity_id"] == "b") & (out["Source"] == DETECTED_SRC)]
assert len(b) == 1 and float(b["rep_secs"].iloc[0]) == 742.0
print("   PASS  the detected 12-min block was not replaced by the 20-min one")

# no added reading within near_s of a detected one, in seconds or in class
for r in added.itertuples():
    have = pd.to_numeric(
        base[base["activity_id"] == r.activity_id]["rep_secs"],
        errors="coerce").dropna()
    assert not any(abs(float(x) - r.rep_secs) <= 120 for x in have), r
    assert not any(_dur_label(float(x)) == _dur_label(r.rep_secs)
                   for x in have), r
print("   PASS  no added reading sits within 2 min of, or in the class of, "
      "a detected set")

# without the peak columns, nothing changes at all
same, k = fill_peak_efforts(base.copy(), pd.DataFrame({"date": []}))
assert k == 0 and len(same) == len(base)
assert set(same["Source"]) == {DETECTED_SRC}
print("   PASS  no peak columns -> no fill, and the Source tag still lands")

# a peak row must survive the same quality screen every other set faces
clean, report, dropped = quality_gates(out)
assert len(clean[clean["Source"] == PEAK_SRC]) == n, \
    "the gates excluded a peak reading"
print("   PASS  quality gates keep every added reading")

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

    iv = read_intervals("2025-01-01T00:00:00")
    df = load_data()
    base = build_sets(iv, None, df)
    filled, n = fill_peak_efforts(base.copy(), df)
    print(f"   sets {len(base)} -> {len(filled)}  (+{n} peak readings)")

    assert len(filled) == len(base) + n
    assert (filled["Source"] == PEAK_SRC).sum() == n
    assert (filled["Source"] == DETECTED_SRC).sum() == len(base)

    add = filled[filled["Source"] == PEAK_SRC]
    assert (pd.to_numeric(add["rep_secs"]) >= 600).all(), \
        "a reading under 10 min came in"
    assert (add["reps"] == 1).all()

    # nothing under the 10-minute floor moved — the fill only ever adds
    def classes(frame):
        r = pd.to_numeric(frame["rep_secs"], errors="coerce")
        short = frame[r < 600]
        return (short.assign(k=short["rep_secs"].map(_dur_label))
                .groupby("k").size().to_dict())
    assert classes(filled) == classes(base), \
        "the short-effort side changed — it must be untouched"

    # no double counting against the detector, activity by activity
    det = {a: [float(x) for x in pd.to_numeric(
        g["rep_secs"], errors="coerce").dropna()]
        for a, g in base.groupby(base["activity_id"].astype(str))}
    for r in add.itertuples():
        have = det.get(str(r.activity_id), [])
        assert not any(abs(x - r.rep_secs) <= 120 for x in have), \
            f"{r.activity_id}: {r.rep_secs} already within 2 min of a detection"
        assert not any(_dur_label(x) == _dur_label(r.rep_secs)
                       for x in have), \
            f"{r.activity_id}: {r.rep_secs} already in a detected class"
    print(f"   PASS  {n} readings added, none of them a re-count of a "
          f"detection, short side unchanged")

    # the complaint: 20-minute FTP sessions were invisible here
    from ml.set_evolution import add_signatures
    from ml.type_comparison import dur_class_label, prep_types

    def ftp_20(frame):
        s, _ = add_signatures(frame)
        s = prep_types(s)
        s = s.assign(lab=[dur_class_label(b) for b in s["dur_b"]])
        g = s[(s["family"] == "FTP") & (s["lab"] == "20 min")]
        return sorted(set(pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d")))

    before, after = ftp_20(base), ftp_20(filled)
    print(f"   20-min FTP sessions  before: {before}")
    print(f"                        after : {after}")
    assert len(after) > len(before), "the 20-minute sessions are still missing"
    for d in ("2026-09-15", "2026-09-22"):
        if d in set(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")):
            assert d in after, f"{d} (an FTP session with a 20:00 peak) " \
                               f"is still not in the 20-min class"

    # the gates keep them on the real data too
    clean, report, dropped = quality_gates(filled)
    kept = clean[clean["Source"] == PEAK_SRC]
    assert len(kept) > 0, "every peak reading was excluded by the gates"
    peak_out = len(dropped) - (len(base) - (len(clean) - n))
    print(f"   PASS  {len(kept)} of {n} survive the quality screen; "
          f"{peak_out} of them excluded as a duplicate or a bad grouping")

print()
print("ALL CHECKS PASS")
