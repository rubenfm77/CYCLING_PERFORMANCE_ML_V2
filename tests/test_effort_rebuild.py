"""Rebuilding the efforts the detector cut into pieces — the rules, synthetic.

The reported defect: "in intervals you are not retrieving both intervals, i do
usually two but you get just one … the same problem in evolution, there's just
one interval on 30 Sep". One meter window per ride cannot show two intervals,
and the segmenter had cut both of them into fragments (158 + 285 + 384 + 180 s
with 36 / 87 / 27 s between them, and 882 + 324 s).

This file pins `ml.interval_watts.rebuilt_efforts` and the two callers of it
with no files and no credentials:

  1. the join itself — what is bridged and what is not;
  2. what makes a rebuild fire at all (the meter's window vouches for it);
  3. what stops it — a detector effort already whole, one already within two
     minutes, a window the runs do not reproduce;
  4. what the pages then draw: one bar per effort, both on one day, one class
     per length, and `iv=None` changing nothing;
  5. never two sources describing one effort.

Run:  python tests/test_effort_rebuild.py     (exit 0 = pass, 1 = fail)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ml import interval_watts as iw
from ml.interval_watts import REBUILD_DIP_S, REBUILD_NEAR_S, REBUILD_SRC, \
    REBUILD_W_TOL

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"   PASS  {name}")
    else:
        FAILED.append(name)
        print(f"   FAIL  {name} {detail}")


def _iv(rows, act="i1", day="2026-09-30"):
    """Detector-shaped rows: (iv_type, secs, avg_w), seq by position."""
    return pd.DataFrame([
        {"activity_id": act, "date": day, "seq": i, "iv_type": t,
         "label": None, "secs": s, "avg_w": w, "group_id": np.nan}
        for i, (t, s, w) in enumerate(rows)])


def _df(rows):
    """Session-shaped rows for the peak-window columns."""
    return pd.DataFrame([{
        "id": r.get("id", "i1"),
        "date": r.get("date", "2026-09-30"),
        "training_type": r.get("tt", "FTP"),
        "icu_pm_ftp_watts": r.get("pw", np.nan),
        "icu_pm_ftp_secs": r.get("ps", np.nan),
        "power_avg": r.get("avg", 170.0),
        "duration_s": r.get("dur", 5400.0),
    } for r in rows])


# 30 Sep 2026, exactly as the cache holds it: two intervals, cut to pieces.
REPORTED = _iv([
    ("RECOVERY", 1100, 173),
    ("WORK", 158, 241), ("RECOVERY", 36, 276),
    ("WORK", 285, 247), ("RECOVERY", 87, 221),
    ("WORK", 384, 247), ("RECOVERY", 27, 181),
    ("WORK", 180, 253), ("RECOVERY", 750, 85),
    ("WORK", 882, 243), ("RECOVERY", 1, 242),
    ("WORK", 324, 252), ("RECOVERY", 345, 61),
    ("WORK", 435, 189), ("RECOVERY", 261, 191),
])
REPORTED_DF = _df([{"id": "i1", "pw": 244.0, "ps": 1320.0}])

print("=" * 74)
print("1. the join: what counts as one effort")
print("=" * 74)

runs = iw._iv_runs(REPORTED)
check("the short breaks are inside the effort they interrupt",
      [int(r["secs"]) for r in runs] == [1157, 1207, 435],
      [int(r["secs"]) for r in runs])
r1, r2 = runs[0], runs[1]
check("the span carries every piece and every bridged break",
      int(r1["secs"]) == 158 + 36 + 285 + 87 + 384 + 27 + 180,
      int(r1["secs"]))
check("the watts are time-weighted over the whole span, not a mean of means",
      abs(r1["w"] - ((158 * 241 + 36 * 276 + 285 * 247 + 87 * 221
                      + 384 * 247 + 27 * 181 + 180 * 253) / 1157)) < 0.5,
      r1["w"])
check("consecutive short breaks accumulate before the decision",
      r1["n_dips"] == 3 and r2["n_dips"] == 1, (r1["n_dips"], r2["n_dips"]))

long_break = _iv([("WORK", 400, 250), ("RECOVERY", 60, 100),
                  ("RECOVERY", 60, 100), ("WORK", 400, 250)])
lr = iw._iv_runs(long_break)
check("60 s + 60 s of rest is a rest, not a 45-second pause",
      len(lr) == 2 and all(int(r["secs"]) == 400 for r in lr),
      [int(r["secs"]) for r in lr])

tail = _iv([("WORK", 700, 250), ("RECOVERY", 500, 90)])
tr = iw._iv_runs(tail)
check("a ride's tail never shortens its last effort",
      len(tr) == 1 and int(tr[0]["secs"]) == 700, [int(r["secs"]) for r in tr])

print()
print("=" * 74)
print("2. the vouch: nothing is rebuilt without the meter's window")
print("=" * 74)

reb = iw.rebuilt_efforts(REPORTED, REPORTED_DF)
check("both intervals of 30 Sep come back, whole",
      len(reb) == 2, len(reb))
check("19:17 and 20:07 — the two runs, to the second",
      sorted(int(s) for s in reb["secs"]) == [1157, 1207],
      sorted(int(s) for s in reb["secs"]))
check("each run is filed in the class of ITS OWN length, never the meter's",
      dict(zip(reb["secs"], reb["cls"]))[1157] == "10-20 min"
      and dict(zip(reb["secs"], reb["cls"]))[1207] == "20-30 min",
      dict(zip(reb["secs"], reb["cls"])))
check("every rebuilt row states the window that vouches for it",
      bool((reb["peak_s"] == 1320).all()) and bool((reb["peak_w"] == 244).all()))
check("the pieces are printed, so a reader can re-add them by hand",
      reb.sort_values("secs")["pieces"].iloc[0].count("+") == 3,
      sorted(reb["pieces"]))
check("no peak window at all -> nothing rebuilt",
      not len(iw.rebuilt_efforts(REPORTED, _df([{"id": "i1"}]))))
check("a window under the 10-minute floor -> nothing rebuilt",
      not len(iw.rebuilt_efforts(REPORTED,
                                 _df([{"id": "i1", "pw": 244.0, "ps": 450.0}]))))
check("no sessions -> nothing rebuilt",
      not len(iw.rebuilt_efforts(REPORTED, pd.DataFrame())))
check("no cache -> nothing rebuilt",
      not len(iw.rebuilt_efforts(None, REPORTED_DF)))

print()
print("=" * 74)
print("3. what stops the rebuild")
print("=" * 74)

whole = _iv([("WORK", 1300, 244), ("RECOVERY", 300, 80)])
check("the detector already reported an effort of the window's class",
      not len(iw.rebuilt_efforts(whole, REPORTED_DF)),
      "a whole 21:40 row already tells this story")

near = _iv([("WORK", 158, 241), ("RECOVERY", 36, 276),
            ("WORK", 285, 247), ("RECOVERY", 87, 221),
            ("WORK", 384, 247), ("RECOVERY", 27, 181),
            ("WORK", 180, 253), ("RECOVERY", 750, 85),
            ("WORK", 1150, 245)])
check("a detected effort within two minutes is that effort, already counted",
      not any(int(s) == 1157 for s in
              iw.rebuilt_efforts(near, REPORTED_DF)["secs"]),
      iw.rebuilt_efforts(near, REPORTED_DF)["secs"].tolist())

weak = _iv([("WORK", 300, 244), ("RECOVERY", 27, 244),
            ("WORK", 857, 190)])
check("a run the meter's watts do not describe is left alone",
      not len(iw.rebuilt_efforts(weak, REPORTED_DF)),
      "190 W over 18:40 is not the 244 W window")

far = _iv([("WORK", 900, 244), ("RECOVERY", 27, 244),
           ("WORK", 900, 244)])
check("a run too far from the meter's window length is left alone",
      not len(iw.rebuilt_efforts(far, REPORTED_DF)),
      iw.rebuilt_efforts(far, REPORTED_DF)["secs"].tolist())
check("the constants keep the guards tighter than what they guard",
      0 < REBUILD_DIP_S <= 90 and REBUILD_NEAR_S >= 60
      and REBUILD_W_TOL <= 0.15,
      (REBUILD_DIP_S, REBUILD_NEAR_S, REBUILD_W_TOL))

print()
print("=" * 74)
print("4. what the pages draw: one bar per effort")
print("=" * 74)

B = iw.effort_best(REPORTED_DF, REPORTED)
check("one row per effort ridden, the meter's single window replaced",
      len(B) == 2, len(B))
check("both rows say where they came from",
      set(B["source"]) == {REBUILD_SRC}, set(B["source"]))
check("every row's class is its own length's class",
      all(rep == cls for rep, cls in zip(B["secs"].map(iw.rep_class),
                                         B["cls"])))

S20 = iw.day_series(REPORTED_DF, "FTP", "20-30 min", iv=REPORTED)
S10 = iw.day_series(REPORTED_DF, "FTP", "10-20 min", iv=REPORTED)
check("one day, both of its efforts, each in the class its length names",
      len(S20) == 1 and len(S10) == 1,
      (len(S20), len(S10)))
both = pd.concat([S20, S10]).sort_values("secs")
check("both efforts are on 30 Sep and neither was dropped",
      list(both["secs"]) == [1157, 1207]
      and all(str(d) == "2026-09-30" for d in pd.to_datetime(both["day"])
              .dt.strftime("%Y-%m-%d")),
      both["secs"].tolist())
check("the day's own average is the plain mean of its bars, class by class",
      abs(float(S20["day_avg"].iloc[0]) - float(S20["w"].iloc[0])) < 1e-9)
check("one ride that drew two bars is still one ride",
      int(both["n_sessions"].iloc[0]) == 1)

same_day = _iv([
    ("RECOVERY", 1100, 173),
    ("WORK", 600, 244), ("RECOVERY", 60, 244),
    ("WORK", 601, 246), ("RECOVERY", 900, 80),
    ("WORK", 600, 244), ("RECOVERY", 60, 244),
    ("WORK", 601, 246), ("RECOVERY", 900, 80),
], day="2026-09-30")
same_df = _df([{"id": "i1", "date": "2026-09-30", "pw": 245.0, "ps": 1250.0}])
Sboth = iw.day_series(same_df, "FTP", "20-30 min", iv=same_day)
check("two efforts of one class on one day draw two bars",
      len(Sboth) == 2, len(Sboth))
check("and the two bars sit on one day with one session behind them",
      int(Sboth["day"].nunique()) == 1 and int(Sboth["n_sessions"].iloc[0]) == 1)
check("the day average is the mean of those two bars",
      abs(float(Sboth["day_avg"].iloc[0])
          - float(Sboth["w"].mean())) < 1e-9)
O = iw.day_options(same_df, iv=same_day)
row = O[(O["tt"] == "FTP") & (O["cls"] == "20-30 min")].iloc[0]
check("the picker counts 1 day, 2 bars, 1 session — and bars == what is drawn",
      int(row["days"]) == 1 and int(row["bars"]) == 2
      and int(row["sessions"]) == 1,
      dict(days=row["days"], bars=row["bars"], sessions=row["sessions"]))
check("bars is never below days",
      bool((O["bars"] >= O["days"]).all()))

print()
print("=" * 74)
print("5. without the cache, nothing changes")
print("=" * 74)

plain = iw.effort_best(REPORTED_DF)
check("iv=None still reports the meter's own window, one row",
      len(plain) == 1 and float(plain["secs"].iloc[0]) == 1320,
      plain["secs"].tolist())
check("and that row says it is the meter's",
      plain["source"].iloc[0] == iw.PEAK_ROW_SRC, plain["source"].tolist())
S_plain = iw.day_series(REPORTED_DF, "FTP", "20-30 min")
check("the day chart without a cache draws one bar, as it did before",
      len(S_plain) == 1, len(S_plain))

unlabelled = _df([{"id": "i1", "tt": "", "pw": 244.0, "ps": 1320.0}])
check("an unlabelled session is never charted, rebuilt or not",
      len(iw.effort_best(unlabelled, REPORTED)) == 0)

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    raise SystemExit(1)
print("ALL CHECKS PASS")
