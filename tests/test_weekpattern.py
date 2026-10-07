"""The weekly-pattern engine, on synthetic sessions.

Association only: weeks are aggregated, matched on season + starting
CTL/TSB bands, and ranked by the median of what followed — with n attached
and small samples refusing to rank. No files, no credentials.

Run:  python tests/test_weekpattern.py     (exit 0 = pass, 1 = fail)
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

from ml import weekpattern as WP

FAILED = []


def check(name, cond, detail=""):
    print(f"   {'PASS' if cond else 'FAIL'}  {name}"
          + (f" — {detail}" if (not cond and detail) else ""))
    if not cond:
        FAILED.append(name)


def _df(rows):
    return pd.DataFrame(rows)


# Three identical September A weeks (2xFTP, TSS ~500, CTL climbing), one
# rest week and one big week with no future after it (its outcome stays NaN
# instead of a partial one). No anchor sessions: outcomes are the natural
# CTL drift over the 28 days that follow each week.
def _sessions():
    rows = []
    def add(day, tt, tss, ctl, tsb):
        rows.append({"date": pd.Timestamp(day), "training_type": tt,
                     "tss": tss, "ctl": ctl, "atl": ctl - tsb, "tsb": tsb,
                     "eftp": np.nan})
    # A weeks (Mon 02, 09, 16 Sep 2024), CTL 100 -> 110
    for i, (mon, c0) in enumerate([("2024-09-02", 100.0),
                                   ("2024-09-09", 104.0),
                                   ("2024-09-16", 108.0)]):
        m = pd.Timestamp(mon)
        add(m, "FTP", 120.0, c0, -5.0)
        add(m + pd.Timedelta(days=2), "FTP", 120.0, c0 + 1, -8.0)
        add(m + pd.Timedelta(days=4), "AEROBIC BASE", 260.0, c0 + 2, -12.0)
    # B rest week (Mon 23 Sep): one easy ride, TSB positive
    m = pd.Timestamp("2024-09-23")
    add(m, "AEROBIC BASE", 150.0, 112.0, 5.0)
    # C big week (Mon 30 Sep): FTP + VO2MAX + long ride, TSB deep
    m = pd.Timestamp("2024-09-30")
    add(m, "FTP", 200.0, 113.0, -20.0)
    add(m + pd.Timedelta(days=2), "VO2MAX", 150.0, 114.0, -15.0)
    add(m + pd.Timedelta(days=4), "AEROBIC BASE", 350.0, 115.0, -25.0)
    return _df(rows)


df = _sessions()
W = WP.weekly_frame(df)
print("=" * 74)
print("1. weeks aggregate shape, conditions and outcomes")
print("=" * 74)
check("five training weeks aggregate", len(W) == 5, len(W))
a = W[W["week"] == pd.Timestamp("2024-09-02")].iloc[0]
check("quality counts only key families",
      int(a["quality"]) == 2 and int(a["ftp_n"]) == 2, a.to_dict())
check("rest days are days with no session",
      int(a["rest_days"]) == 4, int(a["rest_days"]))
check("starting CTL/TSB ride along",
      float(a["ctl_start"]) == 100.0 and float(a["tsb_start"]) == -5.0)
check("the 28-day CTL outcome is the natural drift that followed",
      sorted(W["d_ctl_28"].dropna().tolist()) == [2.0, 3.0, 7.0, 11.0],
      W["d_ctl_28"].tolist())
check("a week with no future keeps NaN, never a partial outcome",
      int(W["d_ctl_28"].isna().sum()) == 1, W["d_ctl_28"].tolist())

print()
print("=" * 74)
print("2. similarity matches season + bands, never the future")
print("=" * 74)
S = WP.similar_weeks(W, 9, 105.0, -8.0, exclude_weeks=0)
check("the three A weeks match each other",
      len(S) == 3, len(S))
S2 = WP.similar_weeks(W, 9, 104.0, -8.0, exclude_weeks=99)
check("excluding everything leaves nothing to rank",
      not len(S2), len(S2))
S3 = WP.similar_weeks(W, 3, 60.0, 20.0, exclude_weeks=0)
check("a March week at CTL 60 matches nothing in September",
      not len(S3), len(S3))

print()
print("=" * 74)
print("3. ranking is medians with n — small samples do not rank")
print("=" * 74)
R = WP.rank_patterns(S)
check("the repeated A pattern ranks with n=3, median +7",
      len(R) == 1 and bool(R["ranked"].iloc[0]) and int(R["n"].iloc[0]) == 3
      and float(R["d_ctl_med"].iloc[0]) == 7.0,
      R.to_dict("records") if len(R) else "no rows")
R1 = WP.rank_patterns(WP.similar_weeks(W, 9, 112.0, 5.0, exclude_weeks=0))
check("a lone rest week is listed, never ranked",
      len(R1) == 1 and not bool(R1["ranked"].iloc[0])
      and float(R1["d_ctl_med"].iloc[0]) == 2.0,
      R1.to_dict("records") if len(R1) else "no rows")

print()
print("=" * 74)
print("4. now-snapshot and form bands")
print("=" * 74)
st = WP.current_state(df)
check("form reads off TSB bands",
      WP.current_state(
          df.assign(tsb=-20.0))["form"] == "fatigued"
      and WP.current_state(df.assign(tsb=12.0))["form"] == "fresh"
      and WP.current_state(df.assign(tsb=0.0))["form"] == "productive")
check("quality share counts key families only, over the last 28 days",
      abs(st["quality_share"] - 6 / 10) < 1e-9, st["quality_share"])

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    raise SystemExit(1)
print("ALL CHECKS PASS")
