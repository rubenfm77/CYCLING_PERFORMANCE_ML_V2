"""The plan page builds from the athlete's own history, nothing invented.

Synthetic frames only (no files, no credentials): the year score ranks a
structured year above an empty one, the mix counts what was ridden, the
icu text keeps blank lines around repeat blocks (the parser requires them),
targets are %FTP, and every planned interval carries its source.

Run:  python tests/test_plan.py     (exit 0 = pass, 1 = fail)
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

from ml import plan as PL

FAILED = []


def check(name, cond, detail=""):
    print(f"   {'PASS' if cond else 'FAIL'}  {name}"
          + (f" — {detail}" if (not cond and detail) else ""))
    if not cond:
        FAILED.append(name)


print("=" * 74)
print("1. years are scored in the open, structure beats emptiness")
print("=" * 74)
df = pd.DataFrame([
    {"date": "2021-03-01", "training_type": "FTP", "id": "a",
     "FTP": 270.0},
    {"date": "2021-04-01", "training_type": "VO2MAX", "id": "b",
     "FTP": 270.0},
    {"date": "2026-03-01", "training_type": "FTP", "id": "c",
     "FTP": np.nan},
])
P = pd.DataFrame([
    {"year": 2021, "tt": "FTP", "n": 3, "secs": 600.0},
    {"year": 2021, "tt": "VO2MAX", "n": 5, "secs": 300.0},
])
M = pd.DataFrame([{"year": 2026, "secs": 1200.0, "n": 1, "w": 240.0}])
yt = PL.year_table(df, P, M)
check("the structured year ranks first",
      int(yt["year"].iloc[0]) == 2021, yt["year"].tolist())
r21 = yt[yt["year"] == 2021].iloc[0]
check("every component is shown, nothing hidden",
      set(["sessions", "ftp_labelled", "prescribed_key", "measured_long",
           "ftp_setting", "score"]) <= set(yt.columns)
      and int(r21["ftp_labelled"]) == 1
      and int(r21["prescribed_key"]) == 2
      and float(r21["ftp_setting"]) == 270.0,
      r21.to_dict())

print()
print("=" * 74)
print("2. the mix counts what was prescribed, most-ridden first")
print("=" * 74)
mix = PL.prescribed_mix(P, 2021)
check("one row per family x length x reps",
      len(mix) == 2 and list(mix["family"]) == ["FTP", "VO2MAX"],
      mix["family"].tolist())
check("nominal durations use the house rounding",
      list(mix["nominal"]) == [10.0, 5.0], mix["nominal"].tolist())
check("an empty year gives an empty mix, never an invented one",
      not len(PL.prescribed_mix(P, 2026)))

print()
print("=" * 74)
print("3. icu text parses: blank lines, repeat header, %FTP")
print("=" * 74)
t = PL.icu_text("FTP 2x20min", 20.0, 2, "90-95%", 10.0)
check("repeat count rides on the header line",
      "Main Set 2x" in t.splitlines(), t)
check("blank line before AND after every repeat block",
      "\n\nMain Set 2x\n" in t and "\n\nCooldown\n" in t
      and "\n\nWarmup\n" in t, repr(t[:60]))
check("steps carry duration, %FTP target and cadence",
      "- 20m 90-95% 90rpm" in t.splitlines()
      and "- 10m 50% 90rpm" in t.splitlines())
check("a single effort has no repeat header",
      "Main Set\n- 20m 90-95% 90rpm" in
      PL.icu_text("FTP 20min", 20.0, 1, "90-95%", None))
check("short reps keep seconds, never 0 min",
      "- 30s 120-130% 90rpm" in
      PL.icu_text("BILLAT 10x30s", 0.5, 10, "120-130%", 0.5).splitlines())

print()
print("=" * 74)
print("4. the plan replays the mix over dated weeks")
print("=" * 74)
refs = {"ftp": 240.0, "best": {10.0: (254.0, "2026-05-13")}}
plan = PL.build_plan(mix, refs, ["Sat", "Wed"], pd.Timestamp("2026-10-12"),
                     weeks=12)
check("twelve weeks, two sessions a week from two templates",
      len(plan) == 12 and all(len(w["sessions"]) == 2 for w in plan),
      f"{len(plan)} weeks")
w1, w5, w12 = plan[0], plan[4], plan[11]
check("base rides the mix as counted",
      w1["phase"] == "Base"
      and [(s["reps"], s["nominal"]) for s in w1["sessions"]]
      == [(3, 10.0), (5, 5.0)],
      w1["phase"])
check("build adds one rep on the long sets only",
      w5["phase"] == "Build"
      and [(s["reps"], s["nominal"]) for s in w5["sessions"]]
      == [(4, 10.0), (5, 5.0)],
      w5["phase"])
check("the easy week halves the reps",
      w12["phase"] == "Easy"
      and [(s["reps"], s["nominal"]) for s in w12["sessions"]]
      == [(1, 10.0), (2, 5.0)],
      w12["phase"])
check("every session has a date on its weekday",
      all(pd.Timestamp(s["date"]).strftime("%a") == s["day"]
          for w in plan for s in w["sessions"]))
check("references cite the athlete's own bests",
      "254 W" in w1["sessions"][0]["reference"],
      w1["sessions"][0]["reference"])
summ = PL.plan_summary(plan)
check("the summary lists every interval with target and rest",
      len(summ) == 24 and set(summ.columns) >= {"Week", "Date", "Session",
                                                "Intervals", "Rest",
                                                "Reference"},
      f"{len(summ)} rows")

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    raise SystemExit(1)
print("ALL CHECKS PASS")
