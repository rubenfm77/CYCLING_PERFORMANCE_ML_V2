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
check("steps carry duration, watts and cadence (%FTP in the title)",
      "- 20m 216-228w 90rpm" in t.splitlines()
      and "- 10m 120w 90rpm" in t.splitlines()
      and t.splitlines()[0] == "FTP 2x20min @ 90-95% (216-228W)")
check("a single effort has no repeat header",
      "Main Set\n- 20m 216-228w 90rpm" in
      PL.icu_text("FTP 20min", 20.0, 1, "90-95%", None))
check("short reps keep seconds, never 0 min",
      "- 30s 288-312w 90rpm" in
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
print("=" * 74)
print("5. book structures: watts beside %, purposes, totals, recovery")
print("=" * 74)
check("watts() renders bands and singles at FTP 240",
      PL.watts("95-100%", 240.0) == (228, 240)
      and PL.watts("120%", 240.0) == (288, 288), PL.watts("95-100%", 240.0))
check("threshold band follows the book at 20 min",
      PL.target_for("FTP", 20.0) == "95-100%"
      and PL.target_for("SST", 12.0) == "88-94%"
      and PL.target_for("TEMPO", 20.0) == "76-88%"
      and PL.target_for("VO2MAX", 5.0) == "106-115%"
      and PL.target_for("VO2MAX", 3.0) == "110-120%"
      and PL.target_for("BILLAT", 0.5) == "120%"
      and PL.target_for("TORQUE", 8.0) == "85-95%")
s20 = PL.build_session("FTP", 20.0, 2, 10.0, ftp=240.0,
                       ref=(244.0, "2026-09-30"))
check("title carries %FTP and watts",
      s20["title"] == "FTP 2x20m @ 95-100% (228-240W)", s20["title"])
check("purpose and total ride along",
      s20["purpose"] == "Sustained threshold power for climbing"
      and s20["total_min"] == 78.0, (s20["purpose"], s20["total_min"]))
check("warm-up opens with the book ramp and openers",
      "- 10m ramp 50%-65% 90rpm" in s20["text"].splitlines()
      and "- 1m 75% 100rpm" in s20["text"].splitlines())
check("one trailing note line, never a second repeat header",
      s20["text"].rstrip().splitlines()[-1].startswith("Targets at FTP 240W")
      and s20["text"].count("Main Set") == 1)
sb = PL.build_session("BILLAT", 0.5, 20, 0.5, ftp=240.0)
check("billat flattens: no nested repeats, 5 min every 10 pairs",
      "Main Set 20x" not in sb["text"]
      and sb["text"].count("- 30s 288w 90rpm") == 20
      and sb["text"].count("- 5m 120w 90rpm") == 1)
st_ = PL.build_session("TORQUE", 8.0, 5, 5.0, ftp=240.0)
check("low-cadence work carries its own cadence",
      "- 8m 204-228w 55rpm" in st_["text"].splitlines())
sr = PL.build_session("REC", 45.0, 1, None, ftp=240.0)
check("recovery week still pastes: spin plus a note",
      "- 45m 120w 90rpm" in sr["text"].splitlines())
sz = PL.build_session("Z2", 120.0, 1, None, ftp=240.0)
check("endurance block reads watts with book %FTP in the title",
      "- 120m 134-180w 90rpm" in sz["text"].splitlines()
      and "(134-180W)" in sz["title"]
      and "56-75%" in sz["title"], sz["title"])
tab = PL.session_table([{**s20, "day": "Sat", "date": "2026-10-17"},
                        {**sb, "day": "Wed", "date": "2026-10-14"}])
check("the summary table lists every pasteable session",
      len(tab) == 2 and set(tab.columns) >= {"Date", "Session", "Intervals",
                                             "Rest", "Reference"})
check("easy-ride duration defaults honestly with nothing to measure",
      PL.endurance_med_duration(pd.DataFrame(
          {"training_type": ["END"], "duration_h": [3.0]})) == 2.0)

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    raise SystemExit(1)
print("ALL CHECKS PASS")
