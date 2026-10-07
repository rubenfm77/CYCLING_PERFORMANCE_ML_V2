"""Rule A and the series names, on synthetic sets.

Rule A (athlete's rule): an effort under 8 minutes with MORE THAN ONE rep
is VO2MAX work — applied where the session carries no label (a label always
wins). It retires the ambiguous heuristic buckets for 15 s–8 min multi-rep
work. Sprints (<15 s) and the 30 s Billat/micro-rep protocol patterns keep
their names. Series names ("FTP (10-20)") are display only: grouping still
uses rep_class, so both pages keep charting the same days.

Run:  python tests/test_taxonomy.py     (exit 0 = pass, 1 = fail)
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

from ml.interval_watts import CLS_SHORT, REP_ORDER, rep_class, series_name
from ml.type_comparison import prep_types

FAILED = []


def check(name, cond, detail=""):
    print(f"   {'PASS' if cond else 'FAIL'}  {name}"
          + (f" — {detail}" if (not cond and detail) else ""))
    if not cond:
        FAILED.append(name)


print("=" * 74)
print("1. series names are display only")
print("=" * 74)
check("FTP bins read exactly as specified",
      series_name("FTP", "10-20 min") == "FTP (10-20)"
      and series_name("FTP", "20-30 min") == "FTP (20-30)"
      and series_name("FTP", "30+ min") == "FTP (>30)")
check("every class has a short name",
      set(CLS_SHORT) == set(REP_ORDER), str(sorted(set(CLS_SHORT))))
check("names never classify: the class is still rep_class of the seconds",
      rep_class(1207) == "10-20 min" and rep_class(1320) == "20-30 min")

print()
print("=" * 74)
print("2. rule A files unlabelled short multi-rep work as VO2MAX")
print("=" * 74)


def _row(fam_pre, rep_secs, reps, db=""):
    return {"activity_id": "i1", "date": pd.Timestamp("2026-05-01"),
            "_key": f"k{rep_secs}", "reps": reps, "rep_secs": float(rep_secs),
            "set_w": 250.0, "set_np": np.nan, "set_hr": np.nan,
            "intensity": 100.0, "load": np.nan, "cad": np.nan,
            "decoupling": np.nan, "first_seq": np.nan, "last_seq": np.nan,
            "rest": 60.0, "name": "", "temp": np.nan, "moving_time": np.nan,
            "act_iv_n": np.nan, "act_iv_secs": np.nan, "act_iv_dist": np.nan,
            "act_start": pd.NaT, "act_iv_rows": np.nan,
            "set_ss_cp_w": np.nan, "set_ss_w_prime_kj": np.nan,
            "set_w5s_cv": np.nan, "db_type": db, "Source": "detected",
            "style": fam_pre, "q_flag": ""}


base = [_row("FTP / threshold sets", 300, 4),      # 5x5 unlabelled
        _row("VO₂ max sets", 180, 5),              # 5x3 unlabelled
        _row("", 90, 3),                           # 3x90s unlabelled
        _row("BILLAT", 30, 10),                    # Billat pattern
        _row("sprints", 12, 4),                    # short sprints
        _row("FTP / threshold sets", 300, 4, db="FTP"),  # labelled: wins
        _row("", 300, 1),                          # single: not multi-rep
        _row("long efforts", 660, 2)]              # 2x11: at/above 8 min
s = prep_types(pd.DataFrame(base))
fam = dict(zip(s["rep_secs"].astype(str) + "/" + s["reps"].astype(str)
               + "/" + s["db_type"].astype(str), s["family"]))
check("5x5 unlabelled -> VO2MAX", fam["300.0/4/"] == "VO2MAX", fam)
check("5x3 unlabelled -> VO2MAX", fam["180.0/5/"] == "VO2MAX", fam)
check("3x90s unlabelled -> VO2MAX", fam["90.0/3/"] == "VO2MAX", fam)
check("Billat pattern keeps its protocol name",
      fam["30.0/10/"] == "BILLAT", fam)
check("sprints keep their name", fam["12.0/4/"] == "sprints", fam)
check("the athlete's label always wins",
      fam["300.0/4/FTP"] == "FTP", fam)
check("a single rep is not multi-rep work",
      fam["300.0/1/"] == "single efforts", fam)
check("2x11 sits above the 8-minute band",
      fam["660.0/2/"] == "long efforts", fam)

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    raise SystemExit(1)
print("ALL CHECKS PASS")
