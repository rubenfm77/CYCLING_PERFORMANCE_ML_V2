"""The book section: EF, decoupling and the nine quadrants — honestly.

Synthetic frames only: monthly medians carry n, performance prefers eFTP
and falls back to CTL with the source attached, quadrants follow the book's
layout translated to English, and RPP reports its missing RPE instead of
being faked.

Run:  python tests/test_book.py     (exit 0 = pass, 1 = fail)
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

from ml import efficiency as ef

_ACCENTS = set("áéíóúñÁÉÍÓÚÑ")

FAILED = []


def check(name, cond, detail=""):
    print(f"   {'PASS' if cond else 'FAIL'}  {name}"
          + (f" — {detail}" if (not cond and detail) else ""))
    if not cond:
        FAILED.append(name)


print("=" * 74)
print("1. monthly medians carry n, nothing pooled across months")
print("=" * 74)
df = pd.DataFrame([
    {"date": "2026-08-01", "efficiency": 1.40},
    {"date": "2026-08-15", "efficiency": 1.60},
    {"date": "2026-09-01", "efficiency": 1.50},
    {"date": "2026-09-02", "efficiency": np.nan},
])
m = ef.monthly_ef(df)
check("two months, medians exact",
      len(m) == 2 and abs(m["med_ef"].iloc[0] - 1.50) < 1e-9
      and abs(m["med_ef"].iloc[1] - 1.50) < 1e-9, m.to_dict("records"))
check("n counts sessions with heart rate only",
      list(m["n"]) == [2, 1], m["n"].tolist())
check("empty in, empty out",
      not len(ef.monthly_ef(pd.DataFrame())))

iv = pd.DataFrame([
    {"date": "2026-08-01", "iv_type": "WORK", "decoupling": 2.0},
    {"date": "2026-08-02", "iv_type": "WORK", "decoupling": 4.0},
    {"date": "2026-08-03", "iv_type": "RECOVERY", "decoupling": 9.0},
    {"date": "2026-08-04", "iv_type": "WORK", "decoupling": np.nan},
])
d = ef.monthly_decoupling(iv)
check("decoupling reads WORK rows only",
      len(d) == 1 and abs(d["med_dec"].iloc[0] - 3.0) < 1e-9
      and int(d["n"].iloc[0]) == 2, d.to_dict("records"))
check("no column, no crash",
      not len(ef.monthly_decoupling(pd.DataFrame())))

print()
print("=" * 74)
print("2. performance prefers eFTP, falls back to CTL, says which")
print("=" * 74)
de = pd.DataFrame([
    {"date": "2026-09-01", "eftp": 230.0, "ctl": 100.0},
    {"date": "2026-09-20", "eftp": 234.0, "ctl": 110.0},
])
check("eFTP delta with source attached",
      ef.perf_delta(de) == (4.0, "eFTP W"), ef.perf_delta(de))
dc = pd.DataFrame([
    {"date": "2026-09-01", "eftp": np.nan, "ctl": 100.0},
    {"date": "2026-09-20", "eftp": np.nan, "ctl": 107.0},
])
check("CTL fallback, labelled as CTL",
      ef.perf_delta(dc) == (7.0, "CTL pts"), ef.perf_delta(dc))
check("one point is not a delta",
      ef.perf_delta(dc.iloc[:1])[0] != ef.perf_delta(dc.iloc[:1])[0],
      ef.perf_delta(dc.iloc[:1]))

print()
print("=" * 74)
print("3. the nine quadrants, in English, book layout")
print("=" * 74)
check("nine cells, numbered like the book",
      len(ef.QUADRANTS) == 9
      and sorted(v[0] for v in ef.QUADRANTS.values()) == list(range(1, 10)))
check("no Spanish anywhere in the matrix",
      not any(set(t + s) & _ACCENTS
              for _, t, s in ef.QUADRANTS.values()))
check("matrix text is English (spot check)",
      "Tapering" in ef.QUADRANTS[("up", "high")][1]
      and "Detraining" in ef.QUADRANTS[("down", "high")][1])
for tsb, dlt, no in [(-19.4, 5.0, 1), (0.0, 0.0, 5), (15.0, -5.0, 9),
                     (-10.0, 2.1, 2), (10.0, -2.1, 8),
                     (-10.0, 2.0, 5), (10.0, -2.0, 5)]:
    check(f"TSB {tsb}, delta {dlt} -> cell {no}",
          ef.readiness(tsb, dlt)["no"] == no, ef.readiness(tsb, dlt))

print()
print("=" * 74)
print("4. RPP reports the gap instead of faking the table")
print("=" * 74)
stt = ef.rpp_status(pd.DataFrame({"icu_rpe": [np.nan, np.nan],
                                  "session_rpe": [np.nan, np.nan]}))
check("no RPE, no RPP — with n and what is needed",
      stt == {"sessions": 2, "rpe_n": 0,
              "needs": "per-session RPE (icu_rpe/session_rpe) to divide "
                       "power by"},
      stt)

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    raise SystemExit(1)
print("ALL CHECKS PASS")
