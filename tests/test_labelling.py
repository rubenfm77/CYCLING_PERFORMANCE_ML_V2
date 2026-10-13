"""Labelling honesty: English display names, corroborated auto-FTP, honest pie.

Synthetic frames plus the local CSV (no credentials, no network):
  1. Catalan/Spanish workout names read in English, 1:1, nothing merged —
     and PYRAMID stays inside MAIN_TYPES/F uniqueness so its 24 sessions
     cannot fall out of any type-based view.
  2. The auto-FTP power gate alone is not a label: a peak-meter window with
     no sustained detector block behind it (surge inside another ride) stays
     unlabelled; a joined block vouches; no cache rows means unknown.
  3. The driver pie divides ALL TSS (drivers + base + unlabelled = 100 %),
     and the driver slice splits by source (athlete vs auto-rule), so any
     gap to the older model (which had no auto-rule) is visible, not hidden.

Run:  python tests/test_labelling.py     (exit 0 = pass, 1 = fail)
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

from core.data import _apply_english_labels, load_data
from core.interval_data import match_threshold_run
from core.theme import FTP_DRIVERS, MAIN_TYPES, TYPE_ENGLISH

FAILED = []


def check(name, cond, detail=""):
    print(f"   {'PASS' if cond else 'FAIL'}  {name}"
          + (f" — {detail}" if (not cond and detail) else ""))
    if not cond:
        FAILED.append(name)


print("=" * 74)
print("1. English labels: exact, 1:1, constants follow PYRAMID")
print("=" * 74)
f = pd.DataFrame({"training_type": ["PIRAMIDAL", "Llançadera.", "FTP",
                                    "PROGRESSIU (105%)", "REC", "END",
                                    "BILLAT", "—", None]})
n = _apply_english_labels(f)
check("four Catalan/Spanish names translated, rest untouched",
      n == 4, f["training_type"].tolist())
check("PYRAMID keeps its sessions inside the agreed types",
      "PYRAMID" in MAIN_TYPES and "PYRAMID" in FTP_DRIVERS
      and "PIRAMIDAL" not in f["training_type"].tolist())
check("every driver is still an agreed type",
      set(FTP_DRIVERS) <= set(MAIN_TYPES), FTP_DRIVERS)
check("mapping is 1:1 — no two names merge",
      len(set(TYPE_ENGLISH.values())) == len(TYPE_ENGLISH))
check("blanks are never translated",
      f["training_type"].iloc[-2] == "—"
      and pd.isna(f["training_type"].iloc[-1]))

print()
print("=" * 74)
print("2. a window without a block behind it is not a session")
print("=" * 74)


def _rows(specs):
    return pd.DataFrame([{"seq": float(i + 1), "iv_type": t,
                          "secs": float(s), "avg_w": float(w)}
                         for i, (t, s, w) in enumerate(specs)])


joined = _rows([("WORK", 683, 242), ("RECOVERY", 18, 145),
                ("WORK", 511, 245)])
check("683 + 511 across an 18 s dip vouches a 1260 s @ 240 W window",
      match_threshold_run(joined, 240.0, 1260.0) is True)
short = _rows([("WORK", 300, 244), ("RECOVERY", 27, 100),
               ("WORK", 300, 244)])
check("two 5-min pieces do not vouch a 21-min window",
      match_threshold_run(short, 244.0, 1260.0) is False)
soft = _rows([("WORK", 700, 150), ("RECOVERY", 30, 90),
              ("WORK", 700, 150)])
check("a long but soft block does not vouch threshold watts",
      match_threshold_run(soft, 244.0, 1450.0) is False)
check("no rows means unknown, never a rejection",
      match_threshold_run(pd.DataFrame(), 244.0, 1260.0) is None)
missing = pd.DataFrame([{"seq": [1.0], "iv_type": ["MISSING"],
                         "secs": [0.0], "avg_w": [0.0]}])
check("a MISSING sentinel proves nothing either way",
      match_threshold_run(missing, 244.0, 1260.0) is None)
check("a missing window is unjudgeable",
      match_threshold_run(joined, np.nan, 1260.0) is None)

print()
print("=" * 74)
print("3. the pie divides all TSS, and the driver slice shows its sources")
print("=" * 74)
df = load_data()
tss = pd.to_numeric(df["tss"], errors="coerce").fillna(0.0)
drv = df["training_type"].isin(FTP_DRIVERS)
base = df["training_type"].isin(MAIN_TYPES) & ~drv
unl = ~df["training_type"].isin(MAIN_TYPES)
shares = {k: float(tss[m].sum() / tss.sum() * 100)
          for k, m in [("drivers", drv), ("base", base),
                       ("unlabelled", unl)]}
check("drivers + base + unlabelled = all TSS",
      abs(sum(shares.values()) - 100.0) < 1e-9, shares)
check("the unlabelled slice is real load, not a rounding error",
      shares["unlabelled"] > 1.0, shares)
w = float(tss[drv & (df["label_source"] == "workout")].sum())
a = float(tss[drv & (df["label_source"] == "auto-ftp")].sum())
check("driver TSS splits by source and the parts sum to the slice",
      abs((w + a) - float(tss[drv].sum())) < 1e-6, (w, a))

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    raise SystemExit(1)
print("ALL CHECKS PASS")
