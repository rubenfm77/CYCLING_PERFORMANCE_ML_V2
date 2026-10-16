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

from core.data import _apply_english_labels, _label_from_description, load_data
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
print("1b. the intervals.icu Description is read like the older model")
print("=" * 74)
for text, want in [("FTP 2x20min", "FTP"), ("Morning END ride", "END"),
                   ("AEROBIC BASE z2", "AEROBIC BASE"),
                   ("VO2 MAX intervals", "VO2MAX"),
                   ("2x20 FTP threshold", "FTP"),
                   ("PIRAMIDAL session", "PIRAMIDAL"),
                   ("around the bEND", None), ("FT", None),
                   ("", None), (None, None)]:
    check(f"{text!r} -> {want!r}",
          _label_from_description(text) == want,
          _label_from_description(text))

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
b = float(tss[drv & (df["label_source"] == "auto-billat")].sum())
check("driver TSS splits by source and the parts sum to the slice",
      abs((w + a + b) - float(tss[drv].sum())) < 1e-6, (w, a, b))

print()
print("=" * 74)
print("4. every unlabelled session names its failing condition")
print("=" * 74)
from views.training import _unlabelled_rows  # noqa: E402
_synth = pd.DataFrame([
    # no window at all
    {"date": "2026-10-01", "tss": 60, "duration_s": 3600, "if_score": 0.70,
     "eftp": 234, "icu_pm_ftp_watts": np.nan, "icu_pm_ftp_secs": np.nan},
    # best window under the 18:00 floor
    {"date": "2026-10-02", "tss": 150, "duration_s": 8589, "if_score": 0.81,
     "eftp": 225, "icu_pm_ftp_watts": 247, "icu_pm_ftp_secs": 720},
    # best window over the 25:00 ceiling
    {"date": "2026-10-03", "tss": 141, "duration_s": 6816, "if_score": 0.86,
     "eftp": 225, "icu_pm_ftp_watts": 220, "icu_pm_ftp_secs": 1860},
    # under 95% of the ride's own eFTP
    {"date": "2026-10-04", "tss": 158, "duration_s": 8238, "if_score": 0.79,
     "eftp": 225, "icu_pm_ftp_watts": 207, "icu_pm_ftp_secs": 1200},
    # under the 15% share
    {"date": "2026-10-05", "tss": 140, "duration_s": 10053, "if_score": 0.71,
     "eftp": 221, "icu_pm_ftp_watts": 215, "icu_pm_ftp_secs": 1380},
    # passes every window check (corroboration's call, not the table's)
    {"date": "2026-10-06", "tss": 134, "duration_s": 5400, "if_score": 0.84,
     "eftp": 234, "icu_pm_ftp_watts": 228, "icu_pm_ftp_secs": 1300},
])
_ur = _unlabelled_rows(_synth)
check("one row per session, newest last",
      list(_ur["Date"]) == [f"2026-10-0{d}" for d in range(1, 7)],
      list(_ur["Date"]))
_whys = list(_ur["Why unlabelled"])
check("no window recorded",
      "no sustained threshold window" in _whys[0], _whys[0])
check("under the 18:00 floor",
      "under the 18:00" in _whys[1], _whys[1])
check("over the 25:00 ceiling",
      "over the 25:00" in _whys[2], _whys[2])
check("under 95% of own eFTP",
      "under 95%" in _whys[3], _whys[3])
check("under the 15% share",
      "under the 15% share" in _whys[4], _whys[4])
check("window checks pass: corroboration owns the refusal",
      "corroboration" in _whys[5], _whys[5])

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    raise SystemExit(1)
print("ALL CHECKS PASS")
