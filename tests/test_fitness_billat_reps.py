"""Fitness BILLAT line: short-work sessions are drawn, honestly.

BILLAT's work is sub-90-second reps by definition, so the sustained
physiology bands can never show its training. Regression test on the LOCAL
file + interval cache (no credentials, no network): fitness_section() with
BILLAT/2026 selected must draw the per-day rep-median line from the
detector's short WORK rows — the same trend the Intervals detail's session
averages show (295 -> 323 W across Aug-Sep 2026) — and must not crash when
the cache is absent.

Run:  python tests/test_fitness_billat_reps.py     (exit 0 = pass, 1 = fail)
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import streamlit as st  # noqa: E402

from core.data import load_data  # noqa: E402
from core.interval_data import read_intervals  # noqa: E402
import views.interval_watts as iwv  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")
        FAILED.append(name)


df_all = load_data()
iv = read_intervals("2019-01-01T00:00:00")

FIGS = []
iwv.show = lambda fig=None, *a, **k: FIGS.append(fig) if fig is not None else None

_orig_sb = st.selectbox


def _pick(label, options, *a, **k):
    if "Training type" in str(label):
        return "BILLAT"
    if "Year" in str(label):
        return 2026
    return _orig_sb(label, options, *a, **k)


st.selectbox = _pick
try:
    iwv.fitness_section(df_all, iv)
finally:
    st.selectbox = _orig_sb

want_title = "BILLAT — interval watts over time, 2026"
fig = next((f for f in FIGS
            if str((f.layout.title.text or "")).startswith(want_title)), None)
check("the BILLAT time chart renders", fig is not None,
      [str(f.layout.title.text or "")[:50] for f in FIGS])
if fig is not None:
    names = [t.name for t in fig.data]
    reps = next((t for t in fig.data if "reps" in str(t.name)), None)
    check("a per-day rep line is drawn beside the sustained bands",
          reps is not None, names)
    if reps is not None:
        check("five BILLAT days, oldest first",
              len(reps.x) == 5, list(reps.x))
        check("rep medians follow the detail's session means, 295 -> 323 W",
              list(reps.y)[0] < 300 < list(reps.y)[-1]
              and 320 <= float(list(reps.y)[-1]) <= 326,
              [float(y) for y in reps.y])
        check("every rep point carries its n",
              all("n =" in str(c[1]) for c in reps.customdata),
              reps.customdata)

# Without the cache the line has no source — the section must say so or
# fall back, never crash.
FIGS2 = []
iwv.show = lambda fig=None, *a, **k: FIGS2.append(fig) if fig is not None else None
st.selectbox = _pick
try:
    iwv.fitness_section(df_all, None)
    nocache_ok = True
except Exception as e:                                  # noqa: BLE001
    nocache_ok, nocache_err = False, f"{type(e).__name__}: {e}"
finally:
    st.selectbox = _orig_sb
check("no cache, no crash", nocache_ok,
      nocache_err if not nocache_ok else "")

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    sys.exit(1)
print("ALL CHECKS PASS")
