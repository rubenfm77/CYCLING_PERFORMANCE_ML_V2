"""The controls a page asks for: training type and year, never an effort filter.

Both pages the athlete pointed at — Fitness ("measured interval watts over
time" + the law under it) and Power law — used to carry an FTP / VO2MAX
selectbox. On Power law it was worse than redundant (the x-axis IS duration,
so a 3-minute and a 20-minute effort are two points on one curve), and on
Fitness it repeated what the training type already says: every rep length
class stops and starts at ten minutes, so long and short efforts are already
on separate lines with separate numbers.

This pins the widgets each section asks for. It renders both sections in
bare-mode Streamlit (every widget returns its default) and records the labels,
so a filter coming back breaks the test instead of the athlete finding it in
the dashboard.

Runs on the local CSV through core.data.load_data() — no credentials, no
network. Run:  python tests/test_page_controls.py   (exit 0 = pass, 1 = fail)
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from types import SimpleNamespace  # noqa: E402

import streamlit as st  # noqa: E402

from core.data import load_data  # noqa: E402
from views import interval_watts as iwv  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")
        FAILS.append(name)


df = load_data()

seen = []
_orig = st.selectbox


def spy(label, *a, **k):
    seen.append(str(label))
    return _orig(label, *a, **k)


st.selectbox = spy

print("=" * 72)
print("Fitness page — fitness_section()")
print("=" * 72)
seen.clear()
iwv.fitness_section(df)
fit = list(seen)
print(f"   controls: {fit}")
check("Training type and Year only", fit == ["Training type", "Year"], fit)
check("no Effort filter", "Effort" not in fit, fit)

print("=" * 72)
print("Power law page — render()")
print("=" * 72)
seen.clear()
iwv.render(None, SimpleNamespace(df_all=df))
law = list(seen)
print(f"   controls: {law}")
check("Training type and Year only", law == ["Training type", "Year"], law)
check("no Effort filter", "Effort" not in law, law)

st.selectbox = _orig

print()
if FAILS:
    print("FAILURES:", FAILS)
    sys.exit(1)
print("ALL CHECKS PASS")
