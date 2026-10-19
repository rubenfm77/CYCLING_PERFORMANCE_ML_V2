"""Every plan template pastes: the chosen week always yields copy blocks.

Renders views/plan.py headless in bare-mode Streamlit (no credentials, no
network — sync fails fast into the cache fallback) once per template
option, spying on st.code. Fails if any option renders zero blocks, a block
without a Main Set, a work step without %FTP, or a title without watts —
the exact "nothing to paste" failure the athlete reported.

Runs on the local CSV through core.data.load_data().
Run:  python tests/test_plan_paste.py     (exit 0 = pass, 1 = fail)
"""
import os
import re
import sys
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import streamlit as st  # noqa: E402
import pandas as pd  # noqa: E402

from core.data import load_data  # noqa: E402
from views import plan as plan_view  # noqa: E402

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
ctx = SimpleNamespace(df_all=df_all, df=df_all, cutoff=None)

_orig_radio = st.radio
_orig_code = st.code
_orig_fetch = plan_view.fetch_events_between


def render_with(radio_choice, fetch=None):
    # The events fetch is stubbed: the template machinery's contract (blocks
    # always build) must hold regardless of what the live calendar holds —
    # the WITHHOLDING rule has its own checks below and in
    # tests.test_plan_calendar. Only when a stub is passed does the page see
    # that calendar.
    codes = []
    split = {"at": None}
    plan_view.fetch_events_between = fetch or (lambda *a, **k: [])

    def _radio(label, options, *a, **k):
        # The template radio sits BETWEEN the page's calendar section (which
        # shows the athlete's own event steps with st.code) and the paste
        # section. Mark that boundary: paste blocks are the st.code calls
        # after the radio, the calendar's steps are not paste blocks.
        split["at"] = len(codes)
        return radio_choice if radio_choice in list(options) else list(options)[0]

    st.radio = _radio
    st.code = lambda body, *a, **k: codes.append(str(body))
    try:
        plan_view.render(None, ctx)
    finally:
        st.radio = _orig_radio
        st.code = _orig_code
        plan_view.fetch_events_between = _orig_fetch
    at = split["at"] if split["at"] is not None else 0
    return codes[at:]


# First pass with the default choice: harvest the option list from the page.
seen_options = []


def _harvest(label, options, *a, **k):
    seen_options.extend(list(options))
    return list(options)[0]


st.radio = _harvest
plan_view.fetch_events_between = lambda *a, **k: []
codes_default, harvest_err = [], None
try:
    st.code = lambda body, *a, **k: codes_default.append(str(body))
    plan_view.render(None, ctx)
except Exception as e:                                       # noqa: BLE001
    harvest_err = f"{type(e).__name__}: {e}"
finally:
    st.radio = _orig_radio
    st.code = _orig_code
    plan_view.fetch_events_between = _orig_fetch

check("the plan page renders headless with no credentials",
      harvest_err is None, harvest_err)
check("template options exist",
      len(seen_options) >= 1, seen_options)

WATT = re.compile(r"\(\d+-\d+W\)|\(\d+W\)")
STEPW = re.compile(r"\d+w")
for opt in seen_options:
    codes = render_with(opt)
    check(f"{opt} yields at least one paste block", len(codes) >= 1,
          f"{len(codes)} blocks")
    for i, b in enumerate(codes):
        lines = b.splitlines()
        check(f"{opt} block {i + 1} has a title with %FTP and watts",
              len(lines) > 0 and "%" in lines[0]
              and bool(WATT.search(lines[0])),
              lines[0] if lines else "empty")
        check(f"{opt} block {i + 1} has a Main Set",
              "Main Set" in b, b[:120])
        steps = [ln for ln in lines if ln.startswith("- ")]
        try:
            _mi = next(i for i, ln in enumerate(lines)
                       if ln.startswith("Main Set"))
            _ci = next(i for i, ln in enumerate(lines)
                       if ln.startswith("Cooldown"))
        except StopIteration:
            _mi, _ci = -1, -1
        work = [ln for ln in lines[_mi + 1:_ci] if ln.startswith("- ")]
        check(f"{opt} block {i + 1} main-set steps carry watts",
              len(work) > 0 and all(bool(STEPW.search(s)) for s in work),
              work[:3])

print()


def _full_calendar(date_from, date_to):
    """A calendar that plans EVERY day of the requested window."""
    return [{"id": i + 1, "date": str(d.date()), "name": "AEROBIC BASE",
             "description": "", "paired_activity_id": None}
            for i, d in enumerate(pd.date_range(date_from, date_to))]


if seen_options:
    _opt = seen_options[0]
    _withheld = render_with(_opt, fetch=_full_calendar)
    check("a calendar that plans every day withholds every paste block",
          len(_withheld) == 0,
          f"{len(_withheld)} block(s) pasted over a fully planned week")

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    raise SystemExit(1)
print("ALL CHECKS PASS")
