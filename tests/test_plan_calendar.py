"""Plan page calendar logic: the athlete's own plan outranks the template.

Pure-function checks (no rendering): week boundaries, event-to-day mapping,
the ridden marker both ways round, and the withholding rule — a template
block never lands on a day the athlete's calendar already defines.

Run: python run_test.py tests.test_plan_calendar
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

for _p in (os.path.join(ROOT, ".env"),
           r"C:\Users\levod\Downloads\CYCLING_DATA\PY\cycling-performance-ml\.env"):
    if os.path.exists(_p):
        for _line in open(_p, encoding="utf-8"):
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
        break

import pandas as pd  # noqa: E402

from views.plan import _is_ridden, _plan_rows, _split_dated, _week_days  # noqa: E402

PASS = []


def ok(cond, msg):
    assert cond, msg
    PASS.append(msg)
    print(f"  PASS  {msg}")


print("=" * 70)
print("1. the calendar week runs Monday to Sunday")
print("=" * 70)
# 2026-10-19 is a Monday; 2026-10-21 a Wednesday — both must give the SAME
# week: Mon 19 .. Sun 25.
for _anchor in ("2026-10-19", "2026-10-21", "2026-10-25"):
    d = _week_days(pd.Timestamp(_anchor))
    ok(len(d) == 7, f"7 days from anchor {_anchor}")
    ok(d[0].dayofweek == 0, f"week from {_anchor} starts Monday")
    ok(str(d[0].date()) == "2026-10-19" and str(d[-1].date()) == "2026-10-25",
       f"anchor {_anchor} lands on 19–25 Oct")
# A Sunday anchor belongs to the week that is ending, not the next one.
d = _week_days(pd.Timestamp("2026-10-18"))  # Sunday
ok(str(d[0].date()) == "2026-10-12" and str(d[-1].date()) == "2026-10-18",
   "Sunday 18 Oct stays in the week of 12–18 Oct")

print()
print("=" * 70)
print("2. events map onto their own day; open days stay open")
print("=" * 70)
events = [
    {"date": "2026-10-21", "name": "FTP", "description": "- 20m 235-243w",
     "id": 1, "paired_activity_id": None},
    {"date": "2026-10-23", "name": "BILLAT", "description": "",
     "id": 2, "paired_activity_id": None},
    {"date": "2026-10-99", "name": "OUT OF RANGE", "description": "",
     "id": 3, "paired_activity_id": None},
]
days = _week_days(pd.Timestamp("2026-10-21"))
rows = _plan_rows(events, days)
ok(len(rows) == 7, "one row per day of the week")
by_date = {r["date"]: r for r in rows}
ok([e["name"] for e in by_date["2026-10-21"]["events"]] == ["FTP"],
   "the FTP event sits on its own Wednesday")
ok([e["name"] for e in by_date["2026-10-23"]["events"]] == ["BILLAT"],
   "the BILLAT event sits on its own Friday")
ok(by_date["2026-10-22"]["events"] == [], "an unplanned Thursday is OPEN, "
   "never filled from any other day")
ok("2026-10-99" not in by_date, "an out-of-range event never lands in the week")

print()
print("=" * 70)
print("3. a template block never lands on a planned day")
print("=" * 70)
dated = [
    {"date": pd.Timestamp("2026-10-21").date(), "day": "Wed", "title": "Z2"},
    {"date": pd.Timestamp("2026-10-22").date(), "day": "Thu", "title": "REC"},
    {"date": pd.Timestamp("2026-10-24").date(), "day": "Sat", "title": "Z2"},
]
paste, held = _split_dated(dated, {"2026-10-21", "2026-10-24"})
ok([s["title"] for s in paste] == ["REC"], "the open Thursday keeps its block")
ok([s["title"] for s in held] == ["Z2", "Z2"],
   "both blocks on planned days are withheld")
paste2, held2 = _split_dated(dated, set())
ok(len(paste2) == 3 and not held2, "no calendar -> nothing withheld")
paste3, held3 = _split_dated(dated, None)
ok(len(paste3) == 3 and not held3, "calendar unknown -> nothing withheld "
   "(honestly, not silently)")
# String plan dates must match date objects on the blocks.
paste4, held4 = _split_dated(dated, ["2026-10-21"])
ok(len(held4) == 1 and len(paste4) == 2, "string plan dates match date blocks")

print()
print("=" * 70)
print("4. the ridden marker reads BOTH sides of the pairing")
print("=" * 70)
df = pd.DataFrame({
    "id": ["iRIDDEN1", "iOTHER"],
    "paired_event_id": [12345.0, float("nan")],
})
ok(_is_ridden({"id": 999, "paired_activity_id": "iRIDDEN1"}, df) is True,
   "event->activity pairing counts as ridden")
ok(_is_ridden({"id": 12345, "paired_activity_id": None}, df) is True,
   "ride->event pairing counts as ridden")
ok(_is_ridden({"id": 42, "paired_activity_id": "iMISSING"}, df) is False,
   "an unlinked event is NOT ridden")
ok(_is_ridden({"id": 12345}, pd.DataFrame({"id": [], "paired_event_id": []}))
   is False, "an empty frame is safe and answers False")
ok(_is_ridden({"id": 7}, None) is False, "no frame at all answers False")

print()
print("=" * 70)
print("5. the live calendar read (structural only — content is the athlete's)")
print("=" * 70)
from core.data import fetch_events_between  # noqa: E402

res = fetch_events_between("2026-10-19", "2026-11-01")
if res is None:
    print("  NOTE  calendar unreachable in this environment — structural "
          "check skipped (the page shows the 'not read' callout instead)")
else:
    ok(isinstance(res, list), "a reachable calendar answers a list, not None")
    ok(all(set(("id", "date", "name", "description", "paired_activity_id"))
           <= set(e) for e in res), "every event carries the documented keys")
    ok([e["date"] for e in res] == sorted(e["date"] for e in res),
       "events come back sorted by date")
    print(f"  NOTE  {len(res)} event(s) in 19 Oct – 1 Nov window")

print()
print("=" * 70)
print(f"RESULT: ALL PASS ({len(PASS)} checks)")
print("=" * 70)
