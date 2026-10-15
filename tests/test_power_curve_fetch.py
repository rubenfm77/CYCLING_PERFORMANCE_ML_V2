"""True mean-maximal curve: parse intervals.icu power-curves, standard points.

The file's own rows can only bound MMP from below (fragments read low and
summary text misses the historic peaks), so the chart that needs TRUE bests
reads them from athlete/{id}/power-curves — same endpoint, shapes and
0.855 meter correction as the older pipeline. Synthetic payloads only (no
credentials, no network): every response shape the endpoint has returned,
nearest-within-10% matching, and the empty-frame fallback.

Run:  python tests/test_power_curve_fetch.py   (exit 0 = pass, 1 = fail)
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd

from core.interval_data import (_parse_power_curves, PC_WANT,
                                POWER_CORRECTION)

FAILED = []


def check(name, cond, detail=""):
    print(f"   {'PASS' if cond else 'FAIL'}  {name}"
          + (f" — {detail}" if (not cond and detail) else ""))
    if not cond:
        FAILED.append(name)


print("=" * 74)
print("1. response shapes all parse to the standard durations")
print("=" * 74)
curve = {"secs": [5, 60, 300, 600, 1200, 1800, 3600],
         "values": [723.0, 360.0, 288.0, 263.0, 245.0, 240.0, 256.0]}
for name, payload in [
    ("list-wrapped", {"list": [curve]}),
    ("bare list", [curve]),
    ("single dict", curve),
    ("alt keys",
     {"list": [{"seconds": [5, 60, 300, 600, 1200, 1800, 3600],
                "watts": [723.0, 360.0, 288.0, 263.0, 245.0, 240.0,
                          256.0]}]}),
]:
    pc = _parse_power_curves(payload, 57.0)
    check(f"{name}: seven standard points",
          len(pc) == 7 and list(pc["secs"]) == [5, 60, 300, 600, 1200,
                                                1800, 3600],
          pc["secs"].tolist() if len(pc) else "empty")
check("labels read like the old chart",
      list(_parse_power_curves({"list": [curve]}, 57.0)["duration"])
      == ["5s", "1 min", "5 min", "10 min", "20 min", "30 min", "60 min"])

print()
print("=" * 74)
print("2. nearest-within-10%, most coverage wins, correction applied")
print("=" * 74)
thin = {"secs": [5, 61, 305, 590, 1190, 1810, 3590],
        "values": [700.0, 350.0, 280.0, 260.0, 240.0, 235.0, 250.0]}
pc = _parse_power_curves({"list": [thin]}, 57.0)
check("off-by-seconds durations still match their target",
      len(pc) == 7, pc["secs"].tolist() if len(pc) else "empty")
sparse = {"secs": [5, 60, 300], "values": [700.0, 350.0, 280.0]}
full = {"secs": [5, 60, 300, 600, 1200, 1800, 3600],
        "values": [700.0, 350.0, 280.0, 260.0, 240.0, 235.0, 250.0]}
pc = _parse_power_curves({"list": [sparse, full]}, 57.0)
check("the fullest curve wins",
      len(pc) == 7 and abs(pc["watts"].iloc[0] - 700.0 * 0.855) < 0.1,
      pc["watts"].tolist() if len(pc) else "empty")
check("0.855 meter correction on every point",
      all(abs(w - v * POWER_CORRECTION) < 0.06
          for w, v in zip(pc["watts"], [700.0, 350.0, 280.0, 260.0, 240.0,
                                        235.0, 250.0])),
      pc["watts"].tolist() if len(pc) else "empty")
check("wkg follows at 57 kg",
      abs(pc["wkg"].iloc[0] - 700.0 * 0.855 / 57.0) < 0.01)
for bad in [None, {}, {"list": []}, {"list": [{"secs": [5]}]},
            {"list": [{"secs": [5, 60], "values": [1.0]}]},
            {"list": [{"nonsense": True}]}]:
    out = _parse_power_curves(bad, 57.0)
    check(f"unparseable {str(bad)[:28]!r} -> empty, well-formed frame",
          isinstance(out, pd.DataFrame) and not len(out)
          and list(out.columns) == ["secs", "duration", "watts", "wkg"])

print()
print("=" * 74)
print("3. the standard set matches the older chart, nothing else")
print("=" * 74)
check("want-durations are exactly the old seven",
      sorted(PC_WANT) == [5, 60, 300, 600, 1200, 1800, 3600]
      and PC_WANT[3600] == "60 min", PC_WANT)

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    raise SystemExit(1)
print("ALL CHECKS PASS")
