"""Physiology classification, isolation rule, ley de potencias fits (credential-free)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from ml.interval_watts import measured, phys_class, power_law

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        FAILS.append(name)
        print(f"  FAIL  {name} :: {detail}")


print("1. physiology classification")
check("599 s is VO2MAX", phys_class(599) == "VO2MAX", phys_class(599))
check("600 s (10:00) is FTP", phys_class(600) == "FTP", phys_class(600))
check("30 s is VO2MAX", phys_class(30) == "VO2MAX", phys_class(30))
check("30:00 is FTP", phys_class(1800) == "FTP", phys_class(1800))

print("2. isolated pushes are discarded")
df = pd.DataFrame([
    ("FTP", "2026-01-01", "MS: 1x15min", "['1x 15m 250w']"),
    ("FTP", "2026-01-02", "MS: 1x15min", "['1x 15m 250w']"),
    ("FTP", "2026-01-02", "MS: 1x20min", "['1x 20m 260w']"),
    ("FTP", "2026-01-03", "MS: 3x15min", "['3x 15m 250w']"),
    ("VO2MAX", "2026-01-04", "MS: 1x30s", "['1x 30s 400w']"),
], columns=["training_type", "date", "WorkoutDescription", "interval_summary"])
df["date"] = pd.to_datetime(df["date"])
M = measured(df)
check("lone 15-min effort on its day is discarded",
      not ((M["date"] == "2026-01-01").any()),
      str(M["date"].tolist()))
check("two single efforts on one day both stay",
      int((M["date"] == "2026-01-02").sum()) == 2,
      str(M[M["date"] == "2026-01-02"]["n"].tolist()))
check("a 3x rep block is several intervals, stays",
      bool((M["date"] == "2026-01-03").any()), str(M["date"].tolist()))
check("sub-minute push is never discarded by this rule",
      bool((M["secs"] < 60).any()), str(M["secs"].tolist()))
check("discard count reported",
      M.attrs.get("isolated_discarded") == 1,
      str(M.attrs.get("isolated_discarded")))
check("every kept effort carries a physiology label",
      set(M["phys"]) <= {"FTP", "VO2MAX"}, str(set(M["phys"])))
check("physiology follows the 10-min boundary",
      set(M[M["secs"] >= 600]["phys"]) == {"FTP"} and
      set(M[M["secs"] < 600]["phys"]) == {"VO2MAX"})

print("3. ley de potencias: both models, side by side")
secs = np.array([30, 60, 120, 180, 300, 480, 600, 900, 1200, 1800], float)
w = 250 + 18000 / secs            # CP = 250 W, W' = 18 kJ exactly
rows = []
for i, (t, ww) in enumerate(zip(secs, w)):
    d = pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)
    # payload shape intervals.icu actually writes: minutes with `m`, seconds
    # with `s` - never "1200s"
    span = (f"{int(t)}s" if t < 60 else f"{int(t) // 60}m")
    rows.append(("FTP", d, "", f"['1x {span} {int(ww)}w']"))
    rows.append(("FTP", d, "", f"['1x {span} {int(ww)}w']"))
df2 = pd.DataFrame(rows, columns=["training_type", "date",
                                  "WorkoutDescription", "interval_summary"])
M2 = measured(df2)
fit = power_law(M2, "FTP")
print("   ", {k: fit[k] for k in ("ok", "cp", "w_prime_kj", "r2_cp",
                                  "b", "r2_law", "n", "t_lo", "t_hi")})
check("fit reports ok on 10 durations", fit["ok"], fit["reason"])
check("CP recovers 250 W", abs(fit["cp"] - 250) < 2, str(fit["cp"]))
check("W' recovers 18 kJ", abs(fit["w_prime_kj"] - 18) < 1,
      str(fit["w_prime_kj"]))
check("CP model R^2 ~ 1", fit["r2_cp"] > 0.999, str(fit["r2_cp"]))
check("power-law R^2 on hyperbolic data is high but < 1",
      0.8 < fit["r2_law"] < 1.0, str(fit["r2_law"]))
check("duration range reported",
      fit["t_lo"] == 30 and fit["t_hi"] == 1800,
      f"{fit['t_lo']}..{fit['t_hi']}")
check("n of distinct durations reported", fit["n"] == 10, str(fit["n"]))

print("4. ley de potencias: refuses to fake a law")
check("fewer than 4 durations -> not ok",
      not power_law(M2.head(2), "FTP")["ok"])
check("wrong type -> not ok",
      not power_law(M2, "SST")["ok"], power_law(M2, "SST")["reason"])
check("empty frame -> not ok", not power_law(pd.DataFrame(), None)["ok"])
check("no phys column breaks nothing",
      "phys" in measured(pd.DataFrame()).columns)

print()
if FAILS:
    print(f"FAILURES: {len(FAILS)}")
    for f in FAILS:
        print(" -", f)
    sys.exit(1)
print("ALL PASS")
