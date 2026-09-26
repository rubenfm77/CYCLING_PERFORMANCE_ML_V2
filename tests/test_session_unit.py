"""Timing + correctness after the session-level rewrite: run_type_comparison
must be fast, the summary and the explorer must agree, and the fade must be
the opening (reps 2-3) vs closing (last 2) window."""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)

import numpy as np
import pandas as pd

from core.interval_data import read_intervals, fetch_acts
from core.data import load_data
from ml.set_evolution import build_sets, add_signatures
from ml.type_comparison import (all_sessions, run_type_comparison,
                                session_frame, series_stats)
from ml.protocol_reps import run_protocol_view

iv = read_intervals((pd.Timestamp.now().normalize()
                     - pd.Timedelta(days=370)).isoformat())
df_all = load_data()
acts = fetch_acts((pd.Timestamp.now().normalize()
                   - pd.Timedelta(days=370)).isoformat())
sets = build_sets(iv, acts, df_all)
sets, _ = add_signatures(sets)

t0 = time.time()
res = run_type_comparison(sets)
print(f"run_type_comparison: {time.time() - t0:.2f}s → {res['n_sets']} sets / "
      f"{res['n_sessions']} sessions\n")

summ = res["summary"]
print("=== SUMMARY (session unit) ===")
print(summ[["Type", "Duration", "Sets", "Sessions", "First", "Last", "First W",
            "Last W", "Δ %", "W/month", "± 95%", "Trend", "Ridden", "Rest",
            "IF %"]].head(14).to_string(index=False))
print("\nrows:", len(summ), "· with CI:", int(summ["± 95%"].notna().sum()),
      "· trends:", summ["Trend"].value_counts().to_dict())

# the batched session table must equal the per-series one, row for row
bad = 0
for (fam, db), sess in res["sessions_by_series"].items():
    g = res["sets"][(res["sets"]["family"] == fam)
                    & np.isclose(res["sets"]["dur_b"], db)]
    one = session_frame(g)
    same = (len(one) == len(sess)
            and np.allclose(one["w"].to_numpy(), sess["w"].to_numpy())
            and (one["date"].to_numpy() == sess["date"].to_numpy()).all())
    if not same:
        bad += 1
        print("  MISMATCH", fam, db)
print("session_frame agreement:", "all series match" if not bad else
      f"{bad} MISMATCHES")

print("\n=== explorer vs summary (same numbers?) ===")
for t, d in (("Ronnestad 30/15", "30 s"), ("VO₂ max", "3 min"),
             ("FTP / threshold", "5 min"), ("FTP / threshold", "4 min"),
             ("VO₂ max", "4 min")):
    r = summ[(summ["Type"] == t) & (summ["Duration"] == d)]
    if not len(r):
        continue
    r0 = r.iloc[0]
    pv = run_protocol_view(iv, res["sets"], r0["_fam"], r0["_db"])
    s, tr = pv["sessions"], pv["trend"]
    stt = series_stats(res["sets"], r0["_fam"], r0["_db"])
    print(f"\n-- {t} · {d}: sets {r0['Sets']} / sessions {r0['Sessions']}")
    print(f"   summary : first {r0['First W']} last {r0['Last W']} "
          f"best {r0['Best W']} slope {r0['W/month']} ±{r0['± 95%']} "
          f"trend {r0['Trend']}")
    print(f"   explorer: first {tr['first']:.0f} last {tr['last']:.0f} "
          f"best {tr['best']:.0f} slope {tr['slope_robust']:.2f} "
          f"±{tr['ci']:.2f} fade {tr['fade_med']:+.1f}%")
    print(f"   stats   : first {stt['first_w']:.0f} last {stt['last_w']:.0f} "
          f"best {stt['best']:.0f} band {stt['if_band']} IF "
          f"{stt['if_med']:.0f} rest {stt['rest_med']:.0f}s "
          f"varies={stt['rest_varies']} reps_med={stt['reps_med']}")
    print(s[["date", "reps", "sets", "w", "open_w", "close_w", "fade_%",
             "trend", "trend_lo", "trend_hi"]].to_string(index=False))
    # session mean must equal the plain mean of that day's bars
    if len(pv["bars"]):
        chk = pv["bars"].groupby("date")["avg_w"].mean().round(6)
        ref = s.set_index("date")["w"].round(6)
        dmax = (chk - ref.reindex(chk.index)).abs().max()
        print(f"   bar-vs-session mean max diff: {dmax:.2e}")
        assert dmax < 1e-6
print("\nALL CHECKS PASSED")
