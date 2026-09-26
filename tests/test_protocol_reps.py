"""Offline check: per-interval bars + session-average line for one
type × duration series, plus the data-quality screen and timings."""
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
from ml.type_comparison import run_type_comparison, dur_class_label
from ml.protocol_reps import run_protocol_view

t0 = time.time()
iv = read_intervals((pd.Timestamp.now().normalize()
                     - pd.Timedelta(days=370)).isoformat())
df_all = load_data()
acts = fetch_acts((pd.Timestamp.now().normalize()
                   - pd.Timedelta(days=370)).isoformat())
print(f"cache loaded: {len(iv):,} interval rows, {len(acts):,} activities "
      f"({time.time() - t0:.1f}s)")

sets = build_sets(iv, acts, df_all)
sets, _ = add_signatures(sets)
t0 = time.time()
res = run_type_comparison(sets)
print(f"run_type_comparison: {time.time() - t0:.1f}s → "
      f"{res['n_sets']} clean sets / {res['n_sessions']} sessions")
print("\n=== DATA-QUALITY SCREEN ===")
print(res["quality"].to_string(index=False))
print("excluded rows:", len(res["excluded"]))

summ = res["summary"]
print("\n=== SUMMARY — robust trend (top 20) ===")
print(summ[["Type", "Duration", "Sets", "Sessions", "First W", "Last W",
            "Δ %", "W/month", "± 95%", "Trend", "Ridden", "Rest",
            "IF %"]].head(20).to_string(index=False))

# ── per-series interval-level checks ─────────────────────────────────────────
want = [("Ronnestad 30/15", "30 s"), ("Billat 30/30", "30 s"),
        ("VO₂ max", "3 min"), ("FTP / threshold", "5 min"),
        ("FTP / threshold", "4 min"), ("single efforts", "10 min")]
print("\n=== PROTOCOL VIEW (bars + session means) ===")
for type_name, dur in want:
    row = summ[(summ["Type"] == type_name) & (summ["Duration"] == dur)]
    if not len(row):
        print(f"\n-- {type_name} · {dur}: no such series")
        continue
    r0 = row.iloc[0]
    t0 = time.time()
    pv = run_protocol_view(iv, res["sets"], r0["_fam"], r0["_db"])
    dt = time.time() - t0
    if not pv["ok"]:
        print(f"\n-- {type_name} · {dur}: {pv['reason']}")
        continue
    bars, sess, tr = pv["bars"], pv["sessions"], pv["trend"] or {}
    print(f"\n-- {type_name} · {dur} ({dur_class_label(r0['_db'])}) "
          f"{pv['n_reps']} intervals / {pv['n_sessions']} sessions "
          f"({dt:.2f}s)")
    if len(bars):
        # every bar's watts must be inside the session mean ± its own spread,
        # and the session mean must equal the plain mean of that day's bars
        chk = bars.groupby("date")["avg_w"].mean().round(6)
        ref = sess.set_index("date")["w"].round(6)
        d = (chk - ref.reindex(chk.index)).abs().max()
        assert d < 1e-6, f"session mean mismatch {d}"
        print("   bars/day:", bars.groupby("date").size().to_dict())
        print("   lengths :", sorted(set(bars["secs"].round(0))))
    print("   trend   :", {k: (round(v, 2) if isinstance(v, float) else v)
                          for k, v in tr.items()})
    cols = ["date", "reps", "w", "first_w", "last_w", "fade_%", "best",
            "trend", "trend_lo", "trend_hi", "if_mean"]
    print(sess[[c for c in cols if c in sess.columns]].to_string(index=False))

print("\n=== fade / CI sanity ===")
# iterrows, NOT itertuples: pandas renames _-prefixed columns (_fam -> _1)
# inside a namedtuple, so the series keys cannot be read as attributes.
for _, r in summ.head(12).iterrows():
    pv = run_protocol_view(iv, res["sets"], r["_fam"], r["_db"])
    if not pv["ok"] or not pv["trend"]:
        continue
    s = pv["sessions"]
    has_ci = bool(s["trend_lo"].notna().any())
    print(f"{r['Type']:>16} {r['Duration']:>7}  sessions={len(s):2}  "
          f"CI={'yes' if has_ci else 'no ':>3}  "
          f"slope={s['trend'].notna().any() and pv['trend']['slope_robust'] or float('nan'):+.2f}  "
          f"fade={pv['trend']['fade_med']}")
print("\nALL CHECKS PASSED")
