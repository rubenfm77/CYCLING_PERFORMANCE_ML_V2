"""Offline check: family × duration comparison from the cached intervals."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)

import pandas as pd

from core.interval_data import read_intervals, fetch_acts
from core.data import load_data
from ml.set_evolution import build_sets, add_signatures
from ml.type_comparison import run_type_comparison, series_detail, series_stats

iv = read_intervals((pd.Timestamp.now().normalize()
                     - pd.Timedelta(days=370)).isoformat())
df_all = load_data()
acts = fetch_acts((pd.Timestamp.now().normalize()
                   - pd.Timedelta(days=370)).isoformat())

sets = build_sets(iv, acts, df_all)
sets, _ = add_signatures(sets)
res = run_type_comparison(sets)
assert res["ok"], res.get("reason")
print("sets:", res["n_sets"], "sessions:", res["n_sessions"])
summ = res["summary"]
print("\n=== SUMMARY (multi-rep first, then singles; top 45) ===")
print(summ[["Type", "Duration", "Sets", "Sessions", "First", "Last",
            "First W", "Last W", "Δ W", "Δ %", "Best W", "W/month",
            "Trend", "Rest", "IF %"]].head(45).to_string(index=False))
print("\nrows total:", len(summ), " families:", res["families"])

print("\n=== RONNESTAD detail ===")
ron = summ[summ["Type"] == "Ronnestad 30/15"]
if len(ron):
    r0 = ron.iloc[0]
    print(r0.to_dict())
    fam = r0["_fam"]; db = r0["_db"]
    print(series_detail(res["sets"], fam, db).to_string(index=False))
    st = series_stats(res["sets"], fam, db)
    print("stats:", {k: v for k, v in st.items()
                     if k not in ("set_w", "date")})

print("\n=== BILLAT detail ===")
bil = summ[summ["Type"] == "Billat 30/30"]
if len(bil):
    b0 = bil.iloc[0]
    print(series_detail(res["sets"], b0["_fam"], b0["_db"]).to_string(index=False))

print("\n=== FTP lines (one per duration — never mixed) ===")
for s in res["lines"].get("FTP / threshold sets", []):
    print(f"  {s['label']:>22} | n={s['n']:2} days={s['n_days']:2} "
          f"trend={'yes' if s['trend'] is not None else 'no'}")
print("=== VO2 lines ===")
for s in res["lines"].get("VO₂ max sets", []):
    print(f"  {s['label']:>22} | n={s['n']:2} days={s['n_days']:2} "
          f"trend={'yes' if s['trend'] is not None else 'no'}")
print("families with lines:", {k: len(v) for k, v in res["lines"].items()})
