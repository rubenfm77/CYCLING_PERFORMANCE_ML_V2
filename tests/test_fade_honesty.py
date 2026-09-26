"""What the explorer shows for a SINGLE-EFFORT series (no fade to measure)
vs a multi-rep series — the exact values the KPI cards and callout use."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd

from core.interval_data import read_intervals, fetch_acts
from core.data import load_data
from ml.set_evolution import build_sets, add_signatures
from ml.type_comparison import run_type_comparison, series_stats
from ml.protocol_reps import run_protocol_view
from views.intervals_view import _session_table

iv = read_intervals((pd.Timestamp.now().normalize()
                     - pd.Timedelta(days=370)).isoformat())
df_all = load_data()
acts = fetch_acts((pd.Timestamp.now().normalize()
                   - pd.Timedelta(days=370)).isoformat())
sets, _ = add_signatures(build_sets(iv, acts, df_all))
res = run_type_comparison(sets)
summ = res["summary"]

for want in (("single efforts", "3 min"), ("Ronnestad 30/15", "30 s"),
             ("FTP / threshold", "5 min")):
    row = summ[(summ["Type"] == want[0]) & (summ["Duration"] == want[1])]
    if not len(row):
        print("missing", want)
        continue
    r0 = row.iloc[0]
    pv = run_protocol_view(iv, res["sets"], r0["_fam"], r0["_db"])
    stt = series_stats(res["sets"], r0["_fam"], r0["_db"])
    tr = pv["trend"]
    fade = tr["fade_med"]
    multi = stt.get("reps_med", 1) >= 2
    print(f"\n=== {want[0]} {want[1]} · reps_med={stt['reps_med']} "
          f"sessions={pv['n_sessions']} intervals={pv['n_reps']} "
          f"sets={int(pd.to_numeric(pv['sessions']['sets']).sum())}")
    print(f"  KPI sessions  : {pv['n_sessions']} sessions · {pv['n_reps']} "
          f"individual intervals in "
          f"{int(pd.to_numeric(pv['sessions']['sets']).sum())} sets")
    print(f"  KPI fade      : {('%+.1f %%' % fade) if not np.isnan(fade) else '—'}"
          f"   foot = {'closing (last 2 reps) vs opening (reps 2–3)' if multi else 'single efforts — one rep, nothing to fade between'}")
    s = pv["sessions"]
    print(f"  fade column   : non-null {int(pd.to_numeric(s['fade_%'], errors='coerce').notna().sum())}"
          f" / {len(s)} sessions")
    print(_session_table(s).head(3).to_string(index=False))
    assert (np.isnan(fade)) == (not multi), \
        f"{want}: fade {'shown' if not np.isnan(fade) else 'blank'} vs reps_med {stt['reps_med']}"
    assert len(s) == r0["Sessions"], f"{want}: explorer {len(s)} vs summary {r0['Sessions']} sessions"
    assert abs(float(s['w'].iloc[0]) - r0["First W"]) <= 0.5, "first W disagrees"
    assert abs(float(s['w'].iloc[-1]) - r0["Last W"]) <= 0.5, "last W disagrees"
    assert abs(tr["slope_robust"] - r0["W/month"]) < 0.02, "slope disagrees"
    assert abs(tr["ci"] - r0["± 95%"]) < 0.02, "CI disagrees"

print("\nOK — fade honest, and summary == explorer on every checked series")
