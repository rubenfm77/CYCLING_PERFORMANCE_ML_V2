import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd

from core.data import load_data
from core.interval_data import fetch_acts, read_intervals, sync_intervals

now = pd.Timestamp.now().normalize()
sync_iso = (now - pd.Timedelta(days=370)).isoformat()

print("syncing (schema v2: full re-pull for group_id) ...")
meta = sync_intervals(sync_iso, on_progress=lambda d, t: print(
    f"\r  {d}/{t}   ", end="", flush=True) if d % 15 == 0 or d == t else None)
print("\nmeta:", meta)

iv = read_intervals(sync_iso)
cov = iv["group_id"].notna().mean() if "group_id" in iv.columns else -1
print(f"rows={len(iv)} group_id coverage={cov:.0%}")

acts = fetch_acts(sync_iso)
temp = pd.to_numeric(acts["temp"], errors="coerce")
print(f"temp coverage: {temp.notna().sum()}/{len(acts)} "
      f"min={temp.min()} max={temp.max()}")

df_all = load_data()
print("df_all:", len(df_all), "cols:",
      all(c in df_all.columns for c in ("date", "ctl", "atl", "tss")))

from ml.set_evolution import add_signatures, build_sets, run_duration_evolution, run_set_evolution

sets = build_sets(iv, acts, df_all)
print(f"\nsets={len(sets)}")
for c in ("rest", "temp", "tsb", "tss_7", "set_hr", "decoupling"):
    if c in sets:
        print(f"  {c}: {pd.to_numeric(sets[c], errors='coerce').notna().mean():.0%} non-null")

s, clusters = add_signatures(sets)
print("\ntop clusters:")
print(clusters.head(16)[["sig", "n_days", "n_sets", "rep_secs", "reps",
                         "rest", "style", "best"]].to_string())

ron = clusters[clusters["style"].str.contains("Ronnestad", na=False)]
bil = clusters[clusters["style"].str.contains("Billat", na=False)]
ftp = clusters[clusters["style"].str.contains("FTP", na=False)]
print(f"\nRonnestad-like clusters: {len(ron)} | Billat-like: {len(bil)} | FTP: {len(ftp)}")

top = clusters.iloc[0]["sig"]
res = run_set_evolution(s[s["sig"] == top])
print(f"\n[top] {top}: ok={res.get('ok')} n={res.get('n')} "
      f"slope={res.get('slope_m'):+.1f} W/mo r={res.get('r_time'):+.2f} "
      f"rho={res.get('rho'):+.2f}")
if res.get("ok"):
    print(f"  assoc rows={len(res['assoc'])}")
    if len(res["assoc"]):
        print(res["assoc"].to_string())
    f = res["fatigue"]
    print("  fatigue:", "none" if not f else
          f"fresh n={f['fresh_n']} med={f['fresh_med']:.0f}W / "
          f"tired n={f['tired_n']} med={f['tired_med']:.0f}W")
    print("  d_best:", res.get("d_best"), "| last:", res.get("last"))

if len(ron):
    sig = ron.iloc[0]["sig"]
    r2 = run_set_evolution(s[s["sig"] == sig])
    print(f"\n[ronnestad] {sig}: ok={r2.get('ok')} n={r2.get('n')} "
          f"days={r2.get('n_days')} slope={r2.get('slope_m'):+.1f} W/mo")
if len(bil):
    sig = bil.iloc[0]["sig"]
    r3 = run_set_evolution(s[s["sig"] == sig])
    print(f"[billat] {sig}: ok={r3.get('ok')} n={r3.get('n')} "
          f"days={r3.get('n_days')}")

d = run_duration_evolution(iv)
print(f"\nduration evolution: ok={d['ok']} months={len(d['months'])} "
      f"labels={d['labels']}")
print(d["table"].to_string())
print("z NaN frac:", round(float(np.isnan(d["z"]).mean()), 2),
      "| text row 0:", d["text"][0][:6])

# view-level smoke: build the exact objects the view touches
from views import intervals_view
print("\nview import OK; functions:",
      all(hasattr(intervals_view, f) for f in
          ("_render_compare", "_render_sets", "_render_duration_evolution",
           "_render_window_search", "_matched_window")))
