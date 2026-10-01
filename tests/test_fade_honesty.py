"""What the explorer shows for a SINGLE-EFFORT series (no fade to measure)
vs a multi-rep series — the exact values the KPI cards and callout use.

Two notes on what this file is allowed to assume.

The multi-rep example is BILLAT rather than "Ronnestad 30/15". Both are 30 s
reps, but a 30 s set from a session the athlete filed under BILLAT is now a
BILLAT set — see `ml/set_evolution._style` — so the five labelled sessions moved
and "Ronnestad 30/15" is down to the one unlabelled session. A hardcoded example
that the data has legitimately moved under is a test that will break for the
wrong reason, and it is checked against the summary below rather than assumed.

`agree()` treats two blanks as agreeing. It has to: a one-session series
correctly has no trend and no CI, and `abs(nan - nan) < 0.02` is False, so the
naive comparison reports a disagreement between two figures that are both
correctly absent. One blank against one number IS a disagreement, and that is
the case worth catching.
"""
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


def agree(a, b, tol=0.02):
    """Two numbers agree, or two blanks agree. One of each does not."""
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    if np.isnan(a) != np.isnan(b):
        return False
    return abs(a - b) < tol


iv = read_intervals((pd.Timestamp.now().normalize()
                     - pd.Timedelta(days=370)).isoformat())
df_all = load_data()
acts = fetch_acts((pd.Timestamp.now().normalize()
                   - pd.Timedelta(days=370)).isoformat())
sets, _ = add_signatures(build_sets(iv, acts, df_all))
res = run_type_comparison(sets)
summ = res["summary"]

for want in (("single efforts", "3 min"), ("BILLAT", "30 s"),
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
    assert agree(tr["slope_robust"], r0["W/month"]), "slope disagrees"
    assert agree(tr["ci"], r0["± 95%"]), "CI disagrees"
    # At least one checked series must have a real trend, or the two agreement
    # asserts above could be passing on two blanks and testing nothing.
    if want[0] == "BILLAT":
        assert not np.isnan(r0["W/month"]), "BILLAT should carry a trend"
        assert not np.isnan(r0["± 95%"]), "BILLAT should carry a CI"

# A series too thin to support a trend must not grow one. "Ronnestad 30/15" is
# the case: the athlete's five labelled 30 s sets are BILLAT now, so this family
# holds the single unlabelled session left, and one point is not a trend.
print("\n=== a one-session series stays blank, in the summary AND the explorer ===")
thin = summ[(summ["Type"] == "Ronnestad 30/15") & (summ["Duration"] == "30 s")]
assert len(thin) == 1, f"expected the thin family to survive, got {len(thin)} row(s)"
q = thin.iloc[0]
assert int(q["Sessions"]) == 1, f"expected 1 session, got {q['Sessions']}"
assert np.isnan(q["W/month"]), f"a one-session series must show no trend, got {q['W/month']}"
assert np.isnan(q["± 95%"]), f"a one-session series must show no CI, got {q['± 95%']}"
assert str(q["Trend"]).strip() in ("·", "-", "—", "n/a"), \
    f"the Trend cell should say there is no trend, not imply a line: {q['Trend']!r}"
pv_t = run_protocol_view(iv, res["sets"], q["_fam"], q["_db"])
tr_t = pv_t["trend"]
assert np.isnan(tr_t["slope_robust"]), "explorer grew a slope from one point"
assert np.isnan(tr_t["ci"]), "explorer grew a CI from one point"
print("  1 session → no W/month, no CI, Trend '·', and the explorer agrees")

print("\nOK — fade honest, summary == explorer, and a thin series stays thin")
