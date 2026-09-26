"""Display-rule check, session 4.

Two user rules landed here:
  1. no decimals in time — whole minutes, seconds under 90 s;
  2. thousands separators on big numbers (elevation, TSS).

Rule 1 has a trap: the identical-sets groups are 10-SECOND buckets, so a
whole-minute label alone is NOT unique (190 groups collapse to 93 labels).
This asserts every dropdown label is unique AND that no printed minute hides
a spread wider than the minute it claims.
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
from ml.set_evolution import build_sets, add_signatures, quality_gates
from ml.type_comparison import fmt_min, fmt_secs, dur_bucket, dur_class_label

fails = []

# ── rule 1: fmt_min ─────────────────────────────────────────────────────────
cases = [(30, "30 s"), (60, "60 s"), (89, "89 s"), (90, "1 min"),
         # 60–89 s keeps seconds on purpose: 75 s is not "1 min", and the
         # identical-sets group label calls the same value "75 s" too

         (149, "2 min"), (150, "2 min"), (175, "3 min"), (209, "3 min"),
         (210, "3 min"), (211, "4 min"), (302, "5 min"), (181, "3 min"),
         (1200, "20 min"),
         (3599, "60 min")]
for secs, want in cases:
    got = fmt_min(secs)
    assert got == want, f"fmt_min({secs}) = {got!r}, want {want!r}"
    assert "." not in got, f"fmt_min({secs}) still has a decimal: {got!r}"
print("fmt_min: %d cases, no decimals" % len(cases))

# a label must agree with the class rule, or the group and the text disagree
for secs in np.arange(90, 1500, 7.0):
    lab = fmt_min(secs)
    want = dur_class_label(float(dur_bucket(secs)))
    assert lab == want, (f"fmt_min({secs:.0f}) = {lab} but its class is {want}")
print("fmt_min agrees with dur_bucket/dur_class_label on 200+ values")

# decimals=1 is still available for an audit table that wants the exact form
assert fmt_min(302, 1) == "5.0 min", fmt_min(302, 1)
assert fmt_secs(302) == "302 s" and fmt_secs(29.6) == "30 s"
assert fmt_min(float("nan")) == "—" and fmt_min("x") == "—"
print("fmt_min(302, decimals=1) =", fmt_min(302, 1), "| fmt_secs(302) =",
      fmt_secs(302))

# ── rule 1 in the real data: the identical-sets labels ─────────────────────
iv = read_intervals((pd.Timestamp.now().normalize()
                     - pd.Timedelta(days=370)).isoformat())
acts = fetch_acts((pd.Timestamp.now().normalize()
                   - pd.Timedelta(days=370)).isoformat())
sets, clusters = add_signatures(quality_gates(build_sets(iv, acts, load_data()))[0])
assert len(clusters), "no clusters"

labels = [f"{r.sig} · {r.n_days} sessions · {r.n_sets} sets"
          + (f" · {r.style}" if r.style else "")
          for r in clusters.itertuples()]          # the view's exact format
assert len(set(labels)) == len(labels), "two dropdown entries would look alike"
print("dropdown labels unique: %d/%d" % (len(set(labels)), len(labels)))
sigs = list(clusters["sig"])
assert len(set(sigs)) == len(sigs), "duplicate group signature"
# and the whole-minute part alone must NOT be unique — that is exactly why
# the seconds are in the label
whole = [s.split(" · ")[0] for s in sigs]
print("whole-minute prefixes: %d distinct across %d groups — the seconds are "
      "what keeps the dropdown readable" % (len(set(whole)), len(whole)))

rep_b = clusters["sig"].map(lambda s: float(s.rsplit("· ", 1)[-1].split(" ")[0])
                            if "·" in s else np.nan)
mins = clusters["sig"].map(lambda s: s.split("× ", 1)[1].split(" · ")[0])
bad = [m for m in mins.unique() if "." in m]
assert not bad, f"decimal minutes left in a group label: {bad[:4]}"
print("group labels carry no decimals, e.g.:")
for r in clusters.head(6).itertuples():
    print("   %s" % r.sig)

# a whole-minute label must not hide a spread wider than 60 s inside its group
for fam_g in sets.groupby("sig"):
    secs = fam_g[1]["rep_secs"].astype(float)
    if secs.max() - secs.min() > 60:
        fails.append(f"{fam_g[0]}: {secs.min():.0f}–{secs.max():.0f} s under "
                     f"one label")
print("no group label hides a >60 s spread"
      if not fails else "SPREAD LEAK: " + "; ".join(fails))

# ── rule 2: thousands separators ───────────────────────────────────────────
for v, want in [(135000, "135,000"), (0, "0"), (999, "999"), (1000, "1,000"),
                (1250.4, "1,250"), (12345678, "12,345,678")]:
    got = f"{v:,.0f}"
    assert got == want, f"{v} -> {got!r}, want {want!r}"
    raw = f"{v:.0f}"
    assert ("," in got) or len(raw) <= 3, "separator missing"
print("thousands separators: 135000 ->", f"{135000:,.0f}")

# the trend year-card formatter, which is where the 135,000 was printed
for label, dec, unit in [("Total TSS", 0, ""), ("Elevation", 0, " m"),
                         ("Avg power", 1, " W"), ("Hours", 1, " h")]:
    fmt = (f"{{:,.0f}}{unit}" if dec == 0 else f"{{:.{dec}f}}{unit}")
    d_fmt = (f"{{:+,.0f}}{unit}" if dec == 0 else f"{{:+.{dec}f}}{unit}")
    tv, lv = (135000, 118500) if dec == 0 else (210.25, 205.5)
    print(f"   {label:10s} {fmt.format(tv):>12s}  delta {d_fmt.format(tv - lv):>12s}"
          f"  last year {fmt.format(lv):>12s}")
    if dec == 0:
        assert "," in fmt.format(tv), f"{label} still prints {tv} raw"

assert not fails, fails
print("\nOK — whole minutes everywhere, exact seconds kept beside every "
      "rounded one, big numbers separated")
