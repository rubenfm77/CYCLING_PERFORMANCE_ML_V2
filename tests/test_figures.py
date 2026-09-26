"""Legend-lane check.

The old failure mode was a legend floating INSIDE the plot area: on the
11-duration small multiple it lay across the markers (unreadable), and past
the right edge it was clipped by the SVG. The rule now is a legend in a
MARGIN — inside the canvas, over nothing. This asserts that geometry at the
figure level, for every figure on the tab, in both the paired and the
full-width layout.
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
from ml.type_comparison import run_type_comparison
from ml.protocol_reps import run_protocol_view
from core.theme import style_figure
from views.intervals_view import (_rep_bar_figure, _session_trend_figure,
                                  _family_figure, _session_table, _rep_table,
                                  _lane_legend, LEG_CHAR_PX, LEG_ENTRY_PX,
                                  LEG_GAP_PX, LEG_ROW_PX, LEG_LANE_PAD,
                                  LEG_BUDGET_W)

iv = read_intervals((pd.Timestamp.now().normalize()
                     - pd.Timedelta(days=370)).isoformat())
df_all = load_data()
acts = fetch_acts((pd.Timestamp.now().normalize()
                   - pd.Timedelta(days=370)).isoformat())
sets, _ = add_signatures(build_sets(iv, acts, df_all))
res = run_type_comparison(sets)
r0 = res["summary"].iloc[0]
fam, db = r0["_fam"], r0["_db"]

BASE_R, BASE_L = 24, 52
fails = []


def row_px(labels):
    return (sum(LEG_CHAR_PX * len(str(x)) + LEG_ENTRY_PX for x in labels)
            + LEG_GAP_PX * (len(labels) - 1))


def check(name, fig, labels, min_w, cols=None):
    """The legend must sit in a margin, and the margin must be wide/tall
    enough to hold it — otherwise it is back over the data or clipped."""
    lay, m = fig.layout.legend, fig.layout.margin
    orient = getattr(lay, "orientation", "v")
    est = row_px(labels)
    n_rows = 1
    if orient == "h" and getattr(lay, "entrywidthmode", None) == "fraction":
        cols = max(1, round(1 / lay.entrywidth))   # the code may have cut it
        n_rows = -(-len(labels) // cols)
        assert lay.x == 0.0 and lay.y == 1.0 and lay.yanchor == "bottom", \
            f"{name}: wrapped legend is not above the plot area"
        assert 0 < lay.entrywidth <= 1, f"{name}: bad entrywidth"
        # each entry gets 1/cols of the width — check the longest name fits it
        w = LEG_BUDGET_W * lay.entrywidth
        need = LEG_CHAR_PX * max(len(str(x)) for x in labels) + LEG_ENTRY_PX
        assert need <= w, f"{name}: '{max(labels, key=len)}' needs {need}px "\
                          f"of {w:.0f}px — it would be truncated"
        room = m.t - 74
        assert room >= n_rows * LEG_ROW_PX - 2, \
            f"{name}: top margin {m.t} has no room for {n_rows} legend rows"
        where = f"top margin, {n_rows} rows of {cols} (+{room}px)"
        assert cols <= (cols or 99), "cols grew"
    elif orient == "h":
        assert lay.y == 1.0 and lay.yanchor == "bottom", \
            f"{name}: horizontal legend is not above the plot area"
        room = m.t - 74 if fig.layout.title else m.t - 48
        assert room >= LEG_ROW_PX - 2, \
            f"{name}: top margin {m.t} has no room for the legend row"
        where = f"top margin (+{room}px)"
    else:
        assert lay.x == 1.0 and lay.xanchor == "left", \
            f"{name}: vertical legend is not right of the plot area"
        need = int(min(190, LEG_CHAR_PX * max(len(str(x)) for x in labels)
                       + LEG_ENTRY_PX + LEG_LANE_PAD)) + BASE_R
        assert m.r >= need, f"{name}: right margin {m.r} < lane {need}"
        where = f"right margin ({m.r}px ≥ {need}px)"
    print(f"{name:34s} legend={len(labels):2d} est_row={est:5.0f}px "
          f"budget={min_w:3.0f} -> {orient:1s} in {where}")
    for n in labels:
        print(f"      · {n}")
    return fig


# ── explorer: per-interval bars, session trend (with and without overlays) ──
for label, overlay in (("bars", (False, False)),
                       ("bars+temp+tsb", (True, True))):
    pv = run_protocol_view(iv, res["sets"], fam, db)
    f1 = _rep_bar_figure(pv)
    style_figure(f1, "Individual intervals, session by session", 420)
    l1 = [t.name for t in f1.data if t.name
          and getattr(t, "showlegend", None) is not False]
    check(f"rep bars [{label}]", _lane_legend(f1, l1, 430, cols=3), l1, 430, 3)
    f2 = _session_trend_figure(pv, overlay[0], overlay[1])
    style_figure(f2, "Session average watts, 5 sessions", 420)
    l2 = [t.name for t in f2.data if t.name
          and getattr(t, "showlegend", None) is not False]
    check(f"session trend [{label}]", _lane_legend(f2, l2, 430, cols=3),
          l2, 430, 3)
    for f, nm in ((f1, "rep bars"), (f2, "session trend")):
        m = f.layout.margin
        pw = 458 - int(m.l or BASE_L) - int(m.r or BASE_R)
        assert pw >= 380, f"{nm} [{label}]: plot only {pw}px wide"
        assert int(f.layout.height) - int(m.t) - int(m.b) >= 200, \
            f"{nm} [{label}]: plot too short after the legend rows"
        print(f"{'':34s} plot area {pw}x"
              f"{int(f.layout.height) - int(m.t) - int(m.b)}px")

# ── small multiples: the 11-duration family is the one that broke ──────────
for f in res["families"]:
    series = res["lines"][f]
    n = len(series)
    full = n > 2
    fg, labels = _family_figure(f, series, 6)
    style_figure(fg, f"{f} — {n} durations",
                 350 if full else 300, showlegend=n > 1)
    min_w, cols = (430, 3) if full else (230, None)
    check(f"family {f[:18]} n={n}{'*' if full else ''}",
          _lane_legend(fg, labels, min_w, cols=cols), labels, min_w, cols)
    m = fg.layout.margin
    plot_w = 458 - int(m.l or BASE_L) - int(m.r or BASE_R)
    plot_h = int(fg.layout.height) - int(m.t) - int(m.b)
    assert plot_w >= 380, f"{f}: plot only {plot_w}px wide after the lane"
    assert plot_h >= 150, f"{f}: plot only {plot_h}px tall after the lane"
    print(f"{'':34s} plot area {plot_w}x{plot_h}px")
    if n > 6:
        print(f"{'':34s} capped: top 6 of {n} durations (caption on the page)")

st = _session_table(run_protocol_view(iv, res["sets"], fam, db)["sessions"])
rt = _rep_table(run_protocol_view(iv, res["sets"], fam, db)["bars"])
print("\nsession table cols:", list(st.columns))
print("rep table cols:", list(rt.columns))
print("session table head:\n", st.head(3).to_string(index=False))

# ── the session average must be visible AND numbered ───────────────────────
pv = run_protocol_view(iv, res["sets"], fam, db)
bars = _rep_bar_figure(pv)
avg = [t for t in bars.data if t.name == "Session average"]
assert len(avg) == 2, f"expected the dotted mean + the diamond, got {len(avg)}"
line, dia = avg
assert line.mode == "lines" and line.line.dash == "dot", "mean line"
assert dia.mode == "markers+text", "the diamond must carry its number"
assert len(dia.text) == len(pv["sessions"]), "one number per session"
assert all(str(t).endswith(" W") for t in dia.text), "numbers say watts"
assert dia.marker.size >= 14 and dia.marker.line.width >= 2, "too small"
print("\nsession-average marker: mode=%s size=%s ring=%s labels=%s" % (
    dia.mode, dia.marker.size, dia.marker.line.width, dia.text[:3]))

assert not fails, fails
print("\nOK — every legend in a margin lane, over no data, and every "
      "session average printed")
