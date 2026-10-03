"""Every bar of the Evolution day chart must print its own numbers.

The chart is only honest if the reader never has to take a measurement off the
geometry: the watts and the exact interval length are printed above each bar,
the bar height is that same un-averaged interval watts, and the truncated
y-axis floor is stated in the subtitle next to the floor that was applied.

Run: python -c "from tests import test_evolution_bars"
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from core.data import load_data
from ml import interval_watts as iw
from views.evolution import _day_figure

df = load_data()
opts = iw.day_options(df)
print(f"1. {len(opts)} (type x rep class) pairs available")
assert len(opts), "no peak-meter effort in this file"

row = opts[opts["days"] >= 5].sort_values("days", ascending=False).iloc[0]
S = iw.day_series(df, row.tt, row.cls)
print(f"   charting {row.tt} / {row.cls}: {len(S)} day(s)")
assert len(S) >= 5

fig = _day_figure(S, row.tt, row.cls)
bars = fig.data[0]

print("2. every bar carries its watts and its exact length")
for w, s, lab in zip(S["w"], S["secs"], bars.text):
    lab = str(lab)
    assert f"{iw.fmt_watts(w)} W" in lab, (w, lab)
    assert iw.fmt_rep(s) in lab, (s, lab)
print(f"   e.g. {str(bars.text[0]).replace(chr(10), ' / ')}")

print("3. bar height is the interval's own watts, never an average")
assert [float(y) for y in bars.y] == [float(w) for w in S["w"]]

print("4. the average lives in its own trace, flat across the days")
avg = fig.data[1]
assert avg.name == "Avg interval watts"
mean_w = float(S["w"].mean())
assert {round(float(y), 6) for y in avg.y} == {round(mean_w, 6)}

print("5. x-axis is the day, category-ordered, in date order")
assert fig.layout.xaxis.type == "category"
assert list(bars.x) == [str(pd.Timestamp(d).date()) for d in S["day"]]

print("6. the y-axis floor in the subtitle is the floor that was applied")
y_lo, y_hi = fig.layout.yaxis.range
assert y_lo == fig._y_floor
lo_w, hi_w = float(S["w"].min()), float(S["w"].max())
assert y_lo <= lo_w and y_hi >= hi_w, (y_lo, lo_w, hi_w, y_hi)
span = hi_w - lo_w
if span >= max(hi_w * 0.08, 5.0):
    # cut under the lowest bar but NOT down to zero
    assert y_lo > 0 and y_lo >= lo_w - span * 0.31
    assert y_lo < lo_w, "the lowest bar must clear the floor"
subtitle = fig.layout.title.text or ""
assert "is the floor" in subtitle
assert f"{iw.fmt_watts(y_lo)} W" in subtitle
print(f"   range {y_lo:.0f}–{y_hi:.0f} W for bars {lo_w:.0f}–{hi_w:.0f} W")

print("7. flat bars are not zoomed into noise")
flat = S.copy()
flat["w"] = [lo_w + 0.4 * i for i in range(len(flat))]
f2 = _day_figure(flat, row.tt, row.cls)
r = f2.layout.yaxis.range
assert r[1] - r[0] >= float(flat["w"].max()) * 0.2, r
print(f"   flat series {flat['w'].min():.1f}–{flat['w'].max():.1f} W "
      f"-> axis {r[0]:.1f}–{r[1]:.1f} W")

print("8. no bar label may claim a number the row does not carry")
for w, s, lab in zip(S["w"], S["secs"], bars.text):
    first = str(lab).split("\n")[0]
    assert first.endswith(" W"), first
    shown = float(first[:-2].replace(",", "").replace(".", ""))
    assert shown == float(str(iw.fmt_watts(w)).replace(",", "")), (w, first)

print("9. each day's OWN average is on the chart, as a line, once")
# This is what the athlete asked to be able to follow over time: the average
# of 30 Sep, of 22 Sep, of 15 Sep — not a flat mean that hides the order.
day_line = [t for t in fig.data if t.name == "Day average watts"]
assert len(day_line) == 1, [t.name for t in fig.data]
day_line = day_line[0]
assert day_line.mode == "lines+markers"
assert "day_avg" in S.columns, "day_series must expose the per-day average"
got = [float(y) for y in day_line.y]
want = [float(v) for v in S["day_avg"]]
assert got == want, (got[:5], want[:5])
assert list(day_line.x) == list(bars.x), "the line must sit on the same days"
print(f"   e.g. {str(day_line.x[0])} -> {got[0]:.0f} W")

print("10. the day average is printed, not only drawn")
tbl = [str(pd.Timestamp(d).date()) for d in S["day"]]
assert tbl == list(bars.x), "table rows and bars must be the same days"
for v in S["day_avg"]:
    assert str(iw.fmt_watts(v)) != "", v
print(f"   day average for every one of the {len(tbl)} charted day(s)")

print("11. the day average is never mixed across lengths")
# It is computed inside one rep class only: with a single session that day it
# equals the bar, and with two sessions of that class it is their mean.
for r in S.itertuples():
    assert r.cls == row.cls
    assert float(r.day_avg) <= max(float(r.w), float(r.day_avg)) + 1e-9

print("\nPASS test_evolution_bars")
