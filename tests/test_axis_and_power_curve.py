"""Power curve built from the file + the x-axis rule, checked on a real render.

Three standing requirements, verified without credentials (local CSV only):

  1. The power curve must DRAW, from the training file alone — no
     `data/power_curve.csv`, no `src/intervals_api.py` (a script this repo has
     never contained). Every point is one observed effort; a duration nobody
     has ridden inside the window is absent, never interpolated.
  2. A bar chart on these pages sits on a DATE axis. The old "one bar per
     rep length" chart put minutes on the x-axis, which the user rejected
     every time it came back.
  3. Nothing user-visible is in Spanish.
"""
import warnings
warnings.filterwarnings("ignore")

from core import components

FIGS = []


def _cap(fig=None, *a, **k):
    if fig is not None:
        FIGS.append(fig)
    return None


components.show = _cap

from core.context import frames            # noqa: E402
from core.data import load_data            # noqa: E402
from core.metrics import headline          # noqa: E402
from ml import interval_watts as iw        # noqa: E402
import views.fitness as vf                 # noqa: E402
import views.interval_watts as viw         # noqa: E402

vf.show = _cap
viw.show = _cap


# ═══════════════════════════════════════════════════════════════════════════
# 1. the power curve
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 74)
print("1. power curve from the file")
print("=" * 74)

df_all = load_data()

pc = iw.power_curve(df_all)
print(f"   {len(pc)} durations: {', '.join(pc['duration'].tolist())}")

assert len(pc) >= 8, f"only {len(pc)} durations — the curve is not usable"
assert pc["watts"].between(1, 2500).all(), "a point carries impossible watts"
assert (pc["n"] >= 1).all(), "a point with no effort behind it"
assert pc["secs"].is_monotonic_increasing, "x (seconds) must be sorted"

# every point must be an OBSERVED effort: the watts at that exact length have
# to exist in measured() at that exact length, same physiology rules included.
M = iw.measured(df_all)
for r in pc.itertuples():
    same = M[M["secs"] == r.secs]
    assert len(same) >= 1, f"{r.secs} s is not an observed duration"
    # the point is a real row at that exact length, and the best one at it:
    # a row with the same length inside the same window would have won.
    assert r.watts == same["w"].max(), (
        f"{r.watts} W at {r.secs} s is not the observed best at that length")

# the window is a real filter: the number of efforts considered per point is
# disclosed and never zero.
print("   n behind each point:", pc["n"].tolist())
print("   best per duration :", [f"{w:.0f} W" for w in pc["watts"]])

# fewer durations than the fit floor is allowed — the LAW needs 4 points, the
# curve does not invent any to reach it.
short = iw.power_curve(df_all.head(0))
assert not len(short) and list(short.columns) == list(pc.columns), (
    "an empty file must give an empty, well-formed curve")

print("   PASS  every point is an observed effort, none invented")


# ═══════════════════════════════════════════════════════════════════════════
# 2 + 3. real render of the two pages that carry these charts
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 74)
print("2. x-axis = time on the rendered pages")
print("=" * 74)

ctx = frames(df_all)
head = headline(ctx)
vf.render(head, ctx)
n_fit = len(FIGS)
viw.render(head, ctx)
print(f"   figures: fitness {n_fit}, + interval watts {len(FIGS) - n_fit}")

titles = [((f.layout.title.text or "") if f.layout.title else "")
          for f in FIGS]

bars = [(f, t) for f, t in zip(FIGS, titles) if "one bar" in t.lower()]
assert bars, "no bar-by-day figure was drawn"
for f, t in bars:
    ax = f.layout.xaxis
    print(f"   bar chart x-axis: {ax.type}")
    assert ax.type == "date", f"x-axis is {ax.type}, expected date"
    for tr in f.data:                       # every trace, not just the first
        for x in (tr.x if tr.x is not None else []):
            assert "-" in str(x), f"bar x value {x!r} is not a date"

old = [t for t in titles if "one bar per rep length" in t.lower()]
assert not old, f"the minutes-on-x chart is back: {old}"
print("   PASS  bar charts sit on a date axis; the rep-length axis is gone")

# a power-duration law chart is drawn, in English
law = [t for t in titles if "law" in t.lower()]
assert law, "no power-duration law figure drawn"
print("   PASS  power-duration law drawn:", law[0][:64])

# the Fitness power curve actually rendered (the old code called out "run
# python src/intervals_api.py" instead of drawing anything)
assert n_fit >= 5, f"Fitness rendered only {n_fit} figures"
pc_titles = [t for t in titles[:n_fit] if "power curve" in t.lower()]
assert pc_titles, "Fitness drew no power curve"
print("   PASS  Fitness power curve:", pc_titles[0][:64])

# ═══════════════════════════════════════════════════════════════════════════
print("=" * 74)
print("3. English only")
print("=" * 74)
span = [t for t in titles if "potencia" in t.lower() or "ley de" in t.lower()]
assert not span, f"Spanish figure titles: {span}"
print("   PASS  no Spanish in any figure title")

print("\nALL CHECKS PASS")
