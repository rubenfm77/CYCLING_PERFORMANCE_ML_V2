"""Power curve built from the file + the x-axis rule, checked on a real render.

Three standing requirements, verified without credentials (local CSV only):

  1. The power curve must DRAW, from the training file alone — no
     `data/power_curve.csv`, no `src/intervals_api.py` (a script this repo has
     never contained). Every point is one observed effort; a duration nobody
     has ridden inside the window is absent, never interpolated.
  2. A bar chart on these pages sits on a DATE axis. The old "one bar per
     rep length" chart put minutes on the x-axis, which the user rejected
     every time it came back. The timeline charts that were on the interval
     watts page have since been given to Evolution, and this file now checks
     that split too: the Power law page draws watts against duration and
     nothing with time on an axis, and the coach's prescribed targets render
     on Evolution.
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

# The axis reads in MINUTES — the standing request. `4824 s` on a tick is
# what the athlete rejected; the exact length stays on the hover instead.
print("   axis ticks :", iw.axis_labels(pc["secs"]))
labs = iw.axis_labels(pc["secs"])
assert len(labs) == len(pc), "the axis lost a point"
assert len(set(labs)) == len(labs), f"two ticks read the same word: {labs}"
for s, lab in zip(pc["secs"], labs):
    if s < 60:
        # the athlete's rule: seconds below a minute, minutes from there up
        assert lab.endswith(" s"), (s, lab)
    else:
        assert lab.endswith("min") or ":" in lab, (
            f"{s} s is still printed in raw seconds as {lab!r}")
assert not [L for L in labs if L.endswith("s") and L[:-1].isdigit()
            and float(L[:-1]) >= 60], f"raw seconds above a minute: {labs}"
# minutes round halves DOWN, the rule dur_bucket/_dur_label file classes by
import math
for s, lab in zip(pc["secs"], labs):
    if lab.endswith("min"):
        k = int(lab.split()[0])
        assert k == math.ceil(s / 60.0 - 0.5), (s, lab)
print("   PASS  x-axis ticks read in minutes, halves down, none alike")


# ═══════════════════════════════════════════════════════════════════════════
# 2 + 3. real render of the three pages that carry these charts
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 74)
print("2. x-axis = time on the rendered pages")
print("=" * 74)

ctx = frames(df_all)
head = headline(ctx)
vf.render(head, ctx)
n_fit = len(FIGS)
viw.render(head, ctx)                 # the Power law page
n_law = len(FIGS)
import views.evolution as ve                       # noqa: E402
ve.show = _cap
ve.render(head, ctx)                 # which now also carries prescriptions
print(f"   figures: fitness {n_fit}, power law {n_law - n_fit}, "
      f"evolution {len(FIGS) - n_law}")

titles = [((f.layout.title.text or "") if f.layout.title else "")
          for f in FIGS]

# ── the page split: the Power law page gives its timelines to Evolution ───
gave_up = [t for t in titles[n_fit:n_law]
           if "one bar" in t.lower() or "over time" in t.lower()]
assert not gave_up, f"the Power law page drew a timeline again: {gave_up}"
print("   PASS  Power law page draws no timeline — watts against duration only")

pres_here = [t for t in titles[n_fit:n_law] if "prescribed" in t.lower()]
assert not pres_here, f"the Power law page still draws prescriptions: {pres_here}"
pres_evo = [t for t in titles[n_law:] if "prescribed" in t.lower()]
assert pres_evo, "Evolution drew no prescribed-target figure"
print("   PASS  prescribed coach targets render on Evolution:", pres_evo[0][:62])

# no chart anywhere may be the artefact that was rejected: bars across rep
# lengths, i.e. minutes on a bar's x-axis
old = [t for t in titles if "one bar per rep length" in t.lower()]
assert not old, f"the minutes-on-x chart is back: {old}"

# the bars that ARE timelines sit on a date (or year) axis, every trace of them
bars = [(f, t) for f, t in zip(FIGS[n_law:], titles[n_law:])
        if any(getattr(tr, "type", None) == "bar" for tr in f.data)]
assert bars, "Evolution drew no bar-by-day figure"
for f, t in bars:
    ax = f.layout.xaxis
    xs = [x for tr in f.data if tr.x is not None for x in tr.x]
    assert xs, f"{t!r}: a bar figure with no x values"
    print(f"   bar chart x-axis: {ax.type or 'linear'}  e.g. {xs[0]!r}")
    if ax.type in ("date", "category"):
        for x in xs:                   # every trace, not just the first
            assert "-" in str(x), f"{t!r}: bar x value {x!r} is not a date"
    else:
        # a numeric timeline is a year axis — still time, never rep length
        try:
            ys = [float(x) for x in xs]
        except (TypeError, ValueError):
            raise AssertionError(
                f"{t!r}: bar x values {xs[:5]!r} are not a time axis")
        assert all(1990 <= y <= 2100 for y in ys), (t, ys[:5])
print("   PASS  bar charts sit on a date or year axis; the rep-length axis "
      "is gone")

# a power-duration law chart is drawn, in English, on the page named after it
law = [t for t in titles[n_fit:n_law] if "law" in t.lower()]
assert law, "no power-duration law figure drawn on the Power law page"
print("   PASS  power-duration law drawn:", law[0][:64])

# the Fitness power curve actually rendered (the old code called out "run
# python src/intervals_api.py" instead of drawing anything)
assert n_fit >= 5, f"Fitness rendered only {n_fit} figures"
pc_titles = [t for t in titles[:n_fit] if "power curve" in t.lower()]
assert pc_titles, "Fitness drew no power curve"
print("   PASS  Fitness power curve:", pc_titles[0][:64])

# …and its x-axis ticks are the minute labels, not the raw seconds.
_pc_fig = FIGS[titles.index(pc_titles[0])]
_ticks = [str(t) for t in (_pc_fig.layout.xaxis.ticktext or [])]
assert _ticks, "the power curve has no tick labels"
assert _ticks == iw.axis_labels(pc["secs"]), (_ticks, iw.axis_labels(pc["secs"]))
assert not [t for t in _ticks if t.endswith("s") and t[:-1].isdigit()
            and float(t[:-1]) >= 60], f"seconds back on the axis: {_ticks}"
print(f"   PASS  power-curve ticks in minutes: {', '.join(_ticks)}")

# …and the power-duration law on the Power law page answers to the same rule
# (it used to print `duration (s)` and a hover of `880 s`).
_law_titles = [t for t in titles[n_fit:n_law] if "power-duration law" in t.lower()]
if _law_titles:
    _lf = FIGS[titles.index(_law_titles[0])]
    _lt = [str(t) for t in (_lf.layout.xaxis.ticktext or [])]
    assert _lt, "the power-duration law has no tick labels"
    assert len(set(_lt)) == len(_lt), f"two law ticks read alike: {_lt}"
    assert not [t for t in _lt if t.endswith("s") and t[:-1].isdigit()
                and float(t[:-1]) >= 60], f"law axis back in raw seconds: {_lt}"
    assert str(_lf.layout.xaxis.title.text or "") != "duration (s)", \
        "the law axis is seconds again"
    print(f"   PASS  law ticks follow the rule: {', '.join(_lt)}")

# ═══════════════════════════════════════════════════════════════════════════
print("=" * 74)
print("3. English only")
print("=" * 74)
span = [t for t in titles if "potencia" in t.lower() or "ley de" in t.lower()]
assert not span, f"Spanish figure titles: {span}"
print("   PASS  no Spanish in any figure title")

print("\nALL CHECKS PASS")
