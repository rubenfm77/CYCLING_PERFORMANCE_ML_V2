# ml/type_comparison.py — training-type comparison, duration-matched. (NEW FILE)
"""
User rule: a training type is only ever compared with ITSELF at the SAME
duration. An 8-minute threshold effort and a 20-minute one are different
series — never averaged, never trended together, never in one row.

This module takes the SETS table (ml/set_evolution.build_sets, already
grouped by intervals.icu's group_id) and:

  * assigns each set a family: FIRST the athlete's own training type
    from the session label (FTP, VO2MAX, BILLAT, AEROBIC BASE …) — the
    charts are grouped by training type, never guessed from rep length.
    Only sessions with no label fall back to the heuristic protocol name
    from _style() (Ronnestad 30/15, Billat 30/30, micro-reps, VO₂ sets,
    FTP/threshold sets, sprints, long efforts) or "single efforts" when
    the effort was not grouped into a set;
  * assigns a DURATION class = the length of ONE interval (a rep) on a
    whole-MINUTE grid with halves rounding DOWN (≤ 3.5 min → "3 min",
    > 3.5 → "4 min"; sub-minute protocols keep seconds) — the detector
    wobbles ±10 s on the same programmed effort, so finer classes would
    split one workout into two fake series, while an 8-min set is never
    averaged with a 20-min one;
  * summarises every (family × duration) SERIES: sets, sessions, first→last
    watts, % change, ROBUST (Theil–Sen) W/month with a bootstrap 95 % CI,
    intensity band and medians — one table row per series, so a conclusion
    can be read directly off the row;
  * runs the data-quality screen (ml.set_evolution.quality_gates) first:
    duplicate rides, non-protocol groupings and detector artefacts are
    excluded, counted and listed — never dropped silently;
  * returns per-series detail (every effort with its date) and per-family
    lines for small-multiple charts — one coloured line per DURATION,
    never one line mixing durations.

Honesty rules:
  * descriptive only — observed history, no causal or predictive claims;
  * n (sets AND sessions) always shown; slopes need ≥ 3 sets and a real
    time span, otherwise left blank ("·") rather than fitted on 2 points;
  * the headline trend is Theil–Sen (robust to one all-out session) with a
    bootstrap 95 % CI; an ↑/↓ arrow is only printed when the CI clears zero
    AND |slope| ≥ 1 W/month — otherwise "→". A CI that straddles zero is
    shown as "→", not dressed up as progress;
  * family labels are the athlete's training types where a label exists,
    otherwise heuristics from rep length + measured rest, always shown
    next to the raw numbers (they are not typed by intervals.icu).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.theme import BLANK_TYPE_TOKENS
from ml.interval_watts import REP_ORDER, rep_class
from src.config import MAIN_TYPES

# Duration CLASSES = the length of ONE interval (a rep), rounded to whole
# minutes with halves going DOWN: up to 3.5 min is a "3 min" effort, more
# than 3.5 min is "4 min", and the same rule applies at every minute
# (2.5–3.5 → 3 min, 3.5–4.5 → 4 min, 4.5–5.5 → 5 min …). The detector
# wobbles ±10 s on the same programmed effort, so finer classes would split
# one workout into two fake series; genuinely different lengths stay apart
# (8 min is never compared with 20 min). Sub-minute protocols keep their
# real-world unit — a 30 s rep IS 30 s, and Ronnestad 30/15 must not
# dissolve into "1 min". Every detail row still shows the exact length.
SUB_MIN_CUT_S = 45.0          # under this, the class is seconds, not minutes
MIN_TREND_N = 3          # points needed before a W/month is shown at all
MIN_TREND_SESSIONS = 2   # ... and at least 2 distinct days
MIN_CI_SESSIONS = 5      # distinct days needed for a bootstrap 95% CI — below
#                         this the resamples collapse onto a handful of dates
#                         and the interval is fiction, so it stays blank

FAMILY_ORDER = [
    # The athlete's own label, and it comes first. The grouping of these
    # charts is the training type the athlete filed the session under —
    # FTP, VO2MAX, BILLAT, AEROBIC BASE … — never a guess made from the
    # rep length. BILLAT leads the list because it is the label whose
    # authority was established first (tests/test_label_authority.py); the
    # rest follow src.config.MAIN_TYPES in their agreed order. The
    # heuristic protocol names below are only the FALLBACK for sessions
    # that carry no label at all.
    "BILLAT",
] + [t for t in MAIN_TYPES if t != "BILLAT"] + [
    "Ronnestad-style (30 s on / 15 s off)",
    "Billat-style (30 s on / 30 s off)",
    "micro-reps",
    "VO₂ max sets",
    "FTP / threshold sets",
    "sprints",
    "long efforts",
    "single efforts",
    "unclassified sets",
]
# Every agreed training type maps to itself — the chart shows the label the
# athlete typed. Heuristic families keep their short form. Built with a
# fallback so a family can never render as NaN.
FAMILY_SHORT = {t: t for t in MAIN_TYPES}
FAMILY_SHORT.update({
    "BILLAT": "BILLAT",
    "Ronnestad-style (30 s on / 15 s off)": "Ronnestad 30/15",
    "Billat-style (30 s on / 30 s off)": "Billat 30/30",
    "micro-reps": "micro-reps",
    "VO₂ max sets": "VO₂ max",
    "FTP / threshold sets": "FTP / threshold",
    "sprints": "sprints",
    "long efforts": "long efforts",
    "single efforts": "single efforts",
    "unclassified sets": "unclassified sets",
})

SUMMARY_COLUMNS = ["Type", "Duration", "Sets", "Sessions", "First", "Last",
                   "First W", "Last W", "Δ W", "Δ %", "Best W", "W/month",
                   "± 95%", "Trend", "Ridden", "Rest", "IF %"]


# ── formatting: whole minutes (sub-minute stays in seconds) ─────────────────
def fmt_min(secs, decimals: int = 0) -> str:
    """Seconds → human duration, in the athlete's rule: SECONDS for an
    interval shorter than a minute, MINUTES from one minute up. A 30 s
    interval is not "0 min" — that is the real-world protocol unit — and a
    75 s one is not printed in seconds either, because it is a minute of
    work. Below ten minutes the print is WHOLE minutes by the halves-DOWN
    class rule dur_bucket uses; from ten minutes up it is the NEAREST 5
    (half up), the same nominal rule — so a printed duration and the group
    it belongs to can never disagree: 2:59 says "3 min" and lands in "3 min",
    20:07 says "20 min" and lands in "20 min", 24:47 says "25 min" and lands
    in "25 min". decimals=1 restores the 1-decimal form for audit tables
    that want the exact measured length."""
    try:
        s = float(secs)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(s):
        return "—"
    if s < 60:
        return f"{s:.0f} s"
    if decimals <= 0:
        if s >= 600:
            return f"{int(np.floor(s / 300.0 + 0.5) * 5):.0f} min"
        return f"{np.ceil(s / 60.0 - 0.5):.0f} min"
    return f"{s / 60:.{decimals}f} min"


def fmt_secs(secs) -> str:
    """The exact measured length in whole seconds — the audit trail that sits
    beside every rounded 'N min' so nothing is lost by the rounding."""
    try:
        s = float(secs)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(s):
        return "—"
    return f"{s:.0f} s"


def fmt_rest(secs) -> str:
    """Recovery: seconds under 90 s, minutes above."""
    try:
        s = float(secs)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(s):
        return "—"
    if s < 90:
        return f"{s:.0f} s"
    return f"{s / 60:.1f} min"


def dur_bucket(secs) -> float:
    """Rep length → duration class in minutes. Whole minutes halves-DOWN
    below ten minutes; the NEAREST 5 minutes (half up) from ten minutes up.

    Below ten minutes the detector's ±10 s wobble is smaller than the minute,
    so whole minutes keep distinct protocols apart (≤ 3.5 min → 3, > 3.5 → 4,
    ≤ 4.5 → 4, > 4.5 → 5 …). From ten minutes up the same wobble was filing
    one 20-minute workout under "20 min", "21 min", "24 min", "25 min",
    "26 min" and "28 min" — six one-session series for one job — so those
    round to the closest nominal (20:07 → 20, 24:47 → 25, 28:22 → 30, an
    exact 22:30 going up to 25). Under 45 s the class stays in seconds
    (0 = ≤ 15 s, 0.5 = 30 s): a 30 s rep is never merged into a minute."""
    try:
        s = float(secs)
    except (TypeError, ValueError):
        return 0.0
    if s < SUB_MIN_CUT_S:
        return 0.0 if s < 15.0 else 0.5
    if s >= 600.0:
        return float(int(np.floor(s / 300.0 + 0.5) * 5))
    return float(np.ceil(s / 60.0 - 0.5))


def dur_class_label(dur_b: float) -> str:
    """Class label: "3 min", "30 s", "≤ 15 s"."""
    b = float(dur_b)
    if b <= 0.0:
        return "≤ 15 s"
    if b < 1.0:
        return f"{b * 60:.0f} s"
    return f"{b:.0f} min"


def duration_options(g: pd.DataFrame) -> dict:
    """{picker label: length class} for ONE training type's detail chart.

    The options are the rep-length classes of `ml.interval_watts.REP_ORDER`
    — the very classes the Evolution day chart and the Fitness class lines
    use — in ascending order, each carrying its own session and set counts.
    Whole-minute durations are deliberately NOT offered here: one 20-minute
    workout read back as 20:03, 20:07 and 21:00 would otherwise be filed
    under three single-session options and read as lost. One criterion, so
    one session list on both pages.

    Nothing is dropped: a class outside REP_ORDER (an unparseable length,
    "unknown") still gets its own entry at the end rather than vanishing
    from the picker.
    """
    if g is None or not len(g) or "dur_cls" not in g.columns:
        return {}
    out = {}
    classes = [c for c in REP_ORDER if (g["dur_cls"] == c).any()]
    classes += sorted(set(g["dur_cls"]) - set(classes))
    for cls in classes:
        sub = g[g["dur_cls"] == cls]
        n_d, n_s = int(sub["date"].nunique()), len(sub)
        out[f"{cls} · {n_d} session{'' if n_d == 1 else 's'} · "
            f"{n_s} set{'' if n_s == 1 else 's'}"] = cls
    return out


# ── family + duration + intensity-band assignment ────────────────────────────
# How the work was Ridden, judged from intervals.icu's intensity factor
# (IF = % of the rider's FTP): a "3-min set" ridden at 85 % IF is a threshold
# block, not a VO₂ set, no matter how the reps are shaped. The family stays a
# statement about the PROTOCOL; the band says how hard it actually was, so a
# mislabelled effort is visible instead of silently averaged in.
IF_VO2 = 105.0            # ≥ 105 % IF — VO₂ / race-pace efforts
IF_THRESHOLD = 88.0       # 88–105 % — FTP / threshold work
# below 88 % — tempo, endurance blocks, easy riding
IF_SHORT_S = 45.0         # under this, intervals.icu's IF is a model
# extrapolation, not a physiological % — reported as unusable
IF_BAND_SHORT = "n/a — IF unreliable below 45 s"
IF_BAND_BROKEN = "n/a — IF model artefact"


def intensity_band(if_pct, secs=None) -> str:
    """How hard the work was ridden, as a band. `secs` = the rep length:
    under 45 s the intensity factor is a duration-model extrapolation, so it
    is reported as unusable rather than as a physiological percentage."""
    if secs is not None:
        try:
            if float(secs) < IF_SHORT_S:
                return IF_BAND_SHORT
        except (TypeError, ValueError):
            pass
    try:
        v = float(if_pct)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(v):
        return "—"
    if v >= IF_VO2:
        return "VO₂ ≥105%"
    if v >= IF_THRESHOLD:
        return "threshold 88–105%"
    return "sub-threshold <88%"


def prep_types(sets: pd.DataFrame) -> pd.DataFrame:
    """Add family, duration class, intensity band and series id to a SETS
    table. Data-quality gates run first (ml.set_evolution.quality_gates) —
    duplicate rides and non-protocol groupings are removed here, unreliable
    intensities are flagged, and the audit is attached for the UI."""
    from ml.set_evolution import quality_gates      # local: avoids a cycle
    clean, report, dropped = quality_gates(sets)
    s = clean
    if not len(s):
        s.attrs["quality"] = report
        s.attrs["excluded"] = dropped
        return s
    s = s.copy()
    s["date"] = pd.to_datetime(s["date"])
    style = s["style"].fillna("") if "style" in s.columns else ""
    reps = s["reps"].astype(int)
    # The family is the ATHLETE'S training type when the session carries one
    # (FTP, VO2MAX, BILLAT, AEROBIC BASE …) — the label the athlete filed the
    # session under, alias-normalised on load. A chart called "Training types
    # over time" must show training types; a 4-min set in a session the
    # athlete filed under VO2MAX is a VO2MAX series, not a "FTP / threshold"
    # heuristic, and it must be able to sit next to every other VO2MAX series
    # and be compared across years. The protocol heuristic from _style() and
    # the single/unclassified fallbacks apply ONLY to sessions with no label
    # (blank, "-", "—", "nan", "None"), where the heuristic is all there is.
    dbt = (s["db_type"].fillna("").astype(str).str.strip()
           if "db_type" in s.columns else pd.Series("", index=s.index))
    s["family"] = [
        (t if t not in BLANK_TYPE_TOKENS else
         (st if st else ("single efforts" if n <= 1 else "unclassified sets")))
        for t, st, n in zip(dbt, style, reps)
    ]
    # Rule A (athlete's rule): an effort under 8 minutes with MORE THAN ONE
    # rep in the session is VO2MAX work. It applies where the athlete left no
    # label — a label always wins — and retires the ambiguous heuristic
    # buckets ("FTP / threshold sets", "VO₂ max sets", "unclassified sets")
    # for 15 s–8 min multi-rep work. Two protocol patterns keep their names:
    # sprints (under 15 s, alactic, not VO2) and the 30 s Billat/micro-rep
    # patterns (a first-class discipline in this file, not generic VO2).
    _rs = pd.to_numeric(s["rep_secs"], errors="coerce")
    _blank = dbt.astype(str).str.strip().isin(BLANK_TYPE_TOKENS)
    _proto = s["family"].isin(["BILLAT", "Billat-style (30 s on / 30 s off)",
                               "micro-reps", "sprints"])
    _rule_a = (_blank & (reps > 1) & _rs.ge(15.0) & _rs.lt(480.0) & ~_proto)
    s.loc[_rule_a, "family"] = "VO2MAX"
    s["dur_b"] = [dur_bucket(x) for x in
                  pd.to_numeric(s["rep_secs"], errors="coerce")]
    s["dur_label"] = [dur_class_label(b) for b in s["dur_b"]]
    # The rep-LENGTH class, the criterion the Evolution day chart and the
    # Fitness class lines draw with (anything under 22:00 is "10-20 min", so
    # a session whose peak meter reads 20:07 is still the athlete's 20-minute
    # work, and a 30:07 still sits in "20-30 min"). `dur_b` above rounds to
    # the nearest 5 minutes from ten minutes up for the comparison table — a
    # 30 s rep is never merged with an 80 s one — but the detail at the
    # bottom of the Intervals page is selected with this class, so both
    # pages chart the same days for the same work instead of the Intervals
    # page losing every session that did not land on a whole minute.
    s["dur_cls"] = [rep_class(x) for x in
                    pd.to_numeric(s["rep_secs"], errors="coerce")]
    flags = (s["q_flag"].fillna("").astype(str)
             if "q_flag" in s.columns else pd.Series("", index=s.index))
    s["if_band"] = [
        IF_BAND_BROKEN if fl else intensity_band(i, r)
        for fl, i, r in zip(flags, s["intensity"], s["rep_secs"])
    ]
    # .fillna(family): a label outside MAIN_TYPES (a type added later) still
    # renders as itself instead of NaN.
    s["series"] = (s["family"].map(FAMILY_SHORT).fillna(s["family"])
                   + " · " + s["dur_label"])
    s.attrs["quality"] = report
    s.attrs["excluded"] = dropped
    return s


# ── one row per (family × duration) series ───────────────────────────────────
def _theil_sen(tt: np.ndarray, yy: np.ndarray):
    """Median of all pairwise slopes (W/day) + its intercept. One wild session
    can move at most half the pair-slopes, so the line barely budges — the
    reason the headline trend here is robust, not least-squares. Vectorised:
    an (n×n) difference matrix, because this runs for every series."""
    n = len(tt)
    if n < 2:
        return np.nan, np.nan
    dt = tt[None, :] - tt[:, None]        # dt[i, j] = t_j − t_i
    dy = yy[None, :] - yy[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = dy / dt
    iu = np.triu_indices(n, 1)
    vals = slopes[iu]
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return np.nan, np.nan
    m = float(np.median(vals))
    return m, float(np.median(yy - m * tt))


def robust_fit(g: pd.DataFrame, boot: int = 400, y_col: str = "set_w") -> dict:
    """Theil–Sen fit of watts vs time + bootstrap 95 % CI, as a dict.

    keys: n, slope_w_month, ci_w_month (half-width), slope_day, intercept,
          x (days since first session), y_hat, ci_lo, ci_hi (band over x,
          or None below MIN_CI_SESSIONS distinct days — with 3 or 4 days the
          resamples collapse onto a handful of dates and the interval is
          fiction, so nothing is claimed).
    With 5–100 sessions per series a single all-out day would drag an OLS
    line around; the median pairwise slope does not, and the CI says plainly
    whether 'improving' is separable from 'noise'.
    """
    out = {"n": 0, "slope_w_month": np.nan, "ci_w_month": np.nan,
           "slope_day": np.nan, "intercept": np.nan, "x": None, "y_hat": None,
           "ci_lo": None, "ci_hi": None}
    if g is None or not len(g):
        return out
    g = g.sort_values("date")
    t = (pd.to_datetime(g["date"]) -
         pd.to_datetime(g["date"]).iloc[0]).dt.total_seconds().to_numpy() / 86400.0
    y = pd.to_numeric(g[y_col], errors="coerce").to_numpy()
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    out["n"] = int(len(t))
    out["x"] = t
    if len(t) < MIN_TREND_N or len(t) < 3 or np.ptp(t) <= 0 or np.ptp(y) <= 0:
        return out
    m, b = _theil_sen(t, y)
    if not np.isfinite(m):
        return out
    out.update(slope_day=m, intercept=b, y_hat=b + m * t,
               slope_w_month=m * 30.44)
    if len(np.unique(t)) >= MIN_CI_SESSIONS:
        # Batched bootstrap: resample sessions, recompute the Sen slope and
        # intercept for each draw, and keep the 2.5/97.5 percentiles. Chunked
        # so the (chunk × n × n) difference matrices stay small. A draw that
        # repeats a date has undefined pair-slopes (0/0) — those pairs are
        # dropped, exactly as the single-draw version did, and a draw needs
        # ≥ 3 distinct dates and a non-degenerate power range to count.
        rng = np.random.default_rng(7)      # fixed seed → same numbers twice
        n, iu = len(t), np.triu_indices(len(t), 1)
        slopes, preds = [], []
        chunk = max(1, min(100, 2_000_000 // max(n * n, 1)))
        done = 0
        while done < boot:
            b = min(chunk, boot - done)
            idx = rng.integers(0, n, size=(b, n))
            tt, yy = t[idx], y[idx]                    # (b, n)
            dt = tt[:, None, :] - tt[:, :, None]       # (b, n, n)
            dy = yy[:, None, :] - yy[:, :, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                S = (dy / dt)[:, iu[0], iu[1]]         # (b, n_pairs)
            finite = np.isfinite(S)
            S = np.where(finite, S, np.nan)
            n_fin = finite.sum(axis=1)
            mm = np.full(b, np.nan)
            has = n_fin > 0
            mm[has] = np.nanmedian(S[has], axis=1)
            ts = np.sort(tt, axis=1)
            distinct = 1 + (np.diff(ts, axis=1) > 0).sum(axis=1)
            ok = (distinct >= 3) & (np.ptp(yy, axis=1) > 0) & np.isfinite(mm)
            if ok.any():
                mm_ok, tt_ok, yy_ok = mm[ok], tt[ok], yy[ok]
                bb = np.median(yy_ok - mm_ok[:, None] * tt_ok, axis=1)
                slopes.append(mm_ok * 30.44)
                preds.append(bb[:, None] + mm_ok[:, None] * t[None, :])
            done += b
        if slopes:
            sl = np.concatenate(slopes)
            if len(sl) >= 50:
                lo, hi = np.percentile(sl, [2.5, 97.5])
                out["ci_w_month"] = float((hi - lo) / 2.0)
                P = np.vstack(preds)
                out["ci_lo"] = np.percentile(P, 2.5, axis=0)
                out["ci_hi"] = np.percentile(P, 97.5, axis=0)
    return out


def robust_slope(g: pd.DataFrame, boot: int = 400):
    """(slope W/month, bootstrap 95 % CI half-width, n) for one frame. Pass a
    SESSION frame (ml.type_comparison.session_frame) for the number the page
    shows; the CI is blank below MIN_CI_SESSIONS distinct days."""
    r = robust_fit(g, boot=boot)
    return r["slope_w_month"], r["ci_w_month"], r["n"]


def session_frame(g: pd.DataFrame) -> pd.DataFrame:
    """Collapse one series' SETS to one row per DAY — the fair unit for any
    comparison: a day with two sets is one session, not two observations.
    Everything time-based (first/last/best/trend) is computed here, so the
    conclusions table and the session-average chart are the same numbers."""
    if g is None or not len(g):
        return pd.DataFrame(columns=["date", "w", "sets", "reps", "if_mean",
                                     "rest", "tsb", "temp"])
    d = g.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    return (d.groupby("date")
             .agg(w=("set_w", "mean"), sets=("set_w", "size"),
                  reps=("reps", "sum"),
                  if_mean=("intensity", "mean"),
                  rest=("rest", "median"),
                  tsb=("tsb", "mean"), temp=("temp", "mean"))
             .sort_values("date").reset_index())


def all_sessions(s: pd.DataFrame) -> dict:
    """{(family, dur_b): session frame} for EVERY series in one aggregation.

    Same numbers as calling session_frame() per series, but the page builds it
    once: 77 tiny pandas groupbys cost ~12 s, one groupby over the whole frame
    costs a few milliseconds."""
    if s is None or not len(s):
        return {}
    d = s.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    agg = (d.groupby(["family", "dur_b", "date"])
             .agg(w=("set_w", "mean"), sets=("set_w", "size"),
                  reps=("reps", "sum"), if_mean=("intensity", "mean"),
                  rest=("rest", "median"), tsb=("tsb", "mean"),
                  temp=("temp", "mean"))
             .reset_index()
             .sort_values(["family", "dur_b", "date"]))
    out = {}
    for (fam, db), g in agg.groupby(["family", "dur_b"], sort=False):
        out[(fam, float(db))] = g.reset_index(drop=True)
    return out


def _ols_line(dates: np.ndarray, y: np.ndarray):
    """OLS fit as (dates, y_hat) — the plain least-squares reference the
    robust fit is compared against. None when there is no real time span."""
    t = (pd.to_datetime(dates) - pd.to_datetime(dates)[0]
         ).days.astype(float).to_numpy()
    if len(t) < MIN_TREND_N or np.ptp(t) <= 0 or np.ptp(y) <= 0:
        return None
    b, a = np.polyfit(t, y, 1)
    return pd.DataFrame({"date": np.asarray(dates), "y_hat": a + b * t})


def type_summary(s: pd.DataFrame, sess_by_series: dict | None = None) -> pd.DataFrame:
    """One row per series — the conclusions table. Sorted multi-rep first.
    The unit of every time-based number is the SESSION (one row per day);
    `Sets` still reports how many sets were averaged."""
    if s is None or not len(s):
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    if sess_by_series is None:
        sess_by_series = all_sessions(s)
    rows = []
    for (fam, db, series), g in s.groupby(["family", "dur_b", "series"],
                                          sort=False):
        sess = sess_by_series.get((fam, float(db)))
        if sess is None or not len(sess):
            sess = session_frame(g)
        n_days, n_sets = len(sess), len(g)
        fit = robust_fit(sess, y_col="w")
        slope, ci = fit["slope_w_month"], fit["ci_w_month"]
        first_w = float(sess["w"].iloc[0])
        last_w = float(sess["w"].iloc[-1])
        rest = pd.to_numeric(g["rest"], errors="coerce")
        med_rest = rest.median() if rest.notna().any() else np.nan
        ifam = pd.to_numeric(g["intensity"], errors="coerce")
        reps_med = int(g["reps"].median())
        if_med = float(ifam.median()) if ifam.notna().any() else np.nan
        rep_s_med = pd.to_numeric(g["rep_secs"], errors="coerce").median()
        # a robust trend is only a conclusion when its CI clears zero
        clear = (not np.isnan(ci)) and (abs(slope) > ci) and abs(slope) >= 1
        # The band comes from the SAME median IF printed beside it, so a row
        # can never contradict itself (no "IF 88" next to "<88 %"). A series
        # where most sets carry a broken intensity field says so outright
        # instead of borrowing a band from the few usable ones.
        flagged = (g["q_flag"].fillna("").astype(str).ne("").mean() > 0.5
                   if "q_flag" in g.columns else False)
        band = IF_BAND_BROKEN if flagged else intensity_band(if_med, rep_s_med)
        rows.append({
            "Type": FAMILY_SHORT.get(fam, fam),
            "Duration": dur_class_label(db),
            "Sets": n_sets,
            "Sessions": n_days,
            "First": pd.Timestamp(sess["date"].iloc[0]).date().isoformat(),
            "Last": pd.Timestamp(sess["date"].iloc[-1]).date().isoformat(),
            "First W": round(first_w),
            "Last W": round(last_w),
            "Δ W": round(last_w - first_w),
            "Δ %": (round((last_w - first_w) / first_w * 100, 1)
                    if first_w else None),
            "Best W": round(float(sess["w"].max())),
            "W/month": (round(slope, 2) if not np.isnan(slope) else None),
            "± 95%": (round(ci, 2) if not np.isnan(ci) else None),
            "Trend": ("↑" if clear and slope >= 1
                      else "↓" if clear and slope <= -1
                      else "→" if not np.isnan(slope) else "·"),
            "Ridden": band,
            "Rest": (fmt_rest(med_rest) if reps_med >= 2 else None),
            "IF %": (round(float(ifam.median())) if ifam.notna().any()
                     else None),
            "_multi": 0 if reps_med >= 2 else 1,
            "_n_days": n_days,
            "_n": n_sets,
            "_fam": fam, "_db": db, "_series": series,
            "_reps": reps_med,
        })
    out = (pd.DataFrame(rows)
           .sort_values(["_multi", "_n_days", "_n"], ascending=[True, False, False])
           .reset_index(drop=True))
    return out


# ── one selected series: full detail + trend ─────────────────────────────────
def series_sel(s: pd.DataFrame, family: str, dur_b=None,
               dur_cls: str | None = None) -> pd.DataFrame:
    """The rows of ONE series: training type × duration, by ONE criterion.

    `dur_b` is the rounded duration of the comparison table (nearest 5 from
    ten minutes up, so an 8-minute set never meets a 20-minute one).
    `dur_cls` is the rep-length class the
    Evolution day chart draws with ("10-20 min" holds anything under 22:00).
    The
    detail at the bottom of the Intervals page selects with `dur_cls`, so
    the same criterion produces the same days on both pages. Exactly one of
    the two is honoured — never a blend of the two.
    """
    if dur_cls is None and dur_b is None:
        raise ValueError("series_sel needs dur_b or dur_cls")
    if dur_cls is not None and "dur_cls" not in s.columns:
        s = s.assign(dur_cls=[rep_class(x) for x in
                              pd.to_numeric(s["rep_secs"], errors="coerce")])
    m = s["family"] == family
    m = m & (s["dur_cls"] == dur_cls) if dur_cls is not None \
        else m & np.isclose(s["dur_b"], dur_b)
    return s[m]


def series_detail(s: pd.DataFrame, family: str, dur_b=None,
                  dur_cls: str | None = None) -> pd.DataFrame:
    g = series_sel(s, family, dur_b, dur_cls).sort_values("date")
    if not len(g):
        return g
    multi = int(g["reps"].median()) >= 2
    tbl = pd.DataFrame({
        "Date": g["date"].dt.strftime("%Y-%m-%d"),
        "Activity": g["name"].fillna("").astype(str),
        "Reps": g["reps"].astype(int),
        "Duration": g["rep_secs"].map(fmt_min),
        "Duration s": g["rep_secs"].map(fmt_secs),
        "Rest": (g["rest"].map(fmt_rest) if multi
                 else "no set — gap to next effort"),
        "Avg W": g["set_w"].round(0),
        "NP W": g["set_np"].round(0),
        "HR": g["set_hr"].round(0),
        "IF %": g["intensity"].round(0),
        "Load": g["load"].round(0),
        "TSB": g["tsb"].round(1),
        "°C": g["temp"].round(1),
    })
    return tbl.iloc[::-1]      # newest first


def series_stats(s: pd.DataFrame, family: str, dur_b=None,
                 dur_cls: str | None = None) -> dict:
    g = series_sel(s, family, dur_b, dur_cls)
    g = g.sort_values("date").reset_index(drop=True)
    n = len(g)
    sess = session_frame(g)
    fit = robust_fit(sess, y_col="w")
    rob, rob_ci = fit["slope_w_month"], fit["ci_w_month"]
    n_days = int(sess["date"].nunique())
    now = pd.Timestamp.now().normalize()
    recent = sess[sess["date"] >= now - pd.Timedelta(days=28)]
    before = sess[sess["date"] < now - pd.Timedelta(days=28)]
    d28 = (float(recent["w"].max()) - float(before["w"].max())
           if len(recent) and len(before) else None)
    rest = pd.to_numeric(g["rest"], errors="coerce")
    rep_s = pd.to_numeric(g["rep_secs"], errors="coerce")
    ifam = pd.to_numeric(g["intensity"], errors="coerce")
    if_med = float(ifam.median()) if ifam.notna().any() else np.nan
    flagged = (g["q_flag"].fillna("").astype(str).ne("").mean() > 0.5
               if "q_flag" in g.columns else False)
    return {
        "n": n, "n_days": n_days,
        "slope_robust": rob, "slope_ci": rob_ci,
        "trend_clear": bool((not np.isnan(rob_ci)) and abs(rob) > rob_ci
                            and abs(rob) >= 1),
        "span_days": (float((sess["date"].iloc[-1] - sess["date"].iloc[0]).days)
                      if n_days else 0.0),
        "best": float(sess["w"].max()) if n_days else np.nan,
        "best_row": g.loc[g["set_w"].idxmax()] if n else None,
        "first_w": float(sess["w"].iloc[0]) if n_days else np.nan,
        "last_w": float(sess["w"].iloc[-1]) if n_days else np.nan,
        "last_date": sess["date"].iloc[-1] if n_days else pd.NaT,
        "d_best_28": d28,
        "reps_med": int(g["reps"].median()) if n else 0,
        "dur_min": (float(rep_s.min()) if n and rep_s.notna().any()
                    else np.nan),
        "dur_max": (float(rep_s.max()) if n and rep_s.notna().any()
                    else np.nan),
        "dur_med": (float(rep_s.median()) if n and rep_s.notna().any()
                    else np.nan),
        "rest_med": (rest.median() if rest.notna().any() else np.nan),
        "rest_varies": bool(rest.nunique(dropna=True) > 1) if n else False,
        "if_med": if_med,
        # same median-IF rule as the summary table — no row can contradict
        # itself
        "if_band": (IF_BAND_BROKEN if flagged
                    else intensity_band(if_med, rep_s.median())
                    if n else "—"),
        "set_w": g["set_w"].astype(float).to_numpy(),
        "date": g["date"].to_numpy(),
    }


def series_trend(s: pd.DataFrame, family: str, dur_b: float):
    """OLS line (date, y_hat) for one series, or None when under-powered.
    Session averages, like every other time-based number here."""
    g = s[(s["family"] == family) & (np.isclose(s["dur_b"], dur_b))]
    sess = session_frame(g)
    if len(sess) < MIN_TREND_N or sess["date"].nunique() < MIN_TREND_SESSIONS:
        return None
    return _ols_line(sess["date"].to_numpy(),
                     pd.to_numeric(sess["w"], errors="coerce").to_numpy())


# ── family small-multiples: one line per DURATION inside the family ──────────
def family_lines(s: pd.DataFrame, sess_by_series: dict | None = None) -> dict:
    """{family: [ {label, dur_label, df(points+OLS if possible)} ... ]}
    Points are SESSION averages, one per date — the same unit as the summary
    table and the explorer chart, so the three can never disagree."""
    if sess_by_series is None:
        sess_by_series = all_sessions(s)
    out = {}
    for fam, gf in s.groupby("family", sort=False):
        series = []
        for db, g in gf.groupby("dur_b", sort=True):
            sess = sess_by_series.get((fam, float(db)), session_frame(g))
            n_reps_med = int(g["reps"].median())
            line = _ols_line(sess["date"].to_numpy(),
                             pd.to_numeric(sess["w"], errors="coerce").to_numpy()) \
                if (len(sess) >= MIN_TREND_N
                    and sess["date"].nunique() >= MIN_TREND_SESSIONS) else None
            series.append({
                "label": f"{dur_class_label(db)} · {n_reps_med}×"
                         f"{' set' if n_reps_med > 1 else ' effort'}",
                "dur": db, "n": len(g), "n_days": int(sess["date"].nunique()),
                "points": sess[["date", "w"]].rename(columns={"w": "set_w"}),
                "trend": line,
            })
        series.sort(key=lambda x: x["dur"])
        out[fam] = series
    return out


# ── page entry ───────────────────────────────────────────────────────────────
def run_type_comparison(sets: pd.DataFrame) -> dict:
    s = prep_types(sets)
    quality = s.attrs.get("quality") if hasattr(s, "attrs") else None
    excluded = s.attrs.get("excluded") if hasattr(s, "attrs") else None
    if s is None or not len(s):
        return {"ok": False, "quality": quality, "excluded": excluded,
                "reason": ("No comparable sets after the data-quality screen "
                           "— see the exclusions below, or sync more "
                           "activities.")}
    sess_by_series = all_sessions(s)
    summary = type_summary(s, sess_by_series)
    fams = [f for f in FAMILY_ORDER
            if (s["family"] == f).any()] + \
           sorted(set(s["family"]) - set(FAMILY_ORDER))
    return {
        "ok": True, "sets": s, "summary": summary, "families": fams,
        "lines": family_lines(s, sess_by_series),
        "sessions_by_series": sess_by_series,
        "n_sets": int(len(s)),
        "n_sessions": int(s["date"].nunique()),
        "quality": quality, "excluded": excluded,
    }
