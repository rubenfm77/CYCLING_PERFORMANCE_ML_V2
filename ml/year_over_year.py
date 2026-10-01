# ml/year_over_year.py — evolution of ONE training type, year over year. (NEW FILE)
"""
The user rule again, at SESSION grain: a training type is only ever compared
with ITSELF at the SAME duration class. This module never puts two types in
the same row, never averages a 90-minute ride with a 4-hour one, and never
mixes heat bands into a year-on-year number.

Grain note: ml/type_comparison.py works on individual REPS (one interval of a
set). This module works on whole SESSIONS, because that is the grain the
athlete's own `training_type` labelling exists at. The two must not be mixed.

What it produces, for one selected type:
  * coverage()   - how many sessions sit in every (duration class x year) cell,
                   and which cells are big enough to be shown at all;
  * evolution()  - one line per DURATION CLASS across the years, a point only
                   where the cell has >= MIN_CELL_N sessions, and the line
                   BREAKS at every thin year instead of being interpolated
                   across it (never promise a line the data does not show);
  * yoy_table()  - the "compare with the previous year" answer: every year with
                   its median, its n, the previous comparable year's median and
                   n, and the delta — a delta is printed only when BOTH years
                   clear MIN_CELL_N, because 5 sessions against 1 is not a
                   change, it is noise;
  * watts_gate() - the honest statement about the OUTCOME side. Measured
                   threshold watts do not exist before Dec 2025, so there is no
                   year-over-year watts comparison to draw for 2019-2025. This
                   function returns the counts so the UI can show the gap
                   rather than quietly omit it.

Metric choice is deliberate. `power_np` is NOT offered: intervals.icu's
normalised power is present for 139/139 rows in 2020 but only 30/141 in 2026,
so any series built on it would end mid-2026 for a data reason, not a
physiological one. `power_avg` (1039 rows, every year) is the wattage series.

Honesty rules:
  * descriptive only — observed history, descriptive association, no causal or
    predictive claims, no "this caused that";
  * n is always on the chart and in every table row, next to the number it
    belongs to;
  * a year with too few sessions is BLANK, never zero, never carried forward;
  * sessions with the metric missing are counted and reported, so a median
    resting on half the cell is visible as such;
  * descriptive spread is the interquartile range — a median without it can
    hide that the cell is bimodal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A year-point needs this many sessions in its (type x duration class x year)
# cell before it is drawn at all. Below this, one all-out ride moves the median
# and the chart would be reporting a single session as a year.
MIN_CELL_N = 3

# The OUTCOME gate needs its own, much higher floor than the training side.
# A threshold reading is not one-per-session like a duration — several can
# fall in one day, and they cluster in whichever weeks the athlete actually
# tested. A year therefore has to carry enough readings for a stable median
# (MIN_WATTS_YEAR_N) AND spread across enough months to represent the year
# (MIN_WATTS_YEAR_MONTHS), or it is not a comparable year. 3 readings inside
# one December week fails both, which is the point.
MIN_WATTS_YEAR_N = 12
MIN_WATTS_YEAR_MONTHS = 3

# Duration CLASSES for whole sessions. The user rule — whole minutes, halves
# down — governs REP lengths; at session scale a minute grid is meaningless
# (nothing is between 2h07m and 2h08m), so sessions are binned into the ranges
# that actually describe them, and the EXACT duration is always reported
# alongside (median h:mm:ss per cell, and the raw seconds in every detail row).
DUR_CLASSES = [
    ("under 45 min", 0.0, 45.0),
    ("45-90 min", 45.0, 90.0),
    ("90-150 min", 90.0, 150.0),
    ("150-240 min", 150.0, 240.0),
    ("240+ min", 240.0, np.inf),
]
DUR_ORDER = [d[0] for d in DUR_CLASSES]

# Metrics offered in the view. `agg` is the statistic shown; every metric also
# reports the interquartile range of the underlying sessions.
METRICS = [
    # key,            label,                     unit,   decimals, agg
    ("power_avg",      "Average power",           "W",    0,       "median"),
    ("w_per_kg",       "Average power",           "W/kg", 2,       "median"),
    ("if_score",       "Intensity factor",        "",     3,       "median"),
    ("tss",            "Training stress score",   "",     1,       "median"),
    ("duration_s",     "Session duration",        "",     0,       "median"),
    ("elevation",      "Elevation gain",          "m",    0,       "median"),
    ("temp_avg",       "Average temperature",     "°C",   1,       "median"),
    ("efficiency",     "Efficiency (W per bpm)",  "",     3,       "median"),
    ("hr_avg",         "Average heart rate",      "bpm",  0,       "median"),
]
METRIC_KEYS = [m[0] for m in METRICS]
METRIC_BY_KEY = {m[0]: m for m in METRICS}

# The blank-label test is NOT redefined here. core.data.BLANK_TYPE_TOKENS is the
# loader's own, case-sensitive set ({"—", "-", "", "nan", "None"}); a second
# copy of that list is how a session ends up counted on this page and absent
# from the Training page, so this module imports the loader's set and uses the
# loader's exact predicate. Labels outside MAIN_TYPES are the athlete's own
# custom values — counted, never folded into a first-class type, never invented.
from core.data import BLANK_TYPE_TOKENS  # noqa: E402


def _has_label(series: pd.Series) -> pd.Series:
    """Exactly core.data's test: present, and not one of its blank tokens."""
    return series.notna() & ~series.astype(str).str.strip().isin(BLANK_TYPE_TOKENS)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric view of a column that may be absent (API rows are sparse)."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def dur_class(mins) -> str:
    """Session length -> duration class label. Never crosses a boundary silently."""
    if mins is None or (isinstance(mins, float) and np.isnan(mins)):
        return "unknown"
    for label, lo, hi in DUR_CLASSES:
        if lo <= mins < hi:
            return label
    return "unknown"


def fmt_dur(secs) -> str:
    """Exact duration as h:mm:ss. The user's rule: minutes shown whole, and the
    exact seconds always visible rather than rounded away."""
    if secs is None or (isinstance(secs, float) and np.isnan(secs)):
        return "—"
    secs = int(round(float(secs)))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def prep(df: pd.DataFrame) -> pd.DataFrame:
    """Add the year / duration-class / metric columns this module needs.

    Rows with no usable label are dropped and COUNTED — the caller reports the
    number, because an unlabelled session is a gap in the athlete's filing, not
    a session that did not happen.
    """
    if df is None or not len(df):
        return pd.DataFrame(columns=["year", "dur_class", "dur_s"] + METRIC_KEYS)

    out = pd.DataFrame(index=df.index)
    label = df["training_type"].astype(object) if "training_type" in df.columns else None
    if label is not None:
        keep = _has_label(label)
    else:
        keep = pd.Series(True, index=df.index)
    out = out[keep].copy()
    if not len(out):
        return out.assign(year=pd.Series(dtype="int64"), dur_class=pd.Series(dtype="object"))

    out["tt"] = label[keep].astype(str).str.strip()
    out["date"] = df.loc[keep, "date"]
    out["year"] = df.loc[keep, "date"].dt.year
    out["dur_s"] = _num(df.loc[keep], "duration_s")
    out["dur_min"] = out["dur_s"] / 60.0
    out["dur_class"] = out["dur_min"].apply(dur_class)
    for k in METRIC_KEYS:
        out[k] = _num(df.loc[keep], k)
    # Carried for the outcome gate only — never plotted as a training metric.
    out["icu_pm_ftp_watts"] = _num(df.loc[keep], "icu_pm_ftp_watts")
    return out.reset_index(drop=True)


def coverage(s: pd.DataFrame, tt: str, min_n: int = MIN_CELL_N) -> pd.DataFrame:
    """Sessions per (duration class x year) for ONE type, flagged drawable or not.

    This is the table the view shows BEFORE any chart, so the user can see the
    shape of their own data and judge whether a line is worth reading.
    """
    if s is None or not len(s):
        return pd.DataFrame(columns=["dur_class", "year", "n", "drawable"])
    g = s[s["tt"] == tt]
    if not len(g):
        return pd.DataFrame(columns=["dur_class", "year", "n", "drawable"])
    out = (g.groupby(["dur_class", "year"]).size().rename("n").reset_index())
    out["drawable"] = out["n"] >= min_n
    out["dur_class"] = pd.Categorical(out["dur_class"], categories=DUR_ORDER, ordered=True)
    return out.sort_values(["dur_class", "year"]).reset_index(drop=True)


def summary(s: pd.DataFrame, tt: str, min_n: int = MIN_CELL_N) -> dict:
    """How much of this type is comparable at all — drives the 'too thin' message."""
    cov = coverage(s, tt, min_n)
    years_covered = sorted(cov.loc[cov["drawable"], "year"].unique().tolist()) if len(cov) else []
    per_class = (
        cov[cov["drawable"]].groupby("dur_class")["year"].nunique().to_dict()
        if len(cov) else {}
    )
    return {
        "type": tt,
        "n_sessions": int(len(s[s["tt"] == tt])) if s is not None and len(s) else 0,
        "n_cells": int(len(cov)),
        "n_drawable": int(cov["drawable"].sum()) if len(cov) else 0,
        "years_drawable": years_covered,
        "years_per_class": per_class,
        "comparable": len(years_covered) >= 2,
        "min_n": min_n,
    }


def cell_stats(s: pd.DataFrame, tt: str, metric: str, min_n: int = MIN_CELL_N) -> pd.DataFrame:
    """One row per (duration class x year): the median, its n, and the IQR.

    `n` is always the number of sessions in the CELL. `n_metric` is how many of
    them actually carry the metric — a median resting on 6 of 20 sessions has to
    say so, otherwise it looks like a 20-session fact.
    """
    _, label, unit, dec, agg = METRIC_BY_KEY[metric]
    g = s[(s["tt"] == tt) & s["dur_class"].isin(DUR_ORDER)] if s is not None else None
    if g is None or not len(g):
        return pd.DataFrame(columns=["dur_class", "year", "n", "n_metric",
                                    "value", "q1", "q3", "drawable"])

    rows = []
    for (dcls, yr), cell in g.groupby(["dur_class", "year"], observed=True):
        v = pd.to_numeric(cell[metric], errors="coerce").dropna()
        rec = {
            "dur_class": dcls, "year": int(yr), "n": int(len(cell)),
            "n_metric": int(len(v)), "value": np.nan,
            "q1": np.nan, "q3": np.nan, "drawable": False,
        }
        if len(v):
            q = v.quantile([0.25, 0.5, 0.75])
            rec["q1"], rec["value"], rec["q3"] = float(q[0.25]), float(q[0.5]), float(q[0.75])
        rec["drawable"] = (rec["n"] >= min_n) and (rec["n_metric"] >= min_n) and bool(
            not np.isnan(rec["value"]))
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["dur_class"] = pd.Categorical(out["dur_class"], categories=DUR_ORDER, ordered=True)
    return out.sort_values(["dur_class", "year"]).reset_index(drop=True)


def evolution(s: pd.DataFrame, tt: str, metric: str, min_n: int = MIN_CELL_N) -> pd.DataFrame:
    """Long frame ready to plot: one row per drawn point, so a thin year simply
    has no row and the line breaks there."""
    cs = cell_stats(s, tt, metric, min_n)
    if not len(cs):
        return cs
    return cs[cs["drawable"]].copy()


def yoy_table(s: pd.DataFrame, tt: str, metric: str, min_n: int = MIN_CELL_N) -> pd.DataFrame:
    """'Compare with the previous year', per duration class.

    `prev_year` is the previous year in the SAME duration class that cleared
    min_n — not simply year-1. If 2024 was too thin, 2025 is compared against
    2023 and the gap in years is stated, so a 2-year change is never printed
    as if it were a 1-year change.
    """
    cs = cell_stats(s, tt, metric, min_n)
    if not len(cs):
        return pd.DataFrame(columns=["dur_class", "year", "n", "value",
                                    "prev_year", "prev_n", "prev_value", "delta",
                                    "pct", "gap_years"])
    cs = cs.sort_values(["dur_class", "year"]).copy()
    cs["prev_year"] = np.nan
    cs["prev_n"] = np.nan
    cs["prev_value"] = np.nan
    for dcls in cs["dur_class"].unique():
        m = cs["dur_class"] == dcls
        last_y = last_v = last_n = np.nan
        for i in cs.index[m]:
            if cs.at[i, "drawable"]:
                cs.at[i, "prev_year"] = last_y
                cs.at[i, "prev_value"] = last_v
                cs.at[i, "prev_n"] = last_n
                last_y, last_v, last_n = cs.at[i, "year"], cs.at[i, "value"], cs.at[i, "n"]
    ok = cs["drawable"] & cs["prev_value"].notna()
    cs["delta"] = np.where(ok, cs["value"] - cs["prev_value"], np.nan)
    cs["pct"] = np.where(ok & (cs["prev_value"] != 0),
                         (cs["value"] - cs["prev_value"]) / cs["prev_value"] * 100.0, np.nan)
    cs["gap_years"] = np.where(ok, cs["year"] - cs["prev_year"], np.nan)
    cs.loc[~cs["drawable"], ["value", "delta", "pct"]] = np.nan
    return cs


def fmt_value(metric: str, v) -> str:
    """Big numbers get thousands separators, per the display rule."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    _, _, _, dec, _ = METRIC_BY_KEY[metric]
    return f"{v:,.{dec}f}"


def fmt_delta(metric: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    _, _, _, dec, _ = METRIC_BY_KEY[metric]
    return f"{v:+,.{dec}f}"


def watts_gate(s: pd.DataFrame) -> dict:
    """The outcome side, stated honestly.

    Threshold watts are MEASURED only from Dec 2025. There is therefore no
    year-over-year threshold comparison for 2019-2025 — not a weak one, not a
    modelled one, none.

    "Comparable year" is deliberately strict, and an earlier looser version of
    this gate got it wrong: counting any year with >= 3 readings declared 2025
    comparable on the strength of 3 sessions, all inside one December week,
    and then claimed a year-over-year watts comparison existed. It does not.
    A year has to carry enough readings to have a stable median AND spread
    across enough months to represent the year, so both gates apply.
    """
    if s is None or not len(s):
        return {}
    pm = s[pd.to_numeric(s["icu_pm_ftp_watts"], errors="coerce").notna()] \
        if "icu_pm_ftp_watts" in s.columns else s.iloc[0:0]
    if not len(pm):
        return {"n_measured": 0, "can_compare_years": False, "comparable_years": [],
                "by_year": {}, "by_year_months": {}, "first_month": None,
                "last_month": None, "n_months": 0, "years_without": []}

    pmy = pm.assign(_m=pm["date"].dt.to_period("M").astype(str))
    by_year = {int(k): int(v) for k, v in pm.groupby("year").size().items()}
    by_year_months = {int(k): int(v) for k, v in pmy.groupby("year")["_m"].nunique().items()}
    months = sorted(pmy["_m"].unique().tolist())
    all_years = sorted(int(y) for y in s["year"].dropna().unique())
    comparable = [y for y in by_year
                  if by_year[y] >= MIN_WATTS_YEAR_N
                  and by_year_months.get(y, 0) >= MIN_WATTS_YEAR_MONTHS]
    return {
        "n_measured": int(len(pm)),
        "by_year": by_year,
        "by_year_months": by_year_months,
        "min_year_n": MIN_WATTS_YEAR_N,
        "min_year_months": MIN_WATTS_YEAR_MONTHS,
        "first_month": months[0] if months else None,
        "last_month": months[-1] if months else None,
        "n_months": len(months),
        "years_without": [y for y in all_years if y not in by_year],
        "comparable_years": sorted(comparable),
        "can_compare_years": len(comparable) >= 2,
    }


def types_in_scope(s: pd.DataFrame, main_types) -> pd.DataFrame:
    """Which first-class types have enough sessions per year to be worth opening.

    Ordered by how many years are comparable, best first, so the type the user
    most likely wants is the default selection.
    """
    rows = []
    for t in main_types:
        sm = summary(s, t)
        if sm["n_sessions"]:
            rows.append({
                "training_type": t, "n_sessions": sm["n_sessions"],
                "n_drawable": sm["n_drawable"], "years": len(sm["years_drawable"]),
                "years_list": sm["years_drawable"], "comparable": sm["comparable"],
            })
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    return out.sort_values(["comparable", "years", "n_sessions"],
                           ascending=[False, False, False]).reset_index(drop=True)
