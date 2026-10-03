# views/interval_watts.py — POWER LAW: watts against duration. (NEW FILE)
r"""
The request was "isolate not avg watts but the interval watts"; later "this page
is not useful based on what we have in evolution or intervals — maybe rename it
power law and focus on that". So the page is named for the one thing only it
does: the power-duration LAW, the relation between watts and how long you can
hold them. Average power is not offered here at all, and the only watt figures
on screen come from a rep.

What was cut, and why: the two "over time" charts (the line per rep length class
and the one-bar-per-day chart) drew the same rows Evolution already draws on a
date axis. A page that repeats another page is not useful, so those charts are
gone and the tab says where they went. The law leads the tab now, with the
coverage tables underneath it — they are what the law is fitted on.

Interval watts exist in two places in this athlete's file, and they cover
different years:

  MEASURED   intervals.icu auto-detected efforts, 2025-2026. Real watts, and the
             only interval data for this year. The law tab below.
  PRESCRIBED the coach's own Spanish/Catalan comments in `WorkoutDescription`,
             2019-2025, and NOTHING in 2026. The watts are a target RANGE and the
             lower end is used, per the athlete's instruction. That half is
             rendered by `prescribed_tab()` — now called from the **Evolution**
             page, where prescribed-vs-actual belongs, and no longer from
             `render()` below.

They are kept apart on purpose. The prescribed comments stop in 2025 and the
measured efforts begin in 2025, so drawing one line through both would mean
interpolating 2025-2026 out of nothing and calling it the athlete's progress. The
boundary between the two pages is the honest rendering of that wall, and the
wall is also printed in numbers so it is not something the reader has to infer
from the layout.

Every chart here is bucketed by REP LENGTH CLASS and never by a single pooled
"interval watts" number, because a 3x1' effort and a 2x20' effort are not the same
measure. Reps below 90 s keep their exact seconds rather than being rounded into a
bucket, and the exact rep length travels with every table row.

One type at a time, always: no chart on this page puts two training types on one
axis, and no cell averages two of them together.

Full history is read from ctx.df_all, not ctx.df, for the same reason as the
Evolution page: the sidebar range filter defaults to six months, which would
delete every year being compared here.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.components import (callout, dataframe, metric_card, page_header,
                             section, show)
from core.theme import C, H_STD, MAIN_TYPES, style_figure
from ml import interval_watts as iw
from ml.type_comparison import fmt_min
from views.intervals_view import _lane_legend

# One colour per REP LENGTH CLASS, fixed by position so the same class is the
# same colour everywhere on this page and on every rerun. Assigned by hand from
# the theme palette rather than generated, so it never shifts under the reader.
REP_COLORS = {
    "under 90s": C["red"],
    "90s-5min": C["orange"],
    "5-10 min": C["yellow"],
    "10-20 min": C["green"],
    "20-30 min": C["teal"],
    "30+ min": C["purple"],
}


def _rep_color(cls: str) -> str:
    return REP_COLORS.get(str(cls).strip(), C["accent"])


def _power_law_frame(M: pd.DataFrame, tt: str):
    """(observed bests, fitted curve) for the ley-de-potencias chart.

    Observed: the best watts per distinct duration, exactly the points the fit
    used. Curve: both fitted models evaluated on a dense grid INSIDE the fitted
    duration range only - extending a fit past its own data is how a model
    starts making promises nobody measured.
    """
    try:
        s = M[M["tt"] == tt].copy()
        if not len(s):
            return None
        s["secs"] = pd.to_numeric(s["secs"], errors="coerce")
        s["w"] = pd.to_numeric(s["w"], errors="coerce")
        s = s.dropna(subset=["secs", "w"])
        s = s[(s["secs"] > 0) & (s["w"] > 0)]
        obs = (s.groupby("secs", as_index=False)["w"].max()
               .rename(columns={"w": "w"}).sort_values("secs"))
        if len(obs) < iw.MIN_FIT_PTS:
            return None
        t_lo, t_hi = float(obs["secs"].min()), float(obs["secs"].max())
        grid = np.linspace(t_lo, t_hi, 80)
        pl = iw.power_law(M, tt)
        if not pl["ok"]:
            return None
        curve = pd.DataFrame({
            "secs": grid,
            "cp": pl["cp"] + pl["w_prime_kj"] * 1000.0 / grid,
            "law": pl["a"] * grid ** (-pl["b"]),
        })
        return obs, curve
    except Exception:
        return None


def _cells_datable(cells: pd.DataFrame, years, label_col: str) -> pd.DataFrame:
    """Coverage grid: one row per class, one column per year, n per cell.

    Bold clears the floor, `·n` means the sessions exist but the cell is too thin
    to draw, and `—` is an empty cell. A thin year is shown as present-but-thin
    rather than hidden, because "you did 2 of these" is information.
    """
    rows = []
    for cls in iw.REP_ORDER:
        sub = cells[cells["cls"].astype(str) == cls]
        if not len(sub):
            continue
        by_year = sub.set_index("year")
        rec = {label_col: cls}
        for y in years:
            if y in by_year.index:
                r = by_year.loc[y]
                n = int(r["n"])
                rec[str(y)] = f"**{n}**" if bool(r["drawable"]) else f"·{n}"
            else:
                rec[str(y)] = "—"
        rec["Total"] = f"{int(sub['n'].sum()):,}"
        rows.append(rec)
    return pd.DataFrame(rows)


def _time_figure(daily: pd.DataFrame, tt: str, phys: str, yr) -> go.Figure:
    """The DATE x-axis figure: one point per day, one line per rep length class.

    Shared by the Interval watts page and the Fitness page, so the Fitness page
    draws the IDENTICAL chart from the identical data — same colours, same
    n-hovers, same dotted average, x-axis type "date".
    """
    fig_t = go.Figure()
    for cls in sorted(daily["cls"].unique()):
        dcls = daily[daily["cls"] == cls]
        fig_t.add_trace(go.Scatter(
            x=dcls["date"], y=dcls["med_w"], mode="lines+markers", name=cls,
            line=dict(color=_rep_color(cls), width=2),
            marker=dict(color=_rep_color(cls), size=9,
                        line=dict(color=C["panel"], width=1)),
            customdata=[[iw.fmt_watts(w), f"n = {int(n):,}",
                         fmt_min(r)]
                        for w, n, r in zip(dcls["med_w"], dcls["n"],
                                           dcls["rep_secs"])],
            connectgaps=False,
            hovertemplate="<b>%{fullData.name}</b> · %{x|%d %b %Y}<br>"
                          "median %{customdata[0]}<br>%{customdata[1]}<br>"
                          "typical rep %{customdata[2]}<extra></extra>",
        ))
    mean_w = float(daily["med_w"].mean())
    fig_t.add_trace(go.Scatter(
        x=[daily["date"].min(), daily["date"].max()],
        y=[mean_w, mean_w], mode="lines", name="Avg of charted points",
        line=dict(color=C["yellow"], width=2, dash="dot"),
        hovertemplate=f"avg {iw.fmt_watts(mean_w)} W<extra></extra>",
    ))
    style_figure(
        fig_t,
        f"{tt} — {phys} interval watts over time, {yr}"
        f"<br><sup>x-axis is the date. One line per rep length class, n on every "
        f"point; the dotted line is the average of the charted points. Observed "
        f"history, not a cause.</sup>",
        H_STD)
    _lane_legend(fig_t, [t.name for t in fig_t.data], min_w=430)
    fig_t.update_xaxes(type="date")
    if len(daily):
        y_max = daily["med_w"].max() * 1.25
        y_min = daily["med_w"].min() * 0.75
        fig_t.update_yaxes(range=[y_min, y_max], tickformat=",.0f",
                           autorange=False)
    else:
        fig_t.update_yaxes(tickformat=",.0f", rangemode="tozero")
    return fig_t


def fitness_section(df_all: pd.DataFrame) -> None:
    """The interval-watts charts on the Fitness page.

    Two of them, both asked for by name: the DATE x-axis "Measured interval
    watts over time" chart, and the power-duration law. Same data (full history
    from df_all, not the sidebar range), same rules as the Interval watts page:
    one training type at a time, physiology split (FTP ≥ 10 min / VO2MAX below),
    isolated pushes already discarded by ml.interval_watts.measured(), and the
    law fitted on every effort of the type in every year.
    """
    M = iw.measured(df_all)
    if not len(M):
        callout("No measured efforts",
                "This file carries no detected effort, so neither chart below "
                "can be drawn.", C["orange"], "\U0001F6A7")
        return

    years = sorted(int(y) for y in M["year"].dropna().unique())
    types = sorted(str(t) for t in M["tt"].dropna().unique())

    c1, c2, c3 = st.columns(3)
    with c1:
        tt = st.selectbox(
            "Training type", types, key="fit_iw_type",
            help="One type at a time. Two training types are never on the same "
                 "axis here, exactly as on the Interval watts page.")
    with c2:
        yr = st.selectbox(
            "Year", years, index=len(years) - 1, key="fit_iw_year",
            help="The chart below is one year; the power-duration law under it "
                 "is always fitted on every year.")
    with c3:
        phys = st.selectbox(
            "Effort", ["FTP", "VO2MAX"], key="fit_iw_phys",
            help="FTP = efforts of 10 minutes and longer. VO2MAX = efforts "
                 "below 10 minutes. The two are never on one chart.")

    M_all = M
    M = M[M["phys"] == phys] if "phys" in M.columns else M
    cells = iw.measured_cells(M, yr)
    mine = cells[cells["tt"] == tt].sort_values("med_w", ascending=False)

    section("\U0001F4C8 Measured interval watts over time")
    if not len(mine):
        callout("Nothing for this type", f"**{tt}** has no {phys} detected "
                f"effort in {yr}.", C["orange"], "\U0001F6A7")
        _law_section(M_all, tt, 0, phys)
        return

    st.caption(
        f"**{phys} efforts of {tt}, {yr}: one point per day, x-axis is the "
        "date.** One series per rep length class; two classes never share a "
        "number. The dotted line is the average of the charted points."
    )
    M_yr = M[M["year"] == yr] if yr is not None else M
    M_tt = M_yr[M_yr["tt"] == tt]
    if not len(M_tt):
        callout("Nothing for this type", f"**{tt}** has no {phys} detected "
                f"effort in {yr}.", C["orange"], "\U0001F6A7")
        _law_section(M_all, tt, 0, phys)
        return

    daily = (M_tt.groupby(["cls", "date"], as_index=False)
             .agg(med_w=("w", "median"), n=("w", "size"),
                  rep_secs=("secs", "median")))
    daily = daily.sort_values("date")
    show(_time_figure(daily, tt, phys, yr))
    st.caption(
        f"{len(daily):,} day(s), {int(daily['n'].sum()):,} detected effort(s). "
        "Every point carries its n in the hover; the class under each line is "
        "printed in the legend, so a 15-minute line is never read as a "
        "3-minute one."
    )

    _law_section(M_all, tt, int(len(M_tt)), phys)


def render(head, ctx):
    page_header(
        "\U0001F3AF",
        "Power law — watts against duration",
        "Not average power, and not a timeline: the relation between how many "
        "watts and how long you can hold them. Every figure comes from a single "
        "interval, and intervals of different lengths are never averaged "
        "together — a 30-second effort and a 20-minute effort are separate "
        "series with separate counts. Watts over DATE are on Evolution, watts "
        "over duration are here.",
    )

    st.caption(
        f"Full history, {ctx.df_all['date'].min().date()} → "
        f"{ctx.df_all['date'].max().date()} · **{len(ctx.df_all):,} sessions**. "
        f"The sidebar range filter does not apply here: it defaults to 6 months, "
        f"which would delete every year compared below."
    )

    P = iw.prescribed(ctx.df_all)
    M = iw.measured(ctx.df_all)
    a = iw.audit(ctx.df_all)
    cov = iw.coverage_note(P, M)

    if not len(M) and not len(P):
        callout("No interval data", "Neither the coach's comments nor the "
                "detected-effort field are present in this file.", C["red"], "⚠️")
        return

    # ── The two sources, stated before any chart ─────────────────────────────
    mc0 = st.columns(4)
    with mc0[0]:
        metric_card("Measured efforts", f"{cov['measured_n']:,}",
                    f"from {cov['measured_years'][0]}" if cov["measured_years"]
                    else "none", "green")
    with mc0[1]:
        metric_card("Measured years",
                    f"{len(cov['measured_years'])}",
                    "–".join(str(y) for y in cov["measured_years"])
                    if cov["measured_years"] else "none", "muted")
    with mc0[2]:
        metric_card("Prescribed sessions", f"{cov['prescribed_n']:,}",
                    f"to {cov['prescribed_last_year']}"
                    if cov["prescribed_last_year"] else "none", "accent")
    with mc0[3]:
        metric_card("Prescribed years",
                    f"{len(cov['prescribed_years'])}",
                    "–".join(str(y) for y in cov["prescribed_years"])
                    if cov["prescribed_years"] else "none", "muted")

    last_p = cov["prescribed_last_year"]
    if last_p and 2026 not in cov["prescribed_years"]:
        callout(
            "These two sources do not overlap",
            f"The coach's written comments carry a target wattage for "
            f"**{cov['prescribed_n']:,} sessions, ending in {last_p}** — there is "
            f"not one prescription in 2026. The detected-effort field is the "
            f"mirror image: **{cov['measured_n']:,} efforts from "
            f"{cov['measured_years'][0]} onward**, and almost none before. "
            f"So this year has measured interval watts and no prescribed ones, and "
            f"the years before have prescribed ones and almost no measured ones. "
            f"They are shown on **separate pages** — the prescriptions live on "
            f"**Evolution**, the measured efforts here — and are never joined "
            f"into one line, "
            f"because 'the watts you were told to do' and 'the watts you actually "
            f"did' are different numbers and bridging the gap between them would "
            f"invent a trend that no measurement supports.",
            C["orange"], "🚧")

    tab_meas, tab_audit = st.tabs([
        "\U0001F3AF The power-duration law",
        "\U0001F50D What could be read",
    ])

    with tab_meas:
        _law_tab(M)

    with tab_audit:
        _audit_tab(a, cov, P, M)


def _durlab(secs) -> str:
    """One duration, in the athlete's rule: seconds while it lasts less than
    a minute, minutes from a minute up.

    From a minute up the exact seconds stay in brackets (`14.7 min (880 s)`),
    because the rule governs the FORM the athlete reads and the earlier
    standing rule says the exact measured length must never be hidden by a
    rounded label. Under a minute the seconds are already exact, so nothing
    is repeated.
    """
    try:
        s = float(secs)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(s):
        return "—"
    if s < 60:
        return fmt_min(s)
    return f"{fmt_min(s, 1)} ({s:.0f} s)"


def _law_section(M_all: pd.DataFrame, tt: str, n_focus: int,
                 phys: str | None) -> None:
    """Power-duration law: CP + W' and the power-law exponent, side by side.

    Fitted on every effort of the type in every year - the law is the relation
    BETWEEN durations, so filtering it to one year or one physiology would fit
    a line to a slice of the curve. n_focus/phys describe the chart above it
    and are only used in the wording.
    """
    section("\U0001F3AF Power-duration law")
    pl = iw.power_law(M_all, tt)
    if not pl["ok"]:
        callout("Not enough durations to fit the law",
                f"{pl['reason']} ({pl['n']} distinct duration(s) of **{tt}**; "
                f"{iw.MIN_FIT_PTS} needed). Sync more activities, or pick "
                f"another training type.", C["yellow"], "\U0001F6A7")
        return
    focus = (f"{n_focus:,} {phys + ' ' if phys else ''}effort(s) charted above · "
             if n_focus else "")
    st.caption(
        f"Fitted on the BEST watts observed at each distinct duration for "
        f"**{tt}** — every year, every effort, because the law IS the "
        f"relation between watts and duration. "
        f"{pl['n']} distinct durations, "
        f"{fmt_min(pl['t_lo'])} to {fmt_min(pl['t_hi'])}."
    )
    pc = st.columns(5)
    with pc[0]:
        metric_card("CP", f"{pl['cp']:.0f} W",
                    f"model P = CP + W′/t · R² = {pl['r2_cp']}", "accent")
    with pc[1]:
        metric_card("W′", f"{pl['w_prime_kj']:.1f} kJ",
                    f"anaerobic capacity · R² = {pl['r2_cp']}", "green")
    with pc[2]:
        metric_card("Exponent b", f"{pl['b']:.3f}",
                    f"P = a·t^-b, a = {pl['a']:.0f} · R² = {pl['r2_law']}",
                    "purple")
    with pc[3]:
        metric_card("Durations fitted", f"{pl['n']}",
                    f"{fmt_min(pl['t_lo'])} – {fmt_min(pl['t_hi'])}",
                    "muted")
    with pc[4]:
        metric_card("n (efforts)", f"{int(len(M_all[M_all['tt'] == tt])):,}",
                    focus + f"all years of {tt}", "yellow")

    # The law itself, drawn: observed bests as points, both fits as lines.
    fit_frame = _power_law_frame(M_all, tt)
    if fit_frame:
        obs, curve = fit_frame
        # Duration on the hover, in the athlete's rule — seconds under a
        # minute, minutes from a minute up — with the exact seconds kept in
        # brackets there, so a rounded label never hides a measured length.
        _lab = [_durlab(v) for v in obs["secs"]]
        _lab_c = [_durlab(v) for v in curve["secs"]]
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(
            x=obs["secs"], y=obs["w"], mode="markers",
            name="Best observed",
            marker=dict(size=9, color=C["accent"],
                        line=dict(color=C["panel"], width=1)),
            customdata=_lab,
            hovertemplate="duration %{customdata} · %{y:,.0f} W<extra></extra>"))
        fig_l.add_trace(go.Scatter(
            x=curve["secs"], y=curve["cp"], mode="lines",
            name=f"CP + W′/t (R²={pl['r2_cp']})",
            line=dict(color=C["yellow"], width=2, dash="dot"),
            customdata=_lab_c,
            hovertemplate="duration %{customdata} · CP model %{y:,.0f} W"
                          "<extra></extra>"))
        fig_l.add_trace(go.Scatter(
            x=curve["secs"], y=curve["law"], mode="lines",
            name=f"Power law (R²={pl['r2_law']})",
            line=dict(color=C["purple"], width=2),
            customdata=_lab_c,
            hovertemplate="duration %{customdata} · power law %{y:,.0f} W"
                          "<extra></extra>"))
        style_figure(
            fig_l,
            f"{tt} — power-duration law: observed bests and both fits"
            f"<br><sup>points are the best watts ridden at each duration; "
            f"neither line is extended beyond the fitted range.</sup>",
            H_STD)
        _lane_legend(fig_l, [t.name for t in fig_l.data], min_w=430)
        # Ticks read seconds under a minute and minutes above it — the same
        # rule as every other duration on the dashboard, never `880 s`. An
        # axis type can carry 198 distinct durations, so the ticks are a
        # readable sample of them (first, last, evenly spaced in between)
        # instead of 198 labels fighting for one line.
        _vals = list(obs["secs"])
        _step = int(np.ceil(len(_vals) / 12.0)) if _vals else 1
        _sel = _vals[::_step]
        if _vals and _sel[-1] != _vals[-1]:
            _sel.append(_vals[-1])
        _ticks = iw.axis_labels(_sel)
        fig_l.update_xaxes(title_text="duration", tickmode="array",
                           tickvals=_sel, ticktext=_ticks)
        fig_l.update_yaxes(tickformat=",.0f", rangemode="tozero")
        show(fig_l)

    st.markdown(
        f"- **CP {pl['cp']:.0f} W** and **W′ {pl['w_prime_kj']:.1f} kJ** come "
        f"from the hyperbolic fit P = CP + W′/t (R² = {pl['r2_cp']}); "
        f"**b = {pl['b']:.3f}** from the power law P = {pl['a']:.0f}·t^−b "
        f"(R² = {pl['r2_law']}). Both are fits to your own best efforts, "
        f"not a test result and not a prescription.\n"
        f"- **Descriptive only**: the law describes how your best watts fell "
        f"with duration in this file. It makes no claim about how you would "
        f"ride a new duration, and no claim about why the numbers are what "
        f"they are."
    )


def _law_tab(M: pd.DataFrame):
    """The power-duration law, then the coverage it was fitted on.

    Renamed from `_measured_tab` when the timelines were given to Evolution:
    what is left on this tab is the relation between watts and duration (the
    law), the year x length-class coverage grid the fit rests on, and the
    year-over-year comparison of one length class. No chart here has time on
    an axis.
    """
    if not len(M):
        callout("No measured efforts", "No session carries a detected-effort "
                "summary.", C["red"], "⚠️")
        return

    section("\U0001F5BA\uFE0F Which type and which year")
    years = sorted(int(y) for y in M["year"].dropna().unique())
    latest = years[-1]

    counts = (M[M["year"] == latest].groupby("tt")
              .agg(n=("w", "size"), cls=("cls", "nunique")).reset_index()
              .sort_values("n", ascending=False))
    counts = counts[counts["tt"].isin(MAIN_TYPES)]
    if not len(counts):
        counts = (M[M["year"] == latest].groupby("tt")
                  .agg(n=("w", "size"), cls=("cls", "nunique"))
                  .reset_index().sort_values("n", ascending=False))
    order = counts["tt"].tolist()
    labels = {r.tt: f"{r.tt} — {r.n:,} detected effort(s), {int(r.cls)} length class(es)"
              for r in counts.itertuples()}

    # Two controls, not three. The FTP / VO2MAX effort filter that used to sit
    # here said nothing the training type does not already say, and on THIS
    # page it was worse than redundant: the x-axis is duration, so a 3-minute
    # effort and a 20-minute effort are two points on one curve, never one
    # number to average. Nothing here mixes durations anyway — every coverage
    # row is one rep length class, and the law is fitted across durations.
    c1, c2 = st.columns([2, 1])
    with c1:
        tt = st.selectbox("Training type", order, index=0,
                          format_func=lambda t: labels[t], key="iw_meas_type",
                          help="One type at a time. Two training types are never "
                               "plotted on the same axis, because that is a "
                               "comparison this page does not make.")
    with c2:
        yr = st.selectbox("Year", years, index=len(years) - 1, key="iw_meas_year",
                          help="Only years carrying detected efforts exist here. "
                               "A year with too few efforts per cell will show its "
                               "cells as present-but-thin rather than as a line.")

    # Every effort of the type: the ley de potencias IS the relation across
    # durations, so it is fitted on every effort of the type, in every year,
    # and the coverage under it shows one row per rep length class.
    M_all = M

    cells = iw.measured_cells(M, yr)
    mine = cells[cells["tt"] == tt].sort_values("med_w", ascending=False)
    if not len(mine):
        callout("Nothing for this type", f"**{tt}** has no detected effort in "
                f"{yr}.", C["orange"], "\U0001F6A7")
        _law_section(M_all, tt, 0, None)
        return

    # The law comes FIRST: this page is named after it. It is fitted on every
    # effort of the type in every year, so the year selector above only
    # describes the coverage tables underneath — never the fit itself.
    M_yr = M[M["year"] == yr] if yr is not None else M
    M_tt = M_yr[M_yr["tt"] == tt]
    _law_section(M_all, tt, int(len(M_tt)), None)

    # Coverage, so the reader sees the shape of the data the law was fitted on.
    grid = _cells_datable(mine, [yr], "Rep length class")
    if len(grid):
        st.markdown(f"**Detected efforts per rep length class, {yr}** — bold "
                    f"clears the {iw.MIN_CELL_N}-effort floor, `·n` is present but "
                    f"too thin to draw, `—` is an empty cell.")
        dataframe(grid, height=260)

    thin = mine[~mine["drawable"]]
    if len(thin):
        st.caption(f"{len(thin)} class(es) hold efforts but stay below the floor "
                   f"and are left unplotted: "
                   + ", ".join(f"{r.cls} (n={int(r.n)})" for r in thin.itertuples()))

    # ── Where the same efforts are drawn over TIME ────────────────────────────
    # Deliberately not here. The date-axis charts of these efforts (the line
    # per rep length class, one bar per day) belong to Evolution, which is the
    # page that answers "how did this evolve" for the whole app; the Intervals
    # page answers "what does one type at one duration look like". Drawing the
    # same rows a third time is what made this page feel not useful.
    st.caption(
        "**Watts over time live on Evolution**, where they are drawn on a date "
        "axis for the whole history. This page is watts AGAINST duration: the "
        "law, and the coverage it was fitted on."
    )

    # ── Every cell, including the thin ones ────────────────────────────────
    section("\U0001F4CB Every cell, thinnest included")
    rows = []
    for r in mine.itertuples():
        rows.append({
            "Rep length class": r.cls,
            "Detected efforts": f"{int(r.n):,}",
            "Sessions": f"{int(r.sessions):,}",
            "Typical length": fmt_min(r.rep_secs),
            "Median W": iw.fmt_watts(r.med_w),
            "Middle half": f"{iw.fmt_watts(r.q1_w)}–{iw.fmt_watts(r.q3_w)}",
            "Plotted": "yes" if bool(r.drawable) else f"no (below {iw.MIN_CELL_N})",
        })
    dataframe(pd.DataFrame(rows), height=300)

    st.caption(
        "Every length class is listed, including those too thin to plot, so the "
        "absence of a bar is always explained by a count rather than by nothing. "
        "These are efforts a head unit detected, not workouts a coach wrote: on a "
        "long ride the detector also reports the rolling sections it found, which "
        "is why an endurance type shows numbers in the 90 s–5 min column."
    )

    # ── Year over year, when there is more than one year ─────────────────────
    if len(years) > 1:
        section("↔️ The same class across years")
        st.caption(
            f"Detected efforts exist for {' and '.join(str(y) for y in years)} only. "
            f"A year-over-year line is drawn where BOTH years clear the "
            f"{iw.MIN_CELL_N}-effort floor in the same class, and the line breaks "
            f"otherwise rather than being interpolated across the gap."
        )
        allc = iw.measured_cells(M, None)
        allc = allc[allc["tt"] == tt]
        multi = []
        for cls in sorted(set(allc["cls"].astype(str))):
            s = allc[allc["cls"].astype(str) == cls].sort_values("year")
            ok = s[s["drawable"]]
            if len(ok) >= 2:
                multi.append(cls)
        if not multi:
            callout("No class has two comparable years",
                    f"**{tt}** has fewer than {iw.MIN_CELL_N} detected efforts per "
                    f"class in more than one year, so there is no year-over-year "
                    f"line to draw here. The counts are in the table above.",
                    C["orange"], "⚠️")
        else:
            fig2 = go.Figure()
            for cls in multi:
                s = allc[allc["cls"].astype(str) == cls].sort_values("year")
                fig2.add_trace(go.Scatter(
                    x=s["year"], y=s["med_w"], mode="lines+markers", name=cls,
                    line=dict(color=_rep_color(cls), width=2),
                    marker=dict(color=_rep_color(cls), size=8,
                                line=dict(color=C["panel"], width=1)),
                    text=[f"n={int(n):,}" for n in s["n"]],
                    connectgaps=False,
                    hovertemplate="<b>%{fullData.name}</b> · %{x}<br>"
                                  "median %{y:,.0f} W<br>%{text}<extra></extra>",
                ))
            style_figure(fig2, f"{tt} — detected interval watts by year, one line "
                              f"per rep length class"
                              f"<br><sup>only classes with {iw.MIN_CELL_N}+ efforts "
                              f"in two or more years are shown</sup>", H_STD)
            _lane_legend(fig2, multi, min_w=430)
            fig2.update_xaxes(dtick=1, tickformat="d")
            fig2.update_yaxes(tickformat=",.0f")
            show(fig2)


def prescribed_tab(P: pd.DataFrame):
    """The coach's target watts. Lower end of the range, year over year.

    Rendered from the **Evolution** page since the page split: prescribed
    watts are a target that evolved over the years, so they belong with the
    charts that answer "how did this change", not with the power-duration law.
    It still lives here because the parser, the coverage numbers and the
    never-join-with-measured rule all live here too.
    """
    if not len(P):
        callout("No prescriptions", "No description in this file carries a "
                "parseable interval target.", C["red"], "⚠️")
        return

    last_year = int(P["year"].max())
    section("\U0001F5BA\uFE0F Which type")
    counts = (P.groupby("tt").agg(n=("w_lo", "size"),
                                  years=("year", "nunique"),
                                  classes=("cls", "nunique")).reset_index()
              .sort_values(["years", "n"], ascending=False))
    order = counts["tt"].tolist()
    labels = {r.tt: f"{r.tt} — {r.n:,} session(s), {int(r.years)} year(s), "
                   f"{int(r.classes)} length class(es)"
              for r in counts.itertuples()}

    tt = st.selectbox("Training type", order, index=0,
                      format_func=lambda t: labels[t], key="iw_pres_type",
                      help="Types are listed with the most years first. Every type "
                           "with a prescription is selectable, even if only one "
                           "year has enough sessions to plot.")

    cells = iw.prescribed_cells(P, tt)
    if not len(cells):
        callout("Nothing for this type", f"**{tt}** has no parsed prescription.",
                C["orange"], "⚠️")
        return

    years = sorted(int(y) for y in cells["year"].unique())
    grid = _cells_datable(cells, years, "Rep length class")
    if len(grid):
        st.markdown(f"**Sessions per rep length class and year** — bold clears the "
                    f"{iw.MIN_CELL_N}-session floor, `·n` is present but too thin "
                    f"to draw, `—` is an empty cell.")
        dataframe(grid, height=300)

    thin = cells[(~cells["drawable"]) & (cells["n"] > 0)]
    if len(thin):
        st.caption(f"{len(thin)} cell(s) hold sessions but fall below the floor and "
                   f"are left blank: "
                   + ", ".join(f"{r.cls} {int(r.year)} (n={int(r.n)})"
                               for r in thin.itertuples()))

    if len(years) < 2:
        callout("Only one year of prescriptions for this type",
                f"**{tt}** has parsed prescriptions in {years[0]} only, so there is "
                f"no year-over-year line. One point is not a trend.",
                C["orange"], "⚠️")
        return

    # ── Evolution, one line per rep length class ────────────────────────────
    section("\U0001F4CA Prescribed target watts by year")
    ev = iw.prescribed_evolution(P, tt)
    if not len(ev):
        callout("Nothing clears the floor",
                f"No **{tt}** cell reaches {iw.MIN_CELL_N} sessions in any year.",
                C["orange"], "⚠️")
        return

    fig = go.Figure()
    classes = [c for c in iw.REP_ORDER if c in set(ev["cls"].astype(str))]
    for cls in classes:
        g = ev[ev["cls"].astype(str) == cls].sort_values("year")
        custom = [[f"{iw.fmt_watts(r.med_w_lo)} W", f"n = {int(r.n):,} session(s)",
                   f"target range up to {iw.fmt_watts(r.med_w_hi)} W",
                   f"middle half: {iw.fmt_watts(r.q1_w_lo)}–{iw.fmt_watts(r.q3_w_lo)} W",
                   f"typical rep {fmt_min(r.rep_secs)}",
                   r.dur_mix]
                  for r in g.itertuples()]
        fig.add_trace(go.Scatter(
            x=g["year"], y=g["med_w_lo"], mode="lines+markers", name=cls,
            line=dict(color=_rep_color(cls), width=2),
            marker=dict(color=_rep_color(cls), size=8,
                        line=dict(color=C["panel"], width=1)),
            customdata=custom, connectgaps=False,
            hovertemplate=f"<b>{cls}</b> · %{{x}}<br>"
                          "lower end of target %{customdata[0]}<br>"
                          "%{customdata[1]}<br>%{customdata[2]}<br>"
                          "%{customdata[3]}<br>rep %{customdata[4]}<br>"
                          "sessions %{customdata[5]}<extra></extra>",
        ))

    style_figure(
        fig,
        f"{tt} — prescribed target watts by year, one line per rep length class"
        f"<br><sup>lower end of the range the coach wrote, median across the "
        f"sessions in that cell. The line breaks wherever a year has fewer than "
        f"{iw.MIN_CELL_N} sessions, and it stops at {last_year} because that is "
        f"where the comments stop.</sup>",
        H_STD)
    _lane_legend(fig, classes, min_w=430)
    fig.update_xaxes(dtick=1, tickformat="d")
    fig.update_yaxes(tickformat=",.0f")
    show(fig)

    st.caption(
        f"Median of the LOWER end of the wattage range the coach wrote for each "
        f"rep length class, n on every point. A length class is never averaged "
        f"with another. This series ends at **{last_year}** — the two sources "
        f"never join: these coach comments stop there while the detected "
        f"efforts begin in 2025, and extending this line would mean drawing "
        f"where no prescription exists. The numbers behind that wall are on "
        f"the **Power law** page."
    )

    # ── Compared with the previous comparable year ──────────────────────────
    section("↔️ Compared with the previous comparable year")
    st.caption(
        "The previous year is the last one in the SAME rep length class that "
        "cleared the floor — not simply the year before. Where the gap is more "
        "than one year it is stated, so a three-year change is never printed as a "
        "one-year change. A delta is left blank unless both years clear the floor."
    )
    yoy = iw.prescribed_yoy(P, tt)
    out = []
    for r in yoy.itertuples():
        prev = "—" if pd.isna(r.prev_year) else f"{int(r.prev_year)}"
        pn = "—" if pd.isna(r.prev_n) else f"{int(r.prev_n):,}"
        pv = "—" if pd.isna(r.prev_med) else iw.fmt_watts(r.prev_med)
        gap = "—" if pd.isna(r.gap_years) else f"{int(r.gap_years)}y"
        delta = "—" if pd.isna(r.delta) else (
            f"+{iw.fmt_watts(r.delta)} W" if r.delta > 0
            else f"{iw.fmt_watts(r.delta)} W")
        out.append({
            "Rep length class": r.cls,
            "Year": int(r.year),
            "Sessions": f"{int(r.n):,}",
            "Typical rep": fmt_min(r.rep_secs),
            "Median target (W)": iw.fmt_watts(r.med_w_lo),
            "Compared with": prev,
            "Sessions then": pn,
            "Median then (W)": pv,
            "Change": delta,
            "Gap": gap,
        })
    if out:
        dataframe(pd.DataFrame(out), height=340)
    else:
        st.info("No cell for this type cleared the floor in any year.")

    st.caption(
        "The change column is a difference between two medians of the same rep "
        "length class. It is descriptive: it does not say the coach raised the "
        "target because of a result, only that both numbers were written down."
    )


def _audit_tab(a: dict, cov: dict, P: pd.DataFrame, M: pd.DataFrame):
    section("\U0001F9EA What the parser could and could not read")

    # The note window comes FIRST: this file's coach notes stop on their own,
    # and a reader has to see that before an empty recent cell is read as a
    # parse failure. It is neither.
    if a.get("note_last") is not None:
        st.markdown(
            f"**Coach notes run {a['note_first']:%d %b %Y} → "
            f"{a['note_last']:%d %b %Y}.** After that date "
            f"**{a['rides_after_last_note']:,}** session(s) carry no note at "
            "all, so there is nothing there to read — a blank cell in those "
            "years is absent data, not a refusal."
        )

    st.markdown(
        f"Of **{a['seen']:,}** coach descriptions in this file:\n\n"
        f"- **{a['parsed']:,}** gave an unambiguous interval target and are used "
        f"({a['reads_direct']:,} written straight after the rep, "
        f"{a['reads_inside']:,} read off a sub-length inside the rep, "
        f"{a['reads_ceiling']:,} written as a ceiling such as `per sota 265w` "
        f"and charted at that ceiling rather than invented).\n"
        f"- **{a['skipped_steady']:,}** carry no repetition at all — a steady "
        f"endurance ride, which has no intervals by definition. That is a correct "
        f"non-match, not a loss.\n"
        f"- **{a['skipped_unread']:,}** do contain a repetition and still gave "
        f"nothing usable. They are split below, because 'refused' hides three "
        f"different facts.\n"
        f"- **{a['skipped_unlabelled']:,}** parsed but sit on a session with no "
        f"training type, so they cannot be attributed to a type."
    )

    st.markdown(
        f"- **{a['refused_no_watts']:,}** have a repetition but no wattage "
        "beside it: the target is in pedal strokes (`+165pols`), in a speed "
        "(`10k/h`), or is simply `MÀX`. Refused rather than guessed.\n"
        f"- **{a['refused_too_short']:,}** do have a wattage, but the rep is "
        f"under the 30-second interval floor (10 s and 15 s sprints). That is a "
        "stated rule, not a parser failure — the watts are there and readable.\n"
        f"- **{a['refused_implausible']:,}** have a wattage under 100 W, which "
        "no interval of this athlete's has ever been.\n"
        f"- **{a['refused_ambiguous']:,}** have wattage that would have to be "
        "attributed to a rep by guessing at the structure."
    )

    if a.get("examples_unread"):
        st.markdown("**Examples refused, verbatim.** These are the descriptions a "
                    "reader is most likely to expect to see on the chart:")
        for ex in a["examples_unread"]:
            st.code(ex, language=None)
        st.caption(
            "The reason each one is refused is counted above: no wattage, "
            "under the 30-second floor, or a structure that could not be "
            "attributed without invention. Reading a number out of any of them "
            "would be invention."
        )

    section("\U0001F4D6 Source coverage")
    rows = []
    if cov["prescribed_years"]:
        rows.append({
            "Source": "Coach's written comment",
            "Column": "WorkoutDescription",
            "Years": ", ".join(str(y) for y in cov["prescribed_years"]),
            "Rows": f"{cov['prescribed_n']:,}",
            "What the number is": "lower end of a target watt range",
        })
    if cov["measured_years"]:
        rows.append({
            "Source": "Detected efforts",
            "Column": "interval_summary",
            "Years": ", ".join(str(y) for y in cov["measured_years"]),
            "Rows": f"{cov['measured_n']:,}",
            "What the number is": "measured watts of a detected effort",
        })
    if rows:
        dataframe(pd.DataFrame(rows), height=200)

    with st.expander("How these numbers were built"):
        st.markdown(
            f"- **Grain:** one training type at a time, one rep length class at a "
            f"time. No chart here puts two types on one axis and no cell averages "
            f"two rep lengths together.\n"
            f"- **Two sources, never joined.** The coach's comments run "
            f"{cov['prescribed_years'][0] if cov['prescribed_years'] else '—'}–"
            f"{cov['prescribed_last_year'] or '—'}; the detected efforts run "
            f"{cov['measured_years'][0] if cov['measured_years'] else '—'}–"
            f"{cov['measured_last_year'] or '—'}. They meet in one year and nowhere "
            f"else, so they sit on separate pages and no line crosses between "
            f"them — the prescriptions on Evolution, the measured efforts here.\n"
            f"- **Lower end of the range** is used for a prescribed target, as "
            f"instructed — the end the coach was willing to call a success. The "
            f"upper end is carried alongside and shown in the hover rather than "
            f"discarded.\n"
            f"- **Where a number came from is counted, never assumed.** A target "
            f"written straight after the rep is read directly. A target written "
            f"on a sub-length inside the rep (`2x20' terreny constant "
            f"(4'265-280w+1' suau x4 cops)`) is read from that sub-length, and "
            f"only when the sub-length fits inside the rep. A ceiling (`per "
            f"sota 265w`, `sense passar 200w`) is charted at the ceiling and "
            f"said to be one. Everything else is refused.\n"
            f"- **Minutes and seconds are different marks.** In these comments "
            f"`15'` is fifteen minutes and `30\"` is thirty seconds. Reading the "
            f"second as the first would inflate a sprint session sixtyfold.\n"
            f"- **Under 90 s keeps its seconds.** That class is not subdivided, "
            f"because at that scale the exact length IS the description. The exact "
            f"rep length is printed next to every median.\n"
            f"- **Floor:** a point needs {iw.MIN_CELL_N} rows in its cell. Below "
            f"that the point is absent and the line breaks — never interpolated, "
            f"never zero, never carried forward.\n"
            f"- **n is always on** the point, the bar and the table row it belongs "
            f"to.\n"
            f"- **`PowerMax` is deliberately absent.** It is populated for 902 "
            f"sessions and looks like interval power, but it is session PEAK — "
            f"median 553 W — a sprint spike, not an interval.\n"
            f"- **Labels are the athlete's own.** Nothing here is classified by "
            f"intensity or by similarity; a session only appears under a type the "
            f"athlete filed it under.\n"
            f"- **Descriptive only.** These are observed numbers and written "
            f"targets. Nothing here claims a prescription caused a result."
        )
