# views/evolution.py — Evolution: one training type, year over year. (NEW FILE)
"""
Answers "did this training type change, and how does it compare with the
previous year?" WITHOUT ever mixing two training types or two duration classes
in the same number.

The page is built coverage-first. Before any chart is drawn it shows how many
sessions sit in every (duration class x year) cell and marks which cells clear
the floor, so a reader can see the shape of their own data and judge the chart
rather than being handed a confident line through two points.

Data source is ctx.df_all, deliberately. The sidebar range filter defaults to
"Last 6 months"; applying it here would delete 2019-2025 and leave a page whose
entire subject has been filtered away. The range is disclosed in the header
instead of being silently ignored.

The outcome side is stated as a wall, not a gap. Measured threshold watts begin
in Dec 2025, so there is no year-over-year watts comparison for 2019-2025 to
make — this page says so with the counts that prove it, rather than leaving the
absence to be noticed.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.components import (callout, dataframe, legend, metric_card, page_header,
                             section, show)
from core.theme import C, H_PAIR, H_STD, MAIN_TYPES, style_figure
from ml import interval_watts as iw
from ml import year_over_year as yoy
from views.intervals_view import _lane_legend

# One colour per DURATION CLASS, fixed by position so the same class is the
# same colour on every chart on every type. Assigned by hand from the theme
# palette rather than generated, so it stays stable across sessions.
DUR_COLORS = {
    "under 45 min": C["yellow"],
    "45-90 min": C["teal"],
    "90-150 min": C["accent"],
    "150-240 min": C["purple"],
    "240+ min": C["orange"],
}


def _dur_color(dcls: str) -> str:
    return DUR_COLORS.get(dcls, C["muted"])


def _present(cells: pd.DataFrame) -> str:
    """One line per duration class: '·' when the cell exists but is too thin."""
    n = int(cells["n"]) if pd.notna(cells["n"]) else 0
    if bool(cells["drawable"]):
        return f"**{n}**"
    return f"·{n}" if n else "—"


def _day_figure(S: pd.DataFrame, tt: str, cls: str) -> go.Figure:
    """Bars of interval watts by day — watts and exact length on every bar.

    Kept separate from the page so the labels and the axis floor can be
    asserted without a Streamlit session: everything a reader must not have to
    estimate from geometry is printed on the chart itself.
    """
    # Equal-width slot per day: on a true date axis a 7-day gap gets one thin
    # sliver and the eye can't read it. Every charted day gets the same width,
    # in chronological order, with its date printed under the bar.
    x_slots = [str(pd.Timestamp(d).date()) for d in S["day"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_slots, y=S["w"], name="Interval watts",
        marker_color=C["accent"], opacity=0.9,
        # Watts first (top line), exact length under it: the reader tracks the
        # evolution off a printed number and still never loses how long that
        # number was held for.
        text=[f"{iw.fmt_watts(w)} W\n{iw.fmt_rep(s)}"
              for w, s in zip(S["w"], S["secs"])],
        textposition="outside", textfont=dict(color=C["muted"], size=10),
        customdata=[[iw.fmt_rep(r.secs), iw.fmt_watts(r.w),
                     iw.fmt_rep(r.dur_s)]
                    for r in S.itertuples()],
        hovertemplate="<b>%{x}</b><br>"
                      "interval %{customdata[1]} W over %{customdata[0]}<br>"
                      "session length %{customdata[2]}<extra></extra>",
    ))
    # Line showing average interval watts across days for comparison
    if len(S) > 0:
        mean_w = float(S["w"].mean())
        fig.add_trace(go.Scatter(
            x=x_slots, y=[mean_w] * len(S), mode="lines",
            name="Avg interval watts",
            line=dict(color=C["yellow"], width=2, dash="dot"),
            connectgaps=False,
            hovertemplate="<b>%{x}</b><br>"
                          f"avg interval watts {iw.fmt_watts(mean_w)} W<extra></extra>",
        ))

    # The day's OWN average, drawn as a line. Bars are read as comparisons, a
    # line is read as evolution — "what did 30 Sep and 22 Sep average, and
    # which way is it going" is a question bars answer badly. One point per
    # charted day, averaged only inside this length class (a 7:30 rep never
    # meets a 20:00 one), and never merged into a bar: the bar stays the
    # hardest effort of the day. Exact value of every point is printed in the
    # table under the chart, so nothing has to be taken off the geometry.
    day_avg = (pd.to_numeric(S["day_avg"], errors="coerce")
               if "day_avg" in getattr(S, "columns", []) else None)
    if day_avg is not None and day_avg.notna().any():
        fig.add_trace(go.Scatter(
            x=x_slots, y=day_avg, mode="lines+markers",
            name="Day average watts",
            line=dict(color=C["green"], width=2),
            marker=dict(size=7, color=C["green"]),
            connectgaps=False,
            hovertemplate="<b>%{x}</b><br>"
                          "day average %{y:.0f} W<extra></extra>",
        ))

    # Cut the axis just under the lowest bar: a zero baseline plus headroom
    # turns 211 -> 244 W into nine near-identical columns, which is exactly
    # when a reader stops seeing the evolution. The floor is printed in the
    # subtitle and the watts are printed on every bar, so the truncation
    # informs the eye instead of flattering the chart. Flat data is NOT zoomed
    # into: when the bars barely differ the axis stays wide and the numbers do
    # the talking.
    lo_w, hi_w = float(S["w"].min()), float(S["w"].max())
    span_w = hi_w - lo_w
    if span_w >= max(hi_w * 0.08, 5.0):
        y_min = max(0.0, lo_w - span_w * 0.30)
        y_max = hi_w + span_w * 0.45 + 8.0     # headroom for the outside label
    else:
        y_min = max(0.0, lo_w - max(hi_w * 0.12, 10.0))
        y_max = hi_w + max(hi_w * 0.12, 10.0) + 8.0

    style_figure(
        fig,
        f"{tt} — interval watts by day, {cls} class"
        "<br><sup>one equal-width slot per charted day, in date order; the label "
        "above each bar is that interval's watts and its exact length. The "
        "green line is each day's OWN average watts for this class, drawn so "
        "the evolution can be read as a line rather than off bar heights. The "
        "dotted line is the average of the interval watts across the charted "
        "days and is never averaged into the bars. The y-axis is cut just under "
        f"the lowest bar rather than starting at zero ({iw.fmt_watts(y_min)} W "
        "is the floor), so day-to-day differences stay visible — read the "
        "printed watts, not the ratio of two bar heights. Where two sessions "
        "share a day the harder one is shown. Observed history, not a cause."
        "</sup>",
        H_STD,
    )
    _lane_legend(fig, [t.name for t in fig.data], min_w=430)
    fig.update_layout(bargap=0.3)
    fig.update_xaxes(type="category", tickangle=-45, automargin=True)
    fig.update_yaxes(tickformat=",.0f", range=[y_min, y_max], autorange=False)
    fig._y_floor = y_min          # noqa: B010 — asserted by tests, not rendered
    return fig


def _interval_by_day(ctx):
    """Bars of INTERVAL watts by day, with the session average as a line.

    The interval is the subject. The bars are the watts of one sustained effort
    on one day; the line is the whole-session average for that same day and
    exists only so the two can be read against each other. The line is never
    averaged into a bar and never substituted for one.

    Source is the peak-meter reading (`icu_pm_ftp_watts` with its own
    `icu_pm_ftp_secs`), not the auto-detected `interval_summary`. Measured on
    this file the detector reports what is UNUSUAL inside a ride, so on a real
    interval session it fills with 10-second accelerations and 1-5 minute rolling
    sections, and exactly one detected effort in the whole file sits between 19
    and 21 minutes. The peak-meter has a reading on every FTP session of 2026
    and reports the length it was sustained over.

    Because that length varies from 5 to 55 minutes across sessions, one type is
    NOT enough on its own: a 258 W effort over 8:00 and a 188 W effort over
    26:00 are not the same thing. So a length class is required too, and both
    the watts and the exact length are written on every bar, because the bar
    geometry alone cannot carry a 211 -> 244 W difference the eye can trust.
    """
    df = ctx.df_all
    section("\U0001F3AF Intervals by day — watts of the interval")
    st.caption(
        "**One bar per day: the watts of that day's interval.** The green line "
        "is each day's own average watts for this class — the number to follow "
        "over time, with its exact value printed in the table below. The dotted "
        "line is the average of the interval watts across the charted days, so "
        "each day can be compared to the average of intervals themselves. One "
        "training type and one length class at a time, so two different "
        "intervals never share a bar."
    )

    opts = iw.day_options(df)
    if opts is None or not len(opts):
        callout("No interval records",
                "This file carries no sustained-effort reading, so there is no "
                "interval to chart by day.", C["red"], "⚠️")
        return

    tsum = (opts.groupby("tt", as_index=False)[["days", "sessions"]].sum()
            .sort_values("days", ascending=False))
    t_opts = tsum["tt"].tolist()
    t_lab = {r.tt: f"{r.tt} — {int(r.days):,} day(s), "
                  f"{int(r.sessions):,} session(s)"
           for r in tsum.itertuples()}

    cls_order = [c for c in iw.REP_ORDER]

    def _cls_list(tt: str):
        sub = opts[opts["tt"] == tt]
        have = set(sub["cls"].astype(str))
        return [c for c in cls_order if c in have], sub

    c1, c2 = st.columns([2, 3])
    with c1:
        tt = st.selectbox("Training type", t_opts, index=0,
                          format_func=lambda t: t_lab[t], key="evo_day_type",
                          help="One type at a time. Days are the calendar days "
                               "the chart will draw; sessions counts the rides "
                               "behind them, which is higher where one day held "
                               "more than one.")
    c_opts, csub = _cls_list(tt)
    with c2:
        c_lab = {r.cls: f"{r.cls} — {int(r.days):,} day(s), "
                        f"median {iw.fmt_watts(r.med_w)} W over "
                        f"{iw.fmt_rep(r.med_secs)}"
               for r in csub.itertuples()}
        cls = st.selectbox("Interval length", c_opts or ["—"], index=0,
                           format_func=lambda c: c_lab.get(c, c),
                           key="evo_day_cls",
                           help="Intervals of different lengths are never mixed. "
                                "A whole 20:00 sits in '20-30 min'; every bar and "
                                "row below also shows its exact length.")
        if not c_opts:
            callout("Nothing to chart", f"No {tt} interval length is recorded.",
                    C["orange"], "⚠️")
            return

    S = iw.day_series(df, tt, cls)
    if S is None or not len(S):
        callout("Nothing to chart", f"No {tt} day carries a {cls} interval.",
                C["orange"], "⚠️")
        return

    has_avg = bool(S["avg_w"].notna().any())
    mc = st.columns(4)
    with mc[0]:
        metric_card("Days charted", f"{len(S):,}",
                    f"{S['day'].min().date()} → {S['day'].max().date()}", "muted")
    with mc[1]:
        metric_card("Median interval", f"{iw.fmt_watts(S['w'].median())} W",
                    f"{iw.fmt_rep(S['secs'].median())} efforts", "accent")
    with mc[2]:
        metric_card("Best day", f"{iw.fmt_watts(S['w'].max())} W",
                    str(S.loc[S['w'].idxmax(), "day"].date()), "green")
    with mc[3]:
        metric_card("Avg interval watts",
                    f"{iw.fmt_watts(S['w'].mean())} W",
                    "mean of interval watts across days", "muted")

    show(_day_figure(S, tt, cls))

    st.caption(
        f"{len(S):,} day(s) of **{int(S['n_sessions'].sum()):,}** session(s). "
        f"Each bar is one sustained interval of the {cls} class — its watts, "
        "printed over the bar, with the exact length it was held under them. "
        "**Day avg** is that day's own average across its intervals of this "
        "class, which is what the green line plots. Exact lengths are listed "
        "below."
    )

    rows = []
    for r in S.itertuples():
        rows.append({
            "Date": str(pd.Timestamp(r.day).date()),
            "Interval (W)": iw.fmt_watts(r.w),
            "Day avg (W)": iw.fmt_watts(getattr(r, "day_avg", float("nan"))),
            "Length": iw.fmt_rep(r.secs),
            "Session length": iw.fmt_rep(r.dur_s),
            "Sessions that day": int(r.n_sessions),
        })
    dataframe(pd.DataFrame(rows), height=300)

    with st.expander("What these watts are, and are not"):
        st.markdown(
            f"- **Source:** the peak-meter reading recorded per session, with "
            f"the window it was sustained over. This is the head unit's best "
            f"sustained average, which is a real measurement of something real.\n"
            f"- **Not a prescription.** The coach's written target watts come "
            f"from the session comments and stop at 2025 — there are none in "
            f"2026, so nothing here is matched against an intended wattage.\n"
            f"- **Not a threshold test.** Nothing on this page is a validated "
            f"measurement of FTP. A 20-minute effort is a real 20-minute effort "
            f"and nothing more.\n"
            f"- **Never mixed.** One training type and one length class at a "
            f"time. A {cls} interval is never averaged with any other length.\n"
            f"- **n is always visible** — in the picker, in the caption, and in "
            f"the table."
        )


def render(head, ctx):
    page_header(
        "\U0001F4C8",
        "Evolution — the intervals, then the years",
        "The first tab is the interval itself: watts of one rep, one length "
        "class at a time, plotted by day. The line shows the average interval "
        "watts across those days. The second tab compares whole training types "
        "across years. Same type, same length class, never mixed anywhere.",
    )

    # The sidebar range is NOT applied here, on purpose — see the module note.
    st.caption(
        f"Full history, {ctx.df_all['date'].min().date()} → "
        f"{ctx.df_all['date'].max().date()} · **{len(ctx.df_all):,} sessions**. "
        f"The sidebar range filter does not apply to this page: it defaults to "
        f"6 months, which would remove every year being compared."
    )

    t_days, t_years = st.tabs([
        "Intervals by day",
        "Year over year — every measure",
    ])

    with t_days:
        _interval_by_day(ctx)

    with t_years:
        s = yoy.prep(ctx.df_all)
        if not len(s):
            callout("No labelled sessions", "No session carries a training type yet.",
                    C["red"], "⚠️")
            return

        scope = yoy.types_in_scope(s, MAIN_TYPES)
        if not len(scope):
            callout("No training types", "None of the 11 training types has sessions.",
                    C["red"], "⚠️")
            return

        # ── Type + metric pickers ───────────────────────────────────────────────
        section("\U0001F5BA\uFE0F What can be compared")

        labels = {r.training_type: f"{r.training_type} — {r.n_sessions:,} sessions, "
                                   f"{r.years} comparable year(s)"
                   for r in scope.itertuples()}
        order = scope["training_type"].tolist()
        default_idx = 0
        c1, c2 = st.columns([2, 2])
        with c1:
            tt = st.selectbox("Training type", order, index=default_idx,
                              format_func=lambda t: labels[t],
                              key="evo_type",
                              help="One type at a time. Types are listed with the most "
                                   "comparable years first; a type with too few cells is "
                                   "still selectable so the gap can be seen.")
        sm = yoy.summary(s, tt)
        with c2:
            cov = yoy.coverage(s, tt)
            # Only offer metrics that have at least one drawable cell for THIS type,
            # so the picker cannot offer a chart that will render nothing.
            usable = [k for k in yoy.METRIC_KEYS if len(yoy.evolution(s, tt, k))]
            metric = st.selectbox(
                "Measure", usable or ["power_avg"],
                index=0,
                format_func=lambda k: yoy.METRIC_BY_KEY[k][1],
                key="evo_metric",
                disabled=not usable,
                help="Median across the sessions in each cell. A cell needs at least "
                     f"{yoy.MIN_CELL_N} sessions carrying the measure to be drawn.",
            ) if usable else "power_avg"

        _, mlabel, munit, mdec, _ = yoy.METRIC_BY_KEY[metric]
        unit_txt = f" {munit}" if munit else ""

        mc = st.columns(4)
        with mc[0]:
            metric_card("Sessions of this type", f"{sm['n_sessions']:,}")
        with mc[1]:
            metric_card("Comparable cells", f"{sm['n_drawable']:,}",
                        f"of {sm['n_cells']:,} cells", "muted")
        with mc[2]:
            yrs = sm["years_drawable"]
            metric_card("Years with data", f"{len(yrs)}" if yrs else "0",
                        f"{yrs[0]}–{yrs[-1]}" if len(yrs) > 1 else
                        (f"{yrs[0]}" if yrs else "none"), "muted")
        with mc[3]:
            metric_card("Threshold watts", "Not comparable",
                        f"measured from {yoy.watts_gate(s).get('first_month') or '—'}",
                        "red")

        if not sm["comparable"]:
            callout(
                "Not enough to compare years",
                f"**{tt}** has sessions in {len(sm['years_drawable'])} year(s) that clear "
                f"the {sm['min_n']}-session floor per duration class. A year-over-year "
                "line needs at least two. Nothing is drawn, because a line through "
                "one point is not a trend.",
                C["orange"], "⚠️")
            return

        # ── Coverage grid ───────────────────────────────────────────────────────
        years = sorted(cov["year"].unique().tolist())
        grid = cov.pivot(index="dur_class", columns="year", values=["n", "drawable"])
        rows = []
        for dcls in yoy.DUR_ORDER:
            sub = cov[cov["dur_class"] == dcls]
            if not len(sub):
                continue
            by_year = sub.set_index("year")
            rec = {"Duration class": dcls}
            for y in years:
                rec[str(y)] = _present(by_year.loc[y]) if y in by_year.index else "—"
            tot = int(sub["n"].sum())
            rec["Total"] = f"{tot:,}"
            rows.append(rec)
        if rows:
            st.markdown("**Sessions per duration class and year** — bold clears the "
                        f"{yoy.MIN_CELL_N}-session floor, `·n` is present but too thin "
                        "to draw, `—` is an empty cell.")
            dataframe(pd.DataFrame(rows), height=200)

        thin = cov[(~cov["drawable"]) & (cov["n"] > 0)]
        if len(thin):
            st.caption(
                f"{len(thin)} cell(s) hold sessions but fall below the floor and are "
                f"left blank: "
                + ", ".join(f"{r.dur_class} {int(r.year)} (n={int(r.n)})"
                            for r in thin.itertuples())
            )

        # ── The evolution chart ─────────────────────────────────────────────────
        section("\U0001F4CA Evolution")

        ev = yoy.evolution(s, tt, metric)
        if not len(ev):
            callout("Nothing to draw", f"No {tt} cell clears the floor for "
                                      f"{mlabel.lower()}.", C["orange"], "⚠️")
            return

        fig = go.Figure()
        classes = [d for d in yoy.DUR_ORDER if d in set(ev["dur_class"].astype(str))]
        for dcls in classes:
            g = ev[ev["dur_class"].astype(str) == dcls].sort_values("year")
            custom = [
                [int(r.year),
                 f"{yoy.fmt_value(metric, r.value)}{unit_txt}",
                 f"n = {int(r.n):,} sessions",
                 f"middle half: {yoy.fmt_value(metric, r.q1)}–{yoy.fmt_value(metric, r.q3)}{unit_txt}"]
                for r in g.itertuples()
            ]
            fig.add_trace(go.Scatter(
                x=g["year"], y=g["value"], mode="lines+markers", name=dcls,
                line=dict(color=_dur_color(dcls), width=2),
                marker=dict(color=_dur_color(dcls), size=8,
                            line=dict(color=C["panel"], width=1)),
                customdata=custom, connectgaps=False,
                hovertemplate="<b>%{fullData.name}</b> · %{x}<br>"
                              "median %{customdata[1]}<br>%{customdata[2]}<br>"
                              "%{customdata[3]}<extra></extra>",
            ))

        # Gaps are already handled by connectgaps=False: a thin year has no point,
        # so the line breaks there instead of being interpolated across it.
        style_figure(
            fig,
            f"{tt} — {mlabel.lower()} by year, one line per duration class"
            f"<br><sup>each point is the median of that cell; the line breaks wherever "
            f"a year has fewer than {yoy.MIN_CELL_N} sessions. Descriptive only — "
            f"observed history, not a cause.</sup>",
            H_STD,
        )
        _lane_legend(fig, classes, min_w=430)
        fig.update_xaxes(dtick=1, tickformat="d")
        fig.update_yaxes(tickformat=",.0f" if mdec == 0 else None)
        show(fig)

        st.caption(
            f"Median {mlabel.lower()} per cell, n on every point. A duration class is "
            "never averaged with another, and no line is drawn through a year that "
            "does not have the sessions to support it."
        )

        # ── Volume by year ──────────────────────────────────────────────────────
        section("\U0001F9EE How much of it was done")
        st.caption("Sessions per year by duration class — how the type was actually "
                   "used, including the years too thin to plot a median.")
        vols = (s[s["tt"] == tt].groupby(["year", "dur_class"]).size()
                .rename("n").reset_index())
        fig_v = go.Figure()
        for dcls in [d for d in yoy.DUR_ORDER if d in set(vols["dur_class"].astype(str))]:
            g = vols[vols["dur_class"].astype(str) == dcls].sort_values("year")
            fig_v.add_trace(go.Bar(
                x=g["year"], y=g["n"], name=dcls,
                marker_color=_dur_color(dcls), opacity=0.85,
                text=[f"{int(v):,}" if v >= 1 else "" for v in g["n"]],
                textposition="outside", textfont=dict(color=C["muted"], size=10),
                hovertemplate="<b>%{fullData.name}</b> · %{x}<br>%{y:,} sessions<extra></extra>",
            ))
        fig_v.update_layout(barmode="stack")
        style_figure(fig_v, f"{tt} — sessions per year by duration class"
                     "<br><sup>counts every session, however few; a bar of 1 is still "
                     "a ride that happened</sup>", H_STD)
        _lane_legend(fig_v, [t.name for t in fig_v.data], min_w=430)
        fig_v.update_xaxes(dtick=1, tickformat="d")
        fig_v.update_yaxes(tickformat=",.0f")
        show(fig_v)

        # ── Compared with the previous comparable year ──────────────────────────
        section("↔️ Compared with the previous comparable year")
        st.caption(
            "The previous year is the last one in the SAME duration class that cleared "
            "the floor — not simply the year before. Where the gap is more than one "
            "year it is stated, so a two-year change is never printed as a one-year "
            "change. A delta is left blank unless both years have the sessions to "
            "support it."
        )

        tbl = yoy.yoy_table(s, tt, metric)
        shown = tbl[tbl["year"].isin(years)].copy()
        out = []
        for r in shown.itertuples():
            if not r.drawable:
                continue
            prev_yr = "—" if pd.isna(r.prev_year) else f"{int(r.prev_year)}"
            prev_n = "—" if pd.isna(r.prev_n) else f"{int(r.prev_n):,}"
            prev_v = "—" if pd.isna(r.prev_value) else f"{yoy.fmt_value(metric, r.prev_value)}"
            gap = "—" if pd.isna(r.gap_years) else f"{int(r.gap_years)}y"
            delta = yoy.fmt_delta(metric, r.delta)
            out.append({
                "Duration class": r.dur_class,
                "Year": int(r.year),
                "Sessions": f"{int(r.n):,}",
                f"Median{' (' + munit + ')' if munit else ''}": yoy.fmt_value(metric, r.value),
                "Compared with": prev_yr,
                "Sessions then": prev_n,
                "Median then": prev_v,
                "Change": delta,
                "Gap": gap,
            })
        if out:
            dataframe(pd.DataFrame(out), height=320)
        else:
            st.info("No cell for this type cleared the floor in any year.")

        med_exact = tbl[tbl["drawable"]].copy()
        if len(med_exact):
            st.caption("Exact session length behind each cell's median — minutes shown "
                       "whole, seconds always visible. Only cells that cleared the "
                       "floor are listed, so no row here rests on a single ride.")
            # Restrict to the (duration class, year) pairs that were actually drawn.
            # Listing every year would put a 1-session median length next to a
            # 37-session one and let the reader assume they were comparable.
            keys = set(zip(med_exact["dur_class"].astype(str),
                           med_exact["year"].astype(int)))
            base = s[s["tt"] == tt]
            ex = (base.groupby(["dur_class", "year"])
                  .agg(dur_s=("dur_s", "median"), n=("dur_s", "size"))
                  .reset_index())
            ex = ex[[(str(r.dur_class), int(r.year)) in keys for r in ex.itertuples()]]
            if len(ex):
                ex["Median length"] = ex["dur_s"].apply(yoy.fmt_dur)
                ex = ex.rename(columns={"dur_class": "Duration class", "year": "Year",
                                        "n": "Sessions"})
                ex = ex[["Duration class", "Year", "Sessions", "Median length"]]
                dataframe(ex.sort_values(["Duration class", "Year"]), height=280)

        # ── The outcome wall ────────────────────────────────────────────────────
        section("\U0001F6A7 Effect on watts — what the data can and cannot answer")
        g = yoy.watts_gate(s)
        if g.get("n_measured"):
            by_year = g["by_year"]
            yr_txt = ", ".join(f"**{y}: {n:,}** reading(s)" for y, n in sorted(by_year.items()))
            callout(
                "Year-over-year threshold watts do not exist",
                f"Threshold watts are measured, not modelled, and the measurement only "
                f"starts in **{g['first_month']}** — {yr_txt}. "
                f"That is **{g['n_months']} months** in total. Every year before "
                f"{min(by_year)} has no threshold number at all, so there is nothing to "
                f"compare 2019–{max(by_year) - 1} against. "
                f"{', '.join(str(y) for y in g['years_without'])} contribute training "
                f"data but no watts outcome.",
                C["red"], "🚧")
            st.caption(
                "This page therefore compares the TRAINING — how long, how hard, how "
                "often, at what temperature — across all eight years, and refuses to "
                "draw a watts line that the measurements do not support. A modelled "
                "proxy was tried here once and was wrong: normalised power is present "
                "for 139 of 139 sessions in 2020 but only 30 of 141 in 2026, so it "
                "stops mid-series for a data reason, not a physiological one."
            )
        else:
            callout("No threshold measurement", "No session carries a peak-meter "
                    "threshold reading.", C["red"], "🚧")

        # ── Disclosure ──────────────────────────────────────────────────────────
        with st.expander("How these numbers were built"):
            st.markdown(
                f"- **Grain:** one training type at a time, one duration class at a "
                f"time. No cell mixes two types or two lengths.\n"
                f"- **Floor:** a year-point needs ≥ {yoy.MIN_CELL_N} sessions in that "
                f"cell *and* ≥ {yoy.MIN_CELL_N} of them carrying the measure. Below "
                f"that the point is absent and the line breaks — never interpolated.\n"
                f"- **Statistic:** the median, with the middle half of the sessions "
                f"(IQR) in the hover, because a median can hide a bimodal cell.\n"
                f"- **n is always shown**, next to the number it belongs to. "
                f"{len(s):,} of {len(ctx.df_all):,} sessions carry a label; the other "
                f"{len(ctx.df_all) - len(s):,} are excluded here rather than guessed "
                f"at.\n"
                f"- **Labels are the athlete's own.** A session is only counted under a "
                f"type the athlete filed it under; nothing is classified by intensity "
                f"automatically.\n"
                f"- **Descriptive only.** These are observed medians. They describe "
                f"what was ridden, not what caused anything."
            )
