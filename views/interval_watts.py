# views/interval_watts.py — interval watts, not average watts. (NEW FILE)
r"""
The request was "isolate not avg watts but the interval watts". This page is that
isolation: average power is not offered here at all, and the only watt figures on
screen come from a rep.

Interval watts exist in two places in this athlete's file, and they cover
different years:

  MEASURED   intervals.icu auto-detected efforts, 2025-2026. Real watts, and the
             only interval data for this year. Tab 1.
  PRESCRIBED the coach's own Spanish/Catalan comments in `WorkoutDescription`,
             2019-2025, and NOTHING in 2026. The watts are a target RANGE and the
             lower end is used, per the athlete's instruction. Tab 2.

They are in separate tabs on purpose. The prescribed comments stop in 2025 and the
measured efforts begin in 2025, so drawing one line through both would mean
interpolating 2025-2026 out of nothing and calling it the athlete's progress. The
tab boundary is the honest rendering of that wall, and the wall is also printed
in numbers so it is not something the reader has to infer from the layout.

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


def render(head, ctx):
    page_header(
        "\U0001F3AF",
        "Interval watts — the watts of the rep",
        "Not average power. Every figure here comes from a single interval, and "
        "intervals of different lengths are never averaged together: a 30-second "
        "effort and a 20-minute effort are separate series with separate counts.",
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
            f"They are shown in separate tabs and are never joined into one line, "
            f"because 'the watts you were told to do' and 'the watts you actually "
            f"did' are different numbers and bridging the gap between them would "
            f"invent a trend that no measurement supports.",
            C["orange"], "🚧")

    tab_meas, tab_pres, tab_audit = st.tabs([
        "\U0001F4CA Measured — this year",
        "\U0001F4DD Prescribed — the coach's target",
        "\U0001F50D What could be read",
    ])

    with tab_meas:
        _measured_tab(M)

    with tab_pres:
        _prescribed_tab(P)

    with tab_audit:
        _audit_tab(a, cov, P, M)


def _measured_tab(M: pd.DataFrame):
    """Real measured interval watts. One type at a time, one series per length."""
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

    cells = iw.measured_cells(M, yr)
    mine = cells[cells["tt"] == tt].sort_values("med_w", ascending=False)
    if not len(mine):
        callout("Nothing for this type", f"**{tt}** has no detected effort in "
                f"{yr}.", C["orange"], "⚠️")
        return

    # Coverage first, so the reader sees the shape of the data before the chart.
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

    # ── The chart ────────────────────────────────────────────────────────────
    section("\U0001F4C8 Measured interval watts")
    drawable = mine[mine["drawable"]]
    if not len(drawable):
        callout("Nothing clears the floor",
                f"No rep length class for **{tt}** in {yr} has "
                f"{iw.MIN_CELL_N} detected efforts.", C["orange"], "⚠️")
    else:
        fig = go.Figure()
        custom = [[f"{iw.fmt_watts(r.med_w)} W",
                   f"n = {int(r.n):,} effort(s)",
                   f"middle half: {iw.fmt_watts(r.q1_w)}–{iw.fmt_watts(r.q3_w)} W",
                   f"typical length {iw.fmt_rep(r.rep_secs)}"]
                  for r in drawable.itertuples()]
        fig.add_trace(go.Bar(
            x=drawable["cls"], y=drawable["med_w"], name=tt,
            marker_color=[_rep_color(c) for c in drawable["cls"]],
            opacity=0.88,
            text=[f"{iw.fmt_watts(m)} W\nn={int(n):,}"
                  for m, n in zip(drawable["med_w"], drawable["n"])],
            textposition="outside", textfont=dict(color=C["muted"], size=10),
            customdata=custom,
            hovertemplate="<b>%{x}</b> · " + tt + "<br>"
                          "median %{customdata[0]}<br>%{customdata[1]}<br>"
                          "%{customdata[2]}<br>%{customdata[3]}<extra></extra>",
        ))
        style_figure(
            fig,
            f"{tt} — detected interval watts, {yr}, one bar per rep length"
            f"<br><sup>each length class is its own measure and is never averaged "
            f"with another. Median of the detected efforts, n on every bar.</sup>",
            H_STD)
        _lane_legend(fig, drawable["cls"].tolist(), min_w=430)
        fig.update_yaxes(tickformat=",.0f")
        show(fig)

        st.caption(
            f"Median watts of the efforts {iw.MIN_CELL_N}+ long that the head unit "
            f"detected in **{tt}** sessions in {yr}, one column per length class. "
            f"Two classes are never merged: the 90 s–5 min column is not a diluted "
            f"version of the under-90 s one."
        )

    # ── Every cell, including the thin ones ────────────────────────────────
    section("\U0001F4CB Every cell, thinnest included")
    rows = []
    for r in mine.itertuples():
        rows.append({
            "Rep length class": r.cls,
            "Detected efforts": f"{int(r.n):,}",
            "Sessions": f"{int(r.sessions):,}",
            "Typical length": iw.fmt_rep(r.rep_secs),
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


def _prescribed_tab(P: pd.DataFrame):
    """The coach's target watts. Lower end of the range, year over year."""
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
                   f"typical rep {iw.fmt_rep(r.rep_secs)}",
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
        f"with another. This series ends at **{last_year}** — see the source "
        f"callout above; extending it would mean drawing a line where no "
        f"prescription exists."
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
            "Typical rep": iw.fmt_rep(r.rep_secs),
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
    st.markdown(
        f"Of **{a['seen']:,}** coach descriptions in this file:\n\n"
        f"- **{a['parsed']:,}** gave an unambiguous interval target and are used.\n"
        f"- **{a['skipped_steady']:,}** carry no repetition at all — a steady "
        f"endurance ride, which has no intervals by definition. That is a correct "
        f"non-match, not a loss.\n"
        f"- **{a['skipped_unread']:,}** do contain a repetition but no readable "
        f"wattage beside it, so they are refused rather than guessed.\n"
        f"- **{a['skipped_unlabelled']:,}** parsed but sit on a session with no "
        f"training type, so they cannot be attributed to a type."
    )

    if a.get("examples_unread"):
        st.markdown("**Examples refused, verbatim.** These are the descriptions a "
                    "reader is most likely to expect to see on the chart:")
        for ex in a["examples_unread"]:
            st.code(ex, language=None)
        st.caption(
            "Common reasons: the target is given in pedal strokes rather than "
            "watts (`pols`), or in a speed (`10k/h`) rather than a power, or the "
            "structure is compound (`3x15' (30\" +330w +14' 250w +30\" +330w)`) "
            "where several efforts share one rep. Reading a number out of those "
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
            f"else, so they are in separate tabs and no line crosses between them.\n"
            f"- **Lower end of the range** is used for a prescribed target, as "
            f"instructed — the end the coach was willing to call a success. The "
            f"upper end is carried alongside and shown in the hover rather than "
            f"discarded.\n"
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
