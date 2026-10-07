# views/intervals_view.py — Intervals page: like-for-like effort work. (NEW FILE)
#
# Data: intervals.icu's interval-level API via core.interval_data (append-only
# disk cache). Section 1 = three comparison tabs: identical SETS over time
# (ml.set_evolution, grouped by group_id), evolution by duration class, and
# the exact duration-window search. Models: ml.exertion_forecast (RPE-
# persistence block — honest strain proxy, since RPE has never been logged),
# ml.interval_forecast (best effort in a matched band), ml.composition_intervals
# (old type-% vs real interval composition, same time-ordered gauntlet vs
# persistence).

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.components import (callout, dataframe, legend, metric_card,
                             page_header, section, show)
from core.interval_data import (fetch_acts, fetch_profile, read_intervals,
                                sync_intervals)
from core.theme import C, H_HERO, H_PAIR, H_STD, SURGERY, style_figure
from ml.composition_intervals import run_composition
from ml.exertion_forecast import metric_counts, run_exertion_forecast
from ml.interval_forecast import run_band_forecast
from ml.protocol_reps import run_protocol_view
from ml.set_evolution import (DETECTED_SRC, PEAK_SRC, REBUILD_SRC,
                              add_signatures,
                              build_sets, fill_peak_efforts,
                              quality_gates,
                              run_duration_evolution, run_set_evolution)
from ml.type_comparison import (FAMILY_SHORT, duration_options, fmt_min,
                                fmt_rest, fmt_secs, intensity_band,
                                run_type_comparison, series_detail,
                                series_stats)
from ml.vo2_estimate import run_vo2_estimate, vo2_l_per_min

ACCENT_RGB = "88,166,255"      # C["accent"] as an rgba core for fan fills
ML_CHART_H = 560                # full-width history+forecast charts


# ── small helpers ────────────────────────────────────────────────────────────
def _add_fan(fig: go.Figure, path: pd.DataFrame, rgb: str = ACCENT_RGB) -> None:
    """50/90% error-band fan + named centre line (bands are fill helpers)."""
    fig.add_trace(go.Scatter(x=path["date"], y=path["hi90"], mode="lines",
                             line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=path["date"], y=path["lo90"], mode="lines",
                             fill="tonexty", fillcolor=f"rgba({rgb},0.16)",
                             line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=path["date"], y=path["hi50"], mode="lines",
                             line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=path["date"], y=path["lo50"], mode="lines",
                             fill="tonexty", fillcolor=f"rgba({rgb},0.32)",
                             line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=path["date"], y=path["y"], mode="lines",
                             name="Forecast",
                             line=dict(color=f"rgba({rgb},1)", width=3.5)))


def _peak_disclosure(sets) -> None:
    """Say which sets came from a source other than the detector, before any
    chart uses them.

    The detector, the peak meter and the rebuild are three ways of describing
    the same ride, and a chart that quietly used all three would be a splice.
    Each count is stated with the rule that keeps the sources from ever
    describing the same effort, and the Source column on every set row keeps
    saying it afterwards.
    """
    if sets is None or not len(sets) or "Source" not in sets.columns:
        return
    n_peak = int((sets["Source"] == PEAK_SRC).sum())
    n_reb = int((sets["Source"] == REBUILD_SRC).sum())
    if not n_peak and not n_reb:
        return
    bits = []
    if n_peak:
        bits.append(
            f"**{n_peak}** read straight off the peak-power meter"
        )
    if n_reb:
        bits.append(
            f"**{n_reb}** rebuilt from the detector's own fragments — one row "
            f"per whole effort the segmenter had cut into pieces, which is "
            f"how a session that rode two 20-minute intervals shows two rows "
            f"instead of the meter's single window"
        )
    st.caption(
        "From a source other than the detector: "
        + " and ".join(bits)
        + ". The detector segments a ride by power and cadence, so it cuts "
        "sustained work into pieces: on 15 Sep 2026 its longest row for that "
        "FTP session was 8 min 05 s, while Intervals.icu's peak meter holds "
        "231 W for 20:00. A row is added only for an effort of 10 min or "
        "longer where that ride has **no** detected set within two minutes of "
        "it **and** none in the same whole-minute duration class, and only "
        "where the meter's own window vouches for the effort — so one effort "
        "is never counted twice. Each enters as a single-rep set (no rest "
        "figure), and the **Source** column of the detail table says which "
        "source reported every row."
    )


def _model_calls(result: dict) -> None:
    """Render the honesty callouts every model result carries."""
    callout("Why this model", result["why"], C["accent"], icon="🧠")
    if result.get("coef_table") is not None:
        st.markdown("**Model coefficients (standardised weights):**")
        dataframe(result["coef_table"], height=320)
    if result.get("note"):
        callout("Plain reading", result["note"], C["muted"], icon="📖")
    st.caption(result.get("band_note", ""))


def _model_table(result: dict) -> None:
    st.markdown("**Walk-forward model comparison:**")
    dataframe(result["model_table"], height=170)


# ── Section 1: compare similar intervals ─────────────────────────────────────
def _matched_window(iv: pd.DataFrame, lo, hi, i_lo, i_hi, work_only):
    w = iv.copy()
    w = w[(pd.to_numeric(w["secs"], errors="coerce") >= lo * 60) &
          (pd.to_numeric(w["secs"], errors="coerce") <= hi * 60)]
    w = w[(pd.to_numeric(w["intensity"], errors="coerce") >= i_lo) &
          (pd.to_numeric(w["intensity"], errors="coerce") <= i_hi)]
    if work_only:
        w = w[w["iv_type"] == "WORK"]
    return w.sort_values("date", ascending=False)


def _render_compare(iv: pd.DataFrame, acts: pd.DataFrame,
                    iv_full: pd.DataFrame, df_all) -> None:
    section("🔍 Compare similar intervals")
    st.caption(
        "Like-for-like work, one protocol at a time: **training types × "
        "duration** (never a 8-minute effort against a 20-minute one — "
        "every type × duration pair is its own series with its own table, "
        "chart and trend), **identical sets over time** (reps grouped by "
        "intervals.icu's own `group_id` — the grouping behind summary "
        "strings like `29x 29s 323w` — matched by reps × rep duration), "
        "**monthly bests by duration class**, and the exact **duration × "
        "intensity window search** (sidebar range; the first three tabs use "
        "the full cached history so trends have room). All durations in "
        "minutes."
    )
    tab_sets, tab_types, tab_dur, tab_win = st.tabs(
        ["🧩 Identical sets over time",
         "🏷️ Training types over time",
         "📈 Evolution by duration class",
         "🔍 Duration-window search"])
    with tab_sets:
        _render_sets(iv_full, acts, df_all)
    with tab_types:
        _render_types(iv_full, acts, df_all)
    with tab_dur:
        _render_duration_evolution(iv_full)
    with tab_win:
        _render_window_search(iv, acts)


def _render_window_search(iv: pd.DataFrame, acts: pd.DataFrame) -> pd.DataFrame:
    st.caption(
        "Exact duration + intensity matching over the rows intervals.icu "
        "detects per ride — the same window its interval-search API uses "
        "(`minSecs/maxSecs/minIntensity/maxIntensity`), computed locally so "
        "you can see every matched effort. Scoped to the sidebar range."
    )

    bc = st.columns(4)
    with bc[0]:
        lo = float(st.number_input("Duration from (min)", 0.1, 180.0, 3.0,
                                   step=0.5, key="ivb_lo"))
    with bc[1]:
        hi = float(st.number_input("Duration to (min)", 0.5, 300.0, 6.0,
                                   step=0.5, key="ivb_hi"))
    with bc[2]:
        i_lo = float(st.number_input("Intensity from (% FTP)", 40.0, 200.0, 90.0,
                                     step=1.0, key="ivi_lo2"))
    with bc[3]:
        i_hi = float(st.number_input("Intensity to (% FTP)", 50.0, 300.0, 120.0,
                                     step=1.0, key="ivi_hi2"))
    work_only = st.checkbox("WORK intervals only (hide warm-ups & recoveries)",
                            value=True, key="iv_work_only")
    if lo > hi:
        lo, hi = hi, lo
    if i_lo > i_hi:
        i_lo, i_hi = i_hi, i_lo

    w = _matched_window(iv, lo, hi, i_lo, i_hi, work_only)
    if not len(w):
        callout("No intervals match",
                f"Nothing in {lo:g}–{hi:g} min × {i_lo:.0f}–{i_hi:.0f}% "
                f"FTP in this range. Widen the window.", C["yellow"], icon="🔍")
        return w

    # KPI row
    w_sorted = w.sort_values("date")
    best_avg = pd.to_numeric(w_sorted["avg_w"], errors="coerce").max()
    best_np = pd.to_numeric(w_sorted["np_w"], errors="coerce").max()
    mean_if = pd.to_numeric(w_sorted["intensity"], errors="coerce").mean() / 100
    last28 = w_sorted[w_sorted["date"] >= pd.Timestamp.now().normalize()
                      - pd.Timedelta(days=28)]
    prior = w_sorted[w_sorted["date"] < pd.Timestamp.now().normalize()
                     - pd.Timedelta(days=28)]
    d_best = (pd.to_numeric(last28["avg_w"], errors="coerce").max()
              - pd.to_numeric(prior["avg_w"], errors="coerce").max()) \
        if len(last28) and len(prior) else None

    kc = st.columns(5)
    with kc[0]:
        metric_card("Matched efforts", f"{len(w)}",
                    foot=f"{lo:g}–{hi:g} min · {i_lo:.0f}–{i_hi:.0f}% FTP",
                    accent=C["accent"])
    with kc[1]:
        metric_card("Best avg power", f"{best_avg:.0f} W",
                    delta=f"{d_best:+.0f} W vs previous best"
                    if d_best is not None and not np.isnan(d_best) else None,
                    tone="good" if (d_best or 0) > 0 else "bad"
                    if d_best is not None else None,
                    accent=C["green"])
    with kc[2]:
        metric_card("Best interval NP", f"{best_np:.0f} W", accent=C["purple"])
    with kc[3]:
        metric_card("Mean intensity", f"{mean_if:.2f} IF",
                    foot="relative to FTP at effort time", accent=C["orange"])
    with kc[4]:
        metric_card("Last effort", str(w_sorted['date'].max().date()),
                    accent=C["muted"])

    # Charts
    cc = st.columns(2)
    with cc[0]:
        dd = w_sorted.groupby("date")["avg_w"].max()
        best28 = dd.rolling(28, min_periods=1).max()
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=w_sorted["date"],
            y=pd.to_numeric(w_sorted["avg_w"], errors="coerce"),
            mode="markers", name="Effort (avg W)",
            marker=dict(
                size=(7 + pd.to_numeric(w_sorted["secs"], errors="coerce")
                      / 60).clip(upper=26),
                color=pd.to_numeric(w_sorted["intensity"], errors="coerce"),
                colorscale=[[0, C["green"]], [0.5, C["yellow"]], [1, C["red"]]],
                showscale=True,
                colorbar=dict(title="IF %", thickness=10, len=0.5),
                line=dict(color=C["border"], width=1),
            ),
            hovertemplate="%{x|%d %b %Y}<br>%{y:.0f} W<extra></extra>",
        ))
        fig1.add_trace(go.Scatter(x=best28.index, y=best28.values,
                                  mode="lines", name="28-day best (avg W)",
                                  line=dict(color=C["accent"], width=3)))
        style_figure(fig1, f"Matched efforts over time — {len(w)} efforts",
                     H_PAIR)
        legend(fig1, "left", horizontal=True)
        show(fig1)

    with cc[1]:
        # duration → power curve from ALL work intervals in range
        work = iv[iv["iv_type"] == "WORK"].copy()
        work["secs"] = pd.to_numeric(work["secs"], errors="coerce")
        work["avg_w"] = pd.to_numeric(work["avg_w"], errors="coerce")
        work = work.dropna(subset=["secs", "avg_w"])
        edges = [0, 30, 60, 120, 180, 300, 480, 720, 1200, 1800, 3600, 1e9]
        labels = ["30s", "1m", "2m", "3m", "5m", "8m", "12m", "20m", "30m",
                  "60m", "60m+"]
        work["bucket"] = pd.cut(work["secs"], bins=edges, labels=labels,
                                right=False)
        curve = work.groupby("bucket", observed=True)["avg_w"].max()
        xs = [edges[i] / 60 for i in range(len(labels))]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=xs, y=curve.reindex(labels).values, mode="lines+markers",
            name="Best avg W by duration",
            line=dict(color=C["accent"], width=3), marker=dict(size=9)))
        fig2.add_vrect(x0=lo, x1=hi, fillcolor=C["accent"], opacity=0.14,
                       line_width=0)
        fig2.update_xaxes(type="log", title="Duration (min, log scale)")
        style_figure(fig2, "Interval power curve — band highlighted", H_PAIR)
        legend(fig2, "left", horizontal=True)
        show(fig2)

    # Table of matched efforts
    with st.expander(f"📋 All {len(w)} matched efforts", expanded=False):
        names = acts.set_index(acts["id"].astype(str))["name"]
        t = w_sorted.copy()
        t["Activity"] = t["activity_id"].astype(str).map(names).fillna("")
        tbl = pd.DataFrame({
            "Date": t["date"].dt.strftime("%Y-%m-%d"),
            "Activity": t["Activity"],
            "Kind": t["iv_type"],
            "Dur": t["secs"].map(fmt_min),
            "Dur s": t["secs"].map(fmt_secs),
            "Avg W": pd.to_numeric(t["avg_w"], errors="coerce").round(0),
            "NP W": pd.to_numeric(t["np_w"], errors="coerce").round(0),
            "HR": pd.to_numeric(t["hr_avg"], errors="coerce"),
            "IF %": pd.to_numeric(t["intensity"], errors="coerce"),
            "Load": pd.to_numeric(t["load"], errors="coerce").round(1),
            "Zone": t["zone"],
        }).iloc[::-1]
        dataframe(tbl, height=420)
    return w


# ── Section 1 tab B: training types over time (type × duration series) ──────
TYPE_COLORS = ["#58a6ff", "#3fb950", "#d29922", "#bc8cff", "#f778ba",
               "#39c5cf", "#ff7b72", "#a5d6ff"]
TYPE_FIG_H = 300
# A full-width family row pays for a two-row legend in the top margin, so it
# needs the extra pixels back or the plot ends up a letterbox.
TYPE_FIG_H_FULL = 350


def _family_figure(fam: str, series: list, show_n: int) -> tuple:
    """One line per DURATION inside a family — durations are never mixed
    into a single series, only drawn on the same axes for context.
    Returns (figure, legend labels): the legend is placed by the caller
    AFTER style_figure, which is what sets the margins the lane lives in."""
    shown = sorted(series, key=lambda x: (-x["n_days"], -x["n"]))[:show_n]
    fig = go.Figure()
    for i, sr in enumerate(shown):
        col = TYPE_COLORS[i % len(TYPE_COLORS)]
        p = sr["points"]
        fig.add_trace(go.Scatter(
            x=p["date"], y=p["set_w"], mode="markers", name=sr["label"],
            marker=dict(size=8, color=col, line=dict(color=C["border"],
                                                     width=1)),
            hovertemplate="%{x|%d %b %Y} · %{y:.0f} W<extra></extra>"))
        if sr["trend"] is not None:
            fig.add_trace(go.Scatter(
                x=sr["trend"]["date"], y=sr["trend"]["y_hat"],
                mode="lines", showlegend=False,
                line=dict(color=col, width=2, dash="dot"),
                hovertemplate="trend · %{y:.0f} W<extra></extra>"))
    return fig, [sr["label"] for sr in shown]


# ── cached analysis entry points ─────────────────────────────────────────────
# The type × duration analysis is a bootstrap over every series (~9 s) and the
# interval explorer walks the raw cache. Both are pure functions of the cached
# data, so Streamlit's value-hash cache keeps a widget change (choosing another
# type, moving a range) from re-running the whole thing. Cached INSIDE the
# session's data TTL — a sync changes the hash and invalidates it by itself.
# _CODE_TAG is part of the cache key: Streamlit hashes ARGUMENTS, not function
# bodies, so an edit inside ml/type_comparison.py or ml/protocol_reps.py would
# otherwise keep serving the old numbers for the full TTL. Bump the string
# whenever those modules change.
_CODE_TAG = "rep-cls-2026-10-04a"


@st.cache_data(show_spinner=False, ttl=1800, max_entries=8)
def _type_comparison_cached(sets: pd.DataFrame, code_tag: str = _CODE_TAG
                            ) -> dict:
    return run_type_comparison(sets)


@st.cache_data(show_spinner=False, ttl=1800, max_entries=64)
def _protocol_view_cached(iv: pd.DataFrame, sets: pd.DataFrame, family: str,
                          dur_b: float | None, dur_cls: str | None = None,
                          code_tag: str = _CODE_TAG) -> dict:
    return run_protocol_view(iv, sets, family, dur_b, dur_cls)


# ── quality screen: show what was excluded, never hide it ───────────────────
def _quality_audit(res: dict) -> None:
    """Every set dropped by the data-quality screen, counted and listed — and
    every set whose intensity field is unusable, flagged rather than deleted."""
    report = res.get("quality")
    dropped = res.get("excluded")
    n_drop = int(len(dropped)) if dropped is not None else 0
    n_kept = int(res.get("n_sets", 0))
    n_flag = 0
    if report is not None and len(report):
        n_flag = int(report.loc[report["Action"].str.startswith("flagged"),
                                "Sets"].sum())
    if not n_drop and not n_flag:
        callout("Data quality — nothing screened out or flagged",
                f"All {n_kept} sets passed: no duplicated rides, no "
                f"multi-minute gaps inside a \"set\", no sub-15 s reps and no "
                f"unusable intensity values. The same screen runs on every "
                f"table in this tab.", C["green"], icon="✅")
        return
    parts = []
    if n_drop:
        parts.append(f"**{n_drop} sets excluded** — the same ride in the cache "
                     f"twice, or a \"set\" whose reps sit minutes-to-hours "
                     f"apart (its average and its rest figure are invalid)")
    if n_flag:
        parts.append(f"**{n_flag} sets flagged, not deleted** — their watts "
                     f"are real, but sub-15 s reps and IF above 150 % mean "
                     f"the intensity field is a model artefact, so their "
                     f"intensity band is reported as unusable instead of "
                     f"being averaged into a conclusion")
    callout("Data quality — the screen, on the record", " · ".join(parts) +
            ". Nothing is dropped silently: the counts are below and every "
            "excluded set is listed.", C["yellow"], icon="🧹")
    with st.expander("🧾 What the screen found", expanded=False):
        if report is not None and len(report):
            dataframe(report, height=220)
        if dropped is not None and len(dropped):
            d = dropped.copy()
            d["Date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
            d["Rep"] = d["rep_secs"].map(fmt_min)
            d["Rep s"] = d["rep_secs"].map(fmt_secs)
            d["Reason"] = d["q_why"]
            d["W"] = pd.to_numeric(d["set_w"], errors="coerce").round(0)
            d["IF %"] = pd.to_numeric(d["intensity"],
                                       errors="coerce").round(0)
            d["Rest"] = d["rest"].map(fmt_rest)
            cols = [c for c in ("Date", "name", "reps", "Rep", "Rep s", "W",
                                "IF %", "Rest", "Reason") if c in d.columns]
            st.dataframe(d[cols].sort_values("Date"), width="stretch",
                         hide_index=True, height=320)


# ── legends get their own LANE, never a corner of the plot ─────────────────
# Two ways a Plotly legend ruins a chart, both of which this page had:
#   1. floating INSIDE the plot area (the shared legend() helper parks it at
#      x=0.005, y=0.995) — on a 6-series small multiple it lies straight
#      across the markers, so the reader cannot see the data AND cannot tell
#      the lines apart. That is the "unreadable" report on the 11-duration
#      single-efforts chart.
#   2. pushed OUT past the canvas edge (x=0.995) — clipped by the SVG.
# A legend in a MARGIN is neither: it is inside the canvas, and it is over
# nothing. Short names take a row in the top margin; long names stack in the
# right margin, which grows to fit the longest one. The row width is estimated
# from the labels — ≈5.0 px per char at font 10 plus 28 px per swatch,
# calibrated against the rendered DOM — so the choice is made here, not
# guessed at paint time. MUST be called after style_figure(), which replaces
# the whole margin dict.
LEG_CHAR_PX = 5.0
LEG_ENTRY_PX = 28
LEG_GAP_PX = 10
LEG_ROW_PX = 22          # one horizontal row incl. padding
LEG_LANE_PAD = 12
LEG_BUDGET_W = 430.0     # narrowest full-width column this page produces


def _lane_legend(fig: go.Figure, labels, min_w: float = 460.0,
                 size: int = 10, cols: int | None = None) -> go.Figure:
    labels = [str(x) for x in labels if x]
    if not labels:
        return fig
    row = (sum(LEG_CHAR_PX * len(x) + LEG_ENTRY_PX for x in labels)
           + LEG_GAP_PX * (len(labels) - 1))
    m = fig.layout.margin
    base_t = 74 if (fig.layout.title and fig.layout.title.text) else 48
    base_r = 24 if m.r is None else int(m.r)
    common = dict(font=dict(size=size, color="#c9d1d9"),
                  bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d",
                  borderwidth=1)
    if row <= min_w:
        # one row, in the top margin, under the title and over nothing
        fig.update_layout(legend=dict(common, orientation="h", x=0.0,
                                      xanchor="left", y=1.0, yanchor="bottom"))
        fig.update_layout(margin=dict(t=base_t + LEG_ROW_PX))
    elif cols:
        # wrap onto rows of `cols` by giving every entry a FIXED FRACTION of
        # the width — a horizontal legend is otherwise one row and cannot
        # wrap. Plotly then lays them out in ceil(n/cols) rows, all of them
        # in the top margin, and the plot keeps the full width instead of
        # surrendering 150 px to a side lane. cols is capped by the longest
        # name so no entry is ever truncated to "…", at the cost of one more
        # row. 430 px is the narrowest full-width column this page produces
        # (measured), used as the budget so the choice survives a resize.
        need = LEG_CHAR_PX * max(len(x) for x in labels) + LEG_ENTRY_PX
        cols = max(1, min(cols, int(LEG_BUDGET_W / need)))
        n_rows = -(-len(labels) // cols)
        fig.update_layout(legend=dict(
            common, orientation="h", x=0.0, xanchor="left", y=1.0,
            yanchor="bottom", entrywidthmode="fraction",
            entrywidth=round(1.0 / cols, 3)))
        fig.update_layout(margin=dict(t=base_t + n_rows * LEG_ROW_PX))
    else:
        # a stack in the right margin, widened to the longest name
        need = (LEG_CHAR_PX * max(len(x) for x in labels) + LEG_ENTRY_PX
                + LEG_LANE_PAD)
        fig.update_layout(legend=dict(common, orientation="v", x=1.0,
                                      xanchor="left", y=1.0, yanchor="top"))
        fig.update_layout(margin=dict(r=int(min(190, need)) + base_r))
    return fig


def _span_txt(stt: dict) -> str:
    """How long the intervals in a series actually were: whole minutes first
    (what the reader wants) with the exact measured seconds beside it, so a
    rounded '3 min' never hides that the reps ran 2:50–3:55."""
    try:
        lo = float(stt["dur_min"])
        hi = float(stt["dur_max"])
        med = float(stt["dur_med"])
    except (KeyError, TypeError, ValueError):
        return "—"
    if not all(np.isfinite(v) for v in (lo, hi, med)):
        return "—"
    if hi < 90:                      # sub-minute: seconds are the honest unit
        return f"{lo:.0f}–{hi:.0f} s, median {med:.0f} s"
    return (f"{fmt_min(lo)}–{fmt_min(hi)} "
            f"({lo:.0f}–{hi:.0f} s, median {med:.0f} s)")


def _as_int(v) -> int:
    """Counts go into hover text; a missing count is 0 there, but every count
    that the page prints as a headline number is computed from the frame."""
    try:
        if pd.isna(v):
            return 0
        return int(v)
    except (TypeError, ValueError):
        return 0


def _tok(v, unit: str = "", fmt: str = "{:.0f}") -> str:
    """A hover value WITH its unit — and a dash, never a bare \"nan\", when
    the source that reported this bar holds no such field (the peak-power
    meter gives watts and length only)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return fmt.format(x) + unit if np.isfinite(x) else "—"


def _rep_bar_figure(pv: dict) -> go.Figure:
    """One bar per individual interval; bars are grouped into one slot per
    session, with that session's average watts marked. No smoothing, no
    invented points — what you see is what intervals.icu recorded."""
    bars = pv["bars"]
    sess = pv["sessions"]
    dates = list(sess["date"])
    slot = {d: i for i, d in enumerate(dates)}
    n_per = bars.groupby("date")["seq"].size()
    busiest = int(n_per.max()) if len(n_per) else 1
    width = 0.86 / max(busiest, 1)
    x, y, cd = [], [], []
    for d, g in bars.groupby("date", sort=True):
        i = slot.get(d)
        if i is None:
            continue
        n = len(g)
        for j, row in enumerate(g.itertuples()):
            x.append(i + (j - (n - 1) / 2.0) * width)
            y.append(row.avg_w)
            # pre-formatted on purpose: the peak meter has no NP, HR, IF or
            # cadence, and a hover that printed \"nan W\" for those would be
            # reading nothing while looking like a number
            cd.append([d, row.rep_idx, n, _tok(row.secs, " s"),
                       _tok(row.np_w, " W"), _tok(row.hr_avg, " bpm"),
                       _tok(row.intensity, " %"), _tok(row.cad_avg, " rpm"),
                       _tok(row.tsb, "", "{:+.0f}"),
                       _tok(row.temp, " °C", "{:.1f}"),
                       str(getattr(row, "src", DETECTED_SRC))])
    cd = np.array(cd, dtype=object)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=y, width=width * 0.86,
        name="Individual interval (avg W)",
        marker=dict(color=C["accent"],
                    line=dict(color=C["panel2"], width=0.5)),
        customdata=cd,
        hovertemplate=("%{customdata[0]|%d %b %Y} · interval "
                       "%{customdata[1]}/%{customdata[2]}<br>%{y:.0f} W · "
                       "%{customdata[3]}<br>"
                       "NP %{customdata[4]} · HR %{customdata[5]} · "
                       "IF %{customdata[6]}<br>"
                       "%{customdata[7]} · TSB %{customdata[8]} · "
                       "%{customdata[9]}<br>"
                       "%{customdata[10]}<extra></extra>")))
    # The session average has to be unmistakable. A bare 14 px marker lost in a
    # row of 138 thin bars reads as "just another bar top", and with no number
    # attached there was nothing to read off the chart at all. So each session
    # gets a dotted line across its own slot AT that session's mean, with a
    # large orange-ringed diamond on it and the wattage printed above.
    seg_x, seg_y = [], []
    for d, mw in zip(dates, sess["w"]):
        if not np.isfinite(mw):
            continue
        i = slot[d]
        seg_x += [i - 0.44, i + 0.44, None]
        seg_y += [mw, mw, None]
    fig.add_trace(go.Scatter(
        x=seg_x, y=seg_y, mode="lines", name="Session average",
        line=dict(color=C["orange"], width=2, dash="dot"),
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[slot[d] for d in dates], y=sess["w"], mode="markers+text",
        name="Session average", showlegend=False, cliponaxis=False,
        text=[f"{mw:.0f} W" if np.isfinite(mw) else "" for mw in sess["w"]],
        textposition="top center",
        textfont=dict(size=11, color=C["orange"]),
        marker=dict(size=15, symbol="diamond", color=C["panel"],
                    line=dict(color=C["orange"], width=3)),
        customdata=np.array([[d, _as_int(r), _as_int(s)] for d, r, s in
                             zip(dates, sess["reps"], sess["sets"])],
                            dtype=object),
        hovertemplate=("%{customdata[0]|%d %b %Y} · session average "
                       "%{y:.0f} W<br>%{customdata[1]} intervals in "
                       "%{customdata[2]} set(s)<extra></extra>")))
    fig.update_layout(barmode="overlay", bargap=0.0, bargroupgap=0.0)
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(dates))),
                     ticktext=[pd.Timestamp(d).strftime("%d %b")
                               for d in dates],
                     range=[-0.6, len(dates) - 0.4], showgrid=False)
    return fig


# ── session-average line + robust trend + bootstrap CI band ─────────────────
def _session_trend_figure(pv: dict, on_temp: bool, on_tsb: bool) -> go.Figure:
    """One point per session = that session's average watts. Trend = Theil–Sen
    (robust to a single all-out session), shaded with its bootstrap 95 % CI.
    The line is only drawn when the data can support it."""
    sess = pv["sessions"]
    tr = pv.get("trend") or {}
    slope = tr.get("slope_robust", np.nan)
    fig = go.Figure()
    if sess["trend_lo"].notna().any():
        fig.add_trace(go.Scatter(
            x=sess["date"], y=sess["trend_hi"], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=sess["date"], y=sess["trend_lo"], mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(210,153,34,0.16)",
            name="Trend 95% CI", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=sess["date"], y=sess["w"], mode="lines+markers",
        name="Session avg (of its reps)",
        line=dict(color=C["accent"], width=2, shape="spline", smoothing=0.4),
        marker=dict(size=9, color=C["accent"],
                    line=dict(color=C["panel"], width=1)),
        customdata=np.array([[_as_int(r), _as_int(s), f, l, b, w] for r, s, f,
                              l, b, w
                              in zip(sess["reps"], sess["sets"], sess["open_w"],
                                    sess["close_w"], sess["best"],
                                    sess["worst"])], dtype=object),
        hovertemplate=("%{x|%d %b %Y} · %{y:.0f} W<br>%{customdata[0]} "
                       "intervals in %{customdata[1]} set(s)<br>"
                       "opening %{customdata[2]:.0f} → closing "
                       "%{customdata[3]:.0f} W<br>best %{customdata[4]:.0f} · "
                       "worst %{customdata[5]:.0f} W<extra></extra>")))
    if sess["trend"].notna().any():
        label = "Theil–Sen trend" if not np.isnan(slope) else "OLS trend"
        if not np.isnan(slope):
            label += f" {slope:+.1f} W/month"
        fig.add_trace(go.Scatter(
            x=sess["date"], y=sess["trend"], mode="lines", name=label,
            line=dict(color=C["orange"], width=2, dash="dot"),
            hovertemplate="trend · %{y:.0f} W<extra></extra>"))
    y2 = False
    if on_temp:
        t = sess.dropna(subset=["temp"])
        if len(t):
            fig.add_trace(go.Scatter(
                x=t["date"], y=t["temp"], mode="lines",
                name="Temperature (°C)",
                line=dict(color=C["yellow"], width=2), yaxis="y2"))
            y2 = True
    if on_tsb:
        t = sess.dropna(subset=["tsb"])
        if len(t):
            fig.add_trace(go.Scatter(
                x=t["date"], y=t["tsb"], mode="lines",
                name="TSB (CTL − ATL)",
                line=dict(color=C["purple"], width=2, dash="dash"),
                yaxis="y2"))
            y2 = True
    if y2:
        fig.update_layout(yaxis2=dict(overlaying="y", side="right",
                                      showgrid=False))
    return fig


def _session_table(sess: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "Date": pd.to_datetime(sess["date"]).dt.strftime("%Y-%m-%d"),
        "Intervals": pd.to_numeric(sess["reps"], errors="coerce")
        .round().astype("Int64"),
        "Sets": pd.to_numeric(sess["sets"], errors="coerce")
        .round().astype("Int64"),
        "Length": sess["secs_mean"].map(fmt_min),
        "Length s": sess["secs_mean"].map(fmt_secs),
        "Avg W": sess["w"].round(0),
        "Open W": sess["open_w"].round(0),
        "Close W": sess["close_w"].round(0),
        "Fade %": sess["fade_%"].round(1),
        "Best W": sess["best"].round(0),
        "Worst W": sess["worst"].round(0),
        "IF %": sess["if_mean"].round(0),
        "Trend W": sess["trend"].round(0),
        "°C": sess["temp"].round(1),
        "TSB": sess["tsb"].round(1),
        "Ride": sess["name"].fillna("").astype(str),
    })
    return out.iloc[::-1]          # newest first


def _rep_table(bars: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "Date": pd.to_datetime(bars["date"]).dt.strftime("%Y-%m-%d"),
        "Interval": bars["rep_idx"].astype(int),
        "In set": bars["reps_in_set"].astype(int),
        "Length": bars["secs"].map(fmt_min),
        "Length s": bars["secs"].map(fmt_secs),
        "Avg W": bars["avg_w"].round(0),
        "NP W": bars["np_w"].round(0),
        "HR": bars["hr_avg"].round(0),
        "IF %": bars["intensity"].round(0),
        "Cad": bars["cad_avg"].round(0),
        "°C": bars["temp"].round(1),
        "TSB": bars["tsb"].round(1),
        # which instrument reported this interval — the detector's own row,
        # or the peak meter's held window where the detector cut the effort
        "Source": (bars["src"].fillna(DETECTED_SRC)
                   if "src" in bars.columns else DETECTED_SRC),
        "Ride": bars["name"].fillna("").astype(str),
    })
    return out.iloc[::-1]


def _render_types(iv_full: pd.DataFrame, acts: pd.DataFrame, df_all) -> None:
    st.caption(
        "Every row is **one training type at one duration** — its own "
        "sessions, its own watts, its own trend. An 8-minute threshold set "
        "and a 20-minute one are different series and are never averaged "
        "together. **Duration is the length of ONE interval (a rep)** — "
        "never the workout length — rounded to the closest nominal, 5 minutes "
        "at a time from ten minutes up: a 20:07 reads \"20 min\", a 24:47 "
        "reads \"25 min\", and the same closest-number rule applies at every "
        "step (an exact 22:30 goes up to 25). Below ten minutes the minutes "
        "stay whole halves-down — up to 3.5 min is a \"3 min\" effort, more "
        "than 3.5 min is \"4 min\" — and the detector wobbles ±10 s on the "
        "same programmed effort, so finer classes would split one workout "
        "into two fake series; a 30 s rep keeps its real-world 30 s. "
        "Every row of the detail table still shows the exact measured "
        "length. **The selector at the bottom of this tab then groups those "
        "durations into the length classes Evolution charts with — "
        "anything under 22:00 is 10-20 min, anything under 32:00 is 20-30 "
        "min — because a 20:00, a 20:07 and a 21:00 are the same 20-minute "
        "effort on that chart; the table above keeps them apart, the detail "
        "below gathers them.** "
        "**The grouping is YOUR training type** — the label the "
        "session was filed under (FTP, VO2MAX, BILLAT, AEROBIC BASE …), "
        "not a guess from the rep length; only sessions with no label fall "
        "back to the protocol heuristic (rep length + measured rest) or "
        "“single efforts”. **Intensity is "
        "reported separately from the protocol name**: a 30-s rep's IF is "
        "intervals.icu extrapolating a 20-min equivalent from 30 s, so below "
        "45 s the band is shown as unusable instead of as a physiological "
        "percentage, and a 3-minute set ridden at 84 % IF is labelled as "
        "sub-threshold rather than quietly counted as VO₂ work."
    )
    sets = build_sets(iv_full, acts, df_all)
    sets, _ = fill_peak_efforts(sets, df_all, iv=iv_full)
    if len(sets) < 3:
        callout("Not enough sets", "Fewer than 3 sets in the cached history "
                "— sync more activities first.", C["yellow"], icon="⏸️")
        return
    _peak_disclosure(sets)
    sets, _ = add_signatures(sets)
    res = _type_comparison_cached(sets)
    if not res.get("ok"):
        callout("Not enough data", res["reason"], C["yellow"], icon="⏸️")
        return
    summ = res["summary"]

    # 1 — the conclusions table
    st.markdown("**Every comparable series — watts by type × duration, over "
                "time:**")
    table_cols = ["Type", "Duration", "Sets", "Sessions", "First", "Last",
                  "First W", "Last W", "Δ W", "Δ %", "Best W", "W/month",
                  "± 95%", "Trend", "Ridden", "Rest", "IF %"]
    dataframe(summ[table_cols], height=470)
    callout(
        "How to read one row",
        "**Type · Duration** is the identity of the series — same type, "
        "different duration class = a different row, never merged. "
        "**Duration = one interval's length**, rounded to whole minutes "
        "with halves down (up to 3.5 min → \"3 min\", more → \"4 min\"); the "
        "detail table shows each rep's exact measured length. **The unit of "
        "every number here is the SESSION** — a day with two sets is one "
        "observation, not two, so a 2×-per-day rider cannot inflate the "
        "trend; `Sets` says how many sets were averaged. First/Last W are the "
        "first and last session's average watts, Best W the best session, "
        "Δ % last − first. **W/month is the robust Theil–Sen slope** (median "
        "of every pairwise slope, so one all-out session cannot drag it) and "
        "**± 95% is its bootstrap confidence interval** — the arrow only "
        "appears when the interval clears zero AND the slope is at least "
        "1 W/month; otherwise → means \"not separable from noise\", which is a "
        "result, not a failure, and below five distinct days the interval is "
        "left blank rather than faked. Ridden is the intensity band of that "
        "median IF (VO₂ ≥105 %, threshold 88–105 %, sub-threshold <88 %, or "
        "\"n/a\" when the intensity field is unusable) — the protocol name "
        "describes the shape, this column describes how hard it was actually "
        "ridden. Rest is the measured median recovery between reps. "
        "Descriptive history — not a forecast, and no claim about why it "
        "changed.",
        C["yellow"], icon="📏")

    # 2 — per-family small multiples, one line per duration
    st.markdown("**Each training type over time — one session average per "
                "day, one line per duration, dotted = robust trend of that "
                "line only:**")
    fams = [f for f in res["families"] if res["lines"].get(f)]
    # A family with more than two series gets its OWN full-width row. Two per
    # row is only ~257 px, and six lines plus a six-name legend do not fit in
    # 257 px — the legend had to sit on top of the markers, which is what made
    # the 11-duration single-efforts chart unreadable. One per row: the plot
    # grows from ~180 px to ~270 px and the legend moves into a margin lane,
    # over nothing. Two-series families still pair up, because they fit.
    simple = [f for f in fams if len(res["lines"][f]) <= 2]
    rich = [f for f in fams if len(res["lines"][f]) > 2]
    grid: list = []
    for i in range(0, len(simple), 2):
        grid.append(list(simple[i:i + 2]))
    grid += [[f] for f in rich]
    for row in grid:
        cols = st.columns(len(row))
        for col_i, fam in enumerate(row):
            with cols[col_i]:
                series = res["lines"][fam]
                show_n = 6
                fig, labels = _family_figure(fam, series, show_n)
                full = len(row) == 1
                style_figure(
                    fig,
                    f"{FAMILY_SHORT.get(fam, fam)} — "
                    f"{len(series)} duration"
                    f"{'s' if len(series) > 1 else ''}",
                    TYPE_FIG_H_FULL if full else TYPE_FIG_H,
                    showlegend=len(series) > 1)
                # after style_figure: it owns the margins the lane sits in.
                # A full-width family with 4-6 durations wraps the legend onto
                # rows of three in the top margin, so the plot keeps the whole
                # width; only a genuinely narrow column falls back to a side
                # lane, because a side lane there would be most of the chart.
                _lane_legend(fig, labels, min_w=430 if full else 230,
                             cols=3 if full else None)
                show(fig)
                if len(series) > show_n:
                    st.caption(f"top {show_n} of {len(series)} durations by "
                               f"sessions — all in the table above")

    # 3 — the data-quality screen, on the record
    _quality_audit(res)

    # 4 — ONE training type × ONE duration, every individual interval
    st.markdown("**Pick a training type, then one duration — every single "
                "interval as a bar, and the session average over time:**")
    fam_lab = {}
    for f in res["families"]:
        g = res["sets"][res["sets"]["family"] == f]
        if not len(g):
            continue
        nd, nb = int(g["date"].nunique()), int(g["dur_cls"].nunique())
        base = (f"{FAMILY_SHORT.get(f, f)} · {nd} sessions · {nb} length "
                f"class{'es' if nb > 1 else ''}")
        lab, k = base, 2
        while lab in fam_lab:
            lab, k = f"{base} ({k})", k + 1
        fam_lab[lab] = f
    fam_choice = st.selectbox("1 · Training type", list(fam_lab),
                              key="iv_type_fam")
    fam = fam_lab[fam_choice]
    gf = res["sets"][res["sets"]["family"] == fam]
    # The options are Evolution's rep-length bands with the athlete's own
    # tolerance: anything under 22:00 is "10-20 min", so a 20:00, a 20:07
    # and a 21:00 sit in ONE option instead of three — one criterion, one
    # session list on both pages.
    dur_lab = duration_options(gf)
    dur_choice = st.selectbox("2 · Duration class — the length of ONE "
                              "interval", list(dur_lab), key="iv_type_dur")
    cls = dur_lab[dur_choice]
    fam_name = FAMILY_SHORT.get(fam, fam)
    # "an under 90s interval", but "a 5-10 min interval": the article follows
    # how the class NAME starts, never a hardcoded "a".
    art = "an" if cls[:1].lower() in "aeiou" else "a"
    st.caption(f"**{fam_name} at {cls}** — {art} {cls} interval is the only "
               f"thing ever compared with {art} {cls} interval. No other "
               f"training type, no other duration class, no workout-average "
               f"substitution. Same classes as the Evolution day chart: "
               f"every day it draws for this class is a session here too.")

    pv = _protocol_view_cached(iv_full, res["sets"], fam, None, cls)
    if not pv.get("ok"):
        callout("No intervals for that series", pv.get("reason", "—"),
                C["yellow"], icon="⏸️")
        return
    sess, bars, tr = pv["sessions"], pv["bars"], (pv.get("trend") or {})
    stt = series_stats(res["sets"], fam, dur_cls=cls)
    slope, ci = tr.get("slope_robust", np.nan), tr.get("ci", np.nan)
    clear = (not np.isnan(ci)) and abs(slope) > ci and abs(slope) >= 1.0
    n_sets = int(pd.to_numeric(sess["sets"], errors="coerce")
                 .fillna(sess["reps"]).sum())
    if_med = pd.to_numeric(stt.get("if_med"), errors="coerce")
    if_band = stt["if_band"] if "if_band" in stt else "—"

    kc = st.columns(5)
    with kc[0]:
        metric_card("Sessions", f"{pv['n_sessions']}",
                    foot=f"{pv['n_reps']} individual intervals in "
                         f"{n_sets} set{'' if n_sets == 1 else 's'}",
                    accent=C["accent"])
    with kc[1]:
        metric_card("First → last session avg",
                    f"{tr.get('first', float('nan')):.0f} → "
                    f"{tr.get('last', float('nan')):.0f} W",
                    foot=f"{pd.Timestamp(sess['date'].iloc[0]).date()} → "
                         f"{pd.Timestamp(sess['date'].iloc[-1]).date()}",
                    tone="good" if tr.get("last", 0) > tr.get("first", 0)
                    else "bad", accent=C["muted"])
    with kc[2]:
        metric_card("Robust trend",
                    f"{slope:+.1f} W/month" if not np.isnan(slope) else "—",
                    foot=(f"Theil–Sen · 95% CI ±{ci:.1f} W/month"
                          if not np.isnan(ci) and not np.isnan(slope)
                          else "needs ≥ 3 sessions over a real time span"
                          if np.isnan(slope)
                          else "CI spans zero — not separable from noise"),
                    tone="good" if clear and slope >= 1 else
                    "bad" if clear and slope <= -1 else None,
                    accent=C["green"] if clear and slope >= 1 else
                    C["red"] if clear and slope <= -1 else C["muted"])
    with kc[3]:
        metric_card("Best session avg", f"{tr.get('best', float('nan')):.0f} W",
                    foot=(f"best single interval {tr['best_rep']:.0f} W"
                          if tr.get("best_rep") else "—"),
                    accent=C["green"])
    with kc[4]:
        fade = tr.get("fade_med", np.nan)
        # A fade exists only where a session actually holds several reps. The
        # series median of the rep counts can say "1" while one session of the
        # four is a three-rep block — and then a fade number would sit over
        # the words "nothing to fade between". Count the sessions that have
        # one, instead of guessing from the median.
        n_fade = (int(sess["fade_%"].notna().sum())
                  if "fade_%" in sess.columns else 0)
        multi_rep = n_fade > 0
        metric_card("Fade inside sessions",
                    f"{fade:+.1f} %" if not np.isnan(fade) else "—",
                    foot=(f"closing (last 2 reps) vs opening (reps 2–3) · "
                          f"{n_fade} of {len(sess)} sessions have both"
                          if multi_rep else
                          "single efforts — one rep, nothing to fade between"),
                    tone="good" if (not np.isnan(fade) and fade <= 2)
                    else "warn" if not np.isnan(fade) else None,
                    accent=C["purple"])

    # 4a — every individual interval, one bar each, grouped per session
    if len(bars):
        # which instrument reported each bar, stated before the chart uses it
        _src = bars["src"] if "src" in bars.columns else pd.Series(dtype=object)
        n_meter, n_reb = int((_src == PEAK_SRC).sum()), int(
            (_src == REBUILD_SRC).sum())
        _who = []
        if n_meter:
            _who.append(
                f"{n_meter} are the peak-power meter's held window — one "
                f"effort with no NP, HR, IF or cadence in that source"
            )
        if n_reb:
            _who.append(
                f"{n_reb} are whole efforts rebuilt from the detector's own "
                f"fragments — its pieces joined across breaks under 1:30, one "
                f"bar per effort ridden rather than the meter's single window"
            )
        st.markdown(f"**Every interval of every session — {pv['n_reps']} "
                    f"bars, nothing smoothed. The orange dotted line across "
                    f"each day is that session's AVERAGE watts; ◇ marks it "
                    f"and prints the number.** "
                    + ("Bars from the long-effort sources: "
                       + "; ".join(_who)
                       + ". Hover the bar or read the Source column to see "
                         "which instrument reported it. "
                       if _who else "")
                    + ("Where a session holds several reps, rep 1 still "
                       "carries the power ramp, so the fade above compares "
                       "the opening reps 2–3 with the closing two:"
                       if multi_rep else
                       "These are single efforts, so there is no within-set "
                       "fade to read — the dotted line is just that day's "
                       "effort:"))
        fig = _rep_bar_figure(pv)
        style_figure(
            fig,
            f"{fam_name} · {cls} — individual intervals, "
            f"session by session", H_STD)
        # after style_figure: it owns the margins the legend lane sits in
        _lane_legend(fig, ["Individual interval (avg W)", "Session average"],
                     min_w=430, cols=3)
        show(fig)
    else:
        callout("Individual interval rows not in the cache",
                "The raw interval rows of this series are not in the local "
                "cache (the set averages are), so the per-interval bars "
                "cannot be drawn — sync that activity to fill them in. The "
                "session-average line below is still exact.",
                C["yellow"], icon="⚠️")

    # 4b — session average over time + robust trend + CI band
    st.markdown("**Session average watts over time — one point per session, "
                "robust trend with its 95% CI:**")
    cb = st.columns(2)
    with cb[0]:
        on_temp = st.checkbox("Overlay temperature (right axis)",
                              key="iv_type_temp")
    with cb[1]:
        on_tsb = st.checkbox("Overlay TSB freshness — CTL − ATL (right axis)",
                             key="iv_type_tsb")
    fig = _session_trend_figure(pv, on_temp, on_tsb)
    style_figure(
        fig,
        f"{fam_name} · {cls} — session average watts, "
        f"{pv['n_sessions']} sessions", H_STD)
    # after style_figure: it owns the margins the legend lane sits in. Only an
    # explicit showlegend=False hides a trace — Plotly leaves the attribute
    # None when it was never set, and None means "show". Wrapped onto rows of
    # three in the TOP margin: a side lane here would sit on the y2 axis when
    # temperature or TSB is overlaid.
    _lane_legend(fig, [t.name for t in fig.data
                       if t.name
                       and getattr(t, "showlegend", None) is not False],
                 min_w=430, cols=3)
    show(fig)

    direction = ("genuinely improving" if clear and slope >= 1 else
                 "genuinely declining" if clear and slope <= -1 else
                 "too noisy to call — 95% CI includes zero"
                 if not np.isnan(slope) else
                 "not enough history for a trend")
    rest_note = ""
    if stt.get("rest_varies") and stt.get("reps_med", 1) >= 2:
        rest_note = (f" Measured recovery between reps varies across sessions "
                     f"(median {fmt_rest(stt.get('rest_med'))}) — same rep "
                     f"length, different recovery, so part of that spread is "
                     f"coaching choice rather than fitness.")
    callout("Plain reading",
            f"**{fam_name} at {cls}**: {direction}. "
            f"Session average {tr.get('first', float('nan')):.0f} W → "
            f"{tr.get('last', float('nan')):.0f} W "
            f"({tr.get('last', 0) - tr.get('first', 0):+.0f} W), best "
            f"{tr.get('best', float('nan')):.0f} W. "
            + (f"Robust trend {slope:+.1f} W/month, 95% CI ±{ci:.1f} — "
               if not np.isnan(slope) and not np.isnan(ci) else
               "No trend line — needs ≥ 3 sessions over a real time span. ")
            + f"n = {n_sets} sets / {pv['n_sessions']} sessions / "
              f"{pv['n_reps']} intervals. Intervals measured "
              f"{_span_txt(stt)}, ridden at {if_band}"
            + (f", median IF {if_med:.0f} %." if not np.isnan(if_med) else ".")
            + rest_note
            + " Same type, same duration class only — observed history, not "
              "a forecast and not a claim about why it changed.",
            C["muted"], icon="📖")
    if if_band == "sub-threshold <88%":
        callout("Ridden below threshold",
                "This series is shaped like threshold or VO₂ work, but the "
                "measured intensity is under 88 % of FTP in the median set. "
                "A coach reads that as tempo or an endurance block, not a "
                "VO₂/threshold set: the family name says which training type "
                "or protocol shape the series belongs to, this band says "
                "HOW HARD it was actually ridden.",
                C["yellow"], icon="🎚️")

    st.markdown(f"**Every session of {fam_name} at {cls}:**")
    dataframe(_session_table(sess), height=380)
    if len(bars):
        with st.expander(f"📋 All {pv['n_reps']} individual intervals",
                         expanded=False):
            dataframe(_rep_table(bars), height=460)
    with st.expander(f"📋 The {stt['n']} sets behind them — measured rest, "
                     f"IF, load", expanded=False):
        dataframe(series_detail(res["sets"], fam, dur_cls=cls), height=420)


# ── Section 1 tab A: identical sets over time ────────────────────────────────
def _render_sets(iv_full: pd.DataFrame, acts: pd.DataFrame, df_all) -> None:
    st.caption(
        "Reps sharing a `group_id` form one SET; sets match across sessions "
        "by rep duration × reps. Durations are whole minutes, and the exact "
        "measured rep length follows in seconds — `4 min · 240 s` and "
        "`4 min · 230 s` are different matches, because two sets are only "
        "ever compared with work inside 10 s of each other. Rest between "
        "reps is measured from the recovery rows and is always shown. Where "
        "you have filed the session under a training type, that label is what "
        "the family is called — for EVERY type, not just BILLAT, so these "
        "charts group by FTP / VO2MAX / AEROBIC BASE … as you filed them. "
        "Only sessions with no label fall back to a protocol heuristic from "
        "rep length + measured rest, always shown next to the raw values."
    )
    sets = build_sets(iv_full, acts, df_all)
    sets, _ = fill_peak_efforts(sets, df_all, iv=iv_full)
    if len(sets) < 3:
        callout("Not enough sets",
                "Fewer than 3 sets in the cached history — sync more "
                "activities first.", C["yellow"], icon="⏸️")
        return
    # same quality screen as the training-types tab: duplicate rides,
    # non-protocol groupings and detector artefacts never enter a comparison
    sets, q_report, q_dropped = quality_gates(sets)
    if len(sets) < 3:
        callout("Not enough sets after the quality screen",
                "Fewer than 3 usable sets remain — see the exclusions.",
                C["yellow"], icon="⏸️")
        return
    if len(q_dropped):
        callout(
            f"Data quality — {len(q_dropped)} sets excluded here too",
            f"{len(q_dropped)} of {len(sets) + len(q_dropped)} sets were "
            f"excluded by the same screen used in the training-types tab: "
            + "; ".join(f"{r['Reason']} — {r['Sets']} sets / "
                        f"{r['Sessions']} sessions"
                        for _, r in q_report[q_report["Action"] == "excluded"]
                        .iterrows())
            + f". {int(q_report.loc[q_report['Action'].str.startswith('flagged'), 'Sets'].sum())}"
              f" further sets are flagged for an unusable intensity field and "
              f"kept. Every excluded row is listed in the training-types tab.",
            C["yellow"], icon="🧹")
    _peak_disclosure(sets)
    sets, clusters = add_signatures(sets)

    sig_of, labels = {}, []
    for r in clusters.itertuples():
        style = f" · {r.style}" if r.style else ""
        lab = f"{r.sig} · {r.n_days} sessions · {r.n_sets} sets{style}"
        labels.append(lab)
        sig_of[lab] = r.sig
    prev = st.session_state.get("iv_sig")
    if prev is not None and prev not in sig_of:
        del st.session_state["iv_sig"]      # stale label after a re-sync
    chosen = st.selectbox("Matched set type (identical work)", labels,
                          key="iv_sig")
    sig = sig_of[chosen]
    res = run_set_evolution(sets[sets["sig"] == sig])
    if not res.get("ok"):
        callout("Not enough sets", res["reason"], C["yellow"], icon="⏸️")
        return
    s = res["sets"]
    months_span = max(res["span_days"] / 30.44, 0.1)

    # KPI row
    best_row = res["best_row"]
    slope = res["slope_m"]
    kc = st.columns(5)
    with kc[0]:
        metric_card("Sets matched", f"{res['n']}",
                    foot=f"{res['n_days']} sessions over "
                         f"{months_span:.0f} months",
                    accent=C["accent"])
    with kc[1]:
        metric_card("Best set", f"{res['best']:.0f} W",
                    foot=f"{int(best_row['reps'])} × "
                         f"{fmt_min(best_row['rep_secs'])} "
                         f"({float(best_row['rep_secs']):.0f} s) · "
                         f"{pd.Timestamp(best_row['date']).date()}",
                    accent=C["green"])
    with kc[2]:
        metric_card("Trend", f"{slope:+.1f} W/month",
                    foot=f"OLS vs time · r = {res['r_time']:+.2f} · "
                         f"{months_span:.0f} months",
                    tone="good" if slope >= 1 else "bad" if slope <= -1 else None,
                    accent=C["green"] if slope >= 1
                    else C["red"] if slope <= -1 else C["muted"])
    with kc[3]:
        db = res["d_best"]
        metric_card("Best, last 28 d",
                    f"{db:+.0f} W" if db is not None else "—",
                    foot="vs previous best of this set type"
                    if db is not None else "no earlier set in this match",
                    tone="good" if (db or 0) > 0 else "bad"
                    if db is not None else None,
                    accent=C["purple"])
    with kc[4]:
        metric_card("Last performed", str(res["last"].date()),
                    foot=f"{int(s.iloc[-1]['reps'])} × "
                         f"{fmt_min(s.iloc[-1]['rep_secs'])} "
                         f"({float(s.iloc[-1]['rep_secs']):.0f} s)",
                    accent=C["muted"])

    # New per-set metrics from intervals.icu's steady-state model
    has_new = any(c in s.columns for c in
                  ("set_ss_cp_w", "set_ss_w_prime_kj", "set_w5s_cv"))
    if has_new:
        st.caption(
            "🔬 **Per-set physiology (intervals.icu steady-state model):**  "
            "**CP (W)** / **W' (kJ)** — critical power & anaerobic work "
            "capacity fitted to the interval's internal power curve. "
            "**Only meaningful for long (>20 min), variable intervals** — "
            "steady threshold sets return model noise.  "
            "**w5s CV** — coefficient of variation of 5 s rolling power "
            "within the set. Lower = more reproducible effort.  "
            "3 % = very steady; 10 %+ = highly variable."
        )

    # Main evolution chart (+ optional context overlays, right axis)
    cb = st.columns(2)
    with cb[0]:
        on_temp = st.checkbox("Overlay temperature (right axis)",
                              key="iv_on_temp")
    with cb[1]:
        on_tsb = st.checkbox("Overlay TSB freshness — CTL − ATL (right axis)",
                             key="iv_on_tsb")

    fig = go.Figure()
    cd = np.array([
        s["name"].fillna("").astype(str).to_numpy(),
        s["reps"].to_numpy(),
        np.array([f"{v:.0f} °C" if pd.notna(v) else "—"
                  for v in s["temp"]], dtype=object),
        np.array([f"{v:+.0f}" if pd.notna(v) else "—"
                  for v in s["tsb"]], dtype=object),
    ], dtype=object).T
    fig.add_trace(go.Scatter(
        x=s["date"], y=s["set_w"], mode="markers", name="Set (avg W)",
        marker=dict(size=11, color=C["accent"],
                    line=dict(color=C["border"], width=1)),
        customdata=cd,
        hovertemplate="%{x|%d %b %Y} · %{y:.0f} W<br>"
                      "%{customdata[1]} reps · %{customdata[0]}<br>"
                      "TSB %{customdata[3]} · %{customdata[2]}"
                      "<extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=res["trend"]["date"], y=res["trend"]["y_hat"],
                             mode="lines", name="Time trend (OLS)",
                             line=dict(color=C["orange"], width=2,
                                       dash="dot")))
    y2 = False
    if on_temp:
        t = s.dropna(subset=["temp"]).groupby("date", sort=True)["temp"].mean()
        if len(t):
            fig.add_trace(go.Scatter(x=t.index, y=t.values, mode="lines",
                                     name="Temperature (°C)",
                                     line=dict(color=C["yellow"], width=2),
                                     yaxis="y2"))
            y2 = True
    if on_tsb:
        t = s.dropna(subset=["tsb"]).groupby("date", sort=True)["tsb"].mean()
        if len(t):
            fig.add_trace(go.Scatter(x=t.index, y=t.values, mode="lines",
                                     name="TSB (CTL − ATL)",
                                     line=dict(color=C["purple"], width=2,
                                               dash="dash"),
                                     yaxis="y2"))
            y2 = True
    if y2:
        fig.update_layout(yaxis2=dict(overlaying="y", side="right",
                                      showgrid=False))
    style_figure(fig,
                 f"{sig} over time — {res['n']} sets · {res['n_days']} "
                 f"sessions · {months_span:.0f} months", ML_CHART_H)
    legend(fig, "left", horizontal=True)
    show(fig)

    direction = ("improving" if slope >= 1
                 else "declining" if slope <= -1 else "essentially flat")
    callout("Plain reading",
            f"On this identical work you are **{direction}**: "
            f"{slope:+.1f} W/month over {months_span:.0f} months "
            f"({slope * months_span:+.0f} W end-to-end), Pearson "
            f"r = {res['r_time']:+.2f}, Spearman ρ = {res['rho']:+.2f}, "
            f"n = {res['n']} sets across {res['n_days']} sessions. "
            f"Descriptive fit of what happened — not a validated forecast.",
            C["muted"], icon="📖")

    # Fresh vs fatigued + context associations
    r1, r2 = st.columns(2)
    with r1:
        f = res["fatigue"]
        if f:
            figb = go.Figure()
            figb.add_trace(go.Box(
                y=f["fresh_vals"], name="Fresh legs (TSB ≥ 0)",
                marker_color=C["green"], marker_size=5, boxpoints="all",
                jitter=0.3, pointpos=0))
            figb.add_trace(go.Box(
                y=f["tired_vals"], name="Fatigued (TSB ≤ −10)",
                marker_color=C["red"], marker_size=5, boxpoints="all",
                jitter=0.3, pointpos=0))
            style_figure(figb, "The SAME set — fresh vs fatigued days",
                         H_STD, showlegend=False)
            show(figb)
            st.caption(f"n = {f['fresh_n']} fresh (median {f['fresh_med']:.0f} "
                       f"W) vs {f['tired_n']} fatigued (median "
                       f"{f['tired_med']:.0f} W) — observed sets only; who "
                       f"rides fresh isn't randomised (weather, course, "
                       f"equipment co-vary): association, not cause.")
        else:
            st.markdown("**Fresh vs fatigued**")
            tsb = pd.to_numeric(s["tsb"], errors="coerce")
            n_f = int((tsb >= 0).sum()) if tsb.notna().any() else 0
            n_t = int((tsb <= -10).sum()) if tsb.notna().any() else 0
            callout("Not enough sets per state",
                    f"Need ≥ 4 sets on each side of TSB 0 and −10 — here "
                    f"{n_f} fresh vs {n_t} fatigued. Pick a more common set "
                    f"type or a longer range.", C["yellow"], icon="⏸️")
    with r2:
        st.markdown("**What co-moves with set power**")
        if len(res["assoc"]):
            dataframe(res["assoc"], height=300)
        else:
            st.caption("No context variable has ≥ 8 paired observations "
                       "with enough variation — showing nothing beats "
                       "showing noise.")
        callout("How to read",
                "r = correlation of set power (after removing the time "
                "trend) with the context, −1…+1, n = paired observations. "
                "Association only — no causal claim; small n, and "
                "course/equipment co-vary.", C["yellow"], icon="⚠️")

    with st.expander(f"📋 All {res['n']} sets in this match",
                     expanded=False):
        # Build optional new columns only if present in the data
        opt_cols = {}
        if "set_ss_cp_w" in s.columns:
            opt_cols["CP (W)"] = pd.to_numeric(s["set_ss_cp_w"],
                                              errors="coerce").round(0)
        if "set_ss_w_prime_kj" in s.columns:
            opt_cols["W' (kJ)"] = pd.to_numeric(s["set_ss_w_prime_kj"],
                                               errors="coerce").round(2)
        if "set_w5s_cv" in s.columns:
            opt_cols["w5s CV"] = pd.to_numeric(s["set_w5s_cv"],
                                              errors="coerce").map(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")

        base = {
            "Date": s["date"].dt.strftime("%Y-%m-%d"),
            "Activity": s["name"].fillna("").astype(str),
            "Reps": s["reps"].astype(int),
            "Rep": s["rep_secs"].map(fmt_min),
            "Rep s": s["rep_secs"].map(fmt_secs),
            "Rest": s["rest"].map(fmt_rest),
            "Avg W": s["set_w"].round(0),
            "NP": s["set_np"].round(0),
            "HR": s["set_hr"].round(0),
            "IF %": s["intensity"].round(0),
            "Load": s["load"].round(0),
            "°C": s["temp"].round(1),
            "TSB": s["tsb"].round(1),
        }
        base.update(opt_cols)
        tbl = pd.DataFrame(base).iloc[::-1]
        dataframe(tbl, height=420)


# ── Section 1 tab B: evolution by duration class ─────────────────────────────
def _render_duration_evolution(iv_full: pd.DataFrame) -> None:
    st.caption(
        "Monthly BEST watts per duration class over the full cached history "
        "(WORK intervals). Each row is coloured by its own worst→best range "
        "— dark = that class's low month, green = its high month — with the "
        "actual watts printed in each cell; blank = no efforts that month."
    )
    res = run_duration_evolution(iv_full)
    if not res.get("ok"):
        callout("Not enough data", res["reason"], C["yellow"], icon="⏸️")
        return

    fig = go.Figure(go.Heatmap(
        z=res["z"], x=res["months"], y=res["labels"],
        customdata=res["count_arr"],
        colorscale=[[0.0, "#0d1117"], [0.5, "#1f6feb"], [1.0, "#3fb950"]],
        colorbar=dict(title="% of class range", thickness=12, len=0.8),
        hovertemplate="%{y} · %{x}<br>best that month<br>"
                      "n = %{customdata} efforts<extra></extra>",
    ))
    anns = []
    for i, lab in enumerate(res["labels"]):
        for j, mo in enumerate(res["months"]):
            txt = res["text"][i][j]
            if not txt:
                continue
            z = res["z"][i][j]
            anns.append(dict(
                x=mo, y=lab, text=txt, showarrow=False,
                font=dict(size=9, color="#0d1117" if z >= 62 else "#c9d1d9")))
    fig.update_layout(annotations=anns)
    fig.update_yaxes(categoryorder="array", categoryarray=res["labels"],
                     autorange="reversed")
    style_figure(fig,
                 f"Monthly best watts by duration class — "
                 f"{res['n_work']:,} WORK intervals",
                 ML_CHART_H, showlegend=False)
    show(fig)

    st.markdown("**Trend of the monthly bests (descriptive):**")
    dataframe(res["table"], height=440)
    st.caption(
        "W/month = OLS slope of that class's monthly bests (≥ 4 months of "
        "data); ↑ ≥ +1, ↓ ≤ −1, → in between. Descriptive history, not a "
        "validated forecast — monthly bests are noisy when a cell has few "
        "efforts (hover a cell for n)."
    )


# ── Section 2: exertion persistence (RPE block) ──────────────────────────────
def _render_exertion(acts: pd.DataFrame, df_all) -> None:
    section("⚖️ Exertion persistence — does hard feel like hard?")
    callout(
        "Why this is a strain proxy, not RPE",
        "Your RPE fields in intervals.icu are **empty for all 1,604 "
        "activities** — RPE has never been logged. This block therefore "
        "models **intervals.icu strain score** (independent of power-based "
        "TSS, r = 0.24) as the exertion proxy. Log RPE on a few activities "
        "and pick **RPE (your logged entries)** below — the same honest "
        "backtest then runs on the real thing.",
        C["yellow"], icon="ℹ️")

    ec = st.columns(2)
    with ec[0]:
        metric = st.selectbox(
            "Exertion metric",
            ["strain", "strain_rate", "trimp", "icu_rpe"],
            format_func=lambda k: {
                "strain": "Strain score (default proxy)",
                "strain_rate": "Strain per hour",
                "trimp": "TRIMP (HR-based)",
                "icu_rpe": "RPE (your logged entries)",
            }[k],
            key="ex_metric")
    with ec[1]:
        h = int(st.selectbox("Forecast horizon", ["7 days", "14 days", "28 days"],
                             index=1, key="ex_h").split()[0])

    counts = metric_counts(acts)
    st.caption(
        f"Observations available in range — strain: {counts['strain']} · "
        f"strain/h: {counts['strain_rate']} · TRIMP: {counts['trimp']} · "
        f"RPE: {counts['icu_rpe']}"
    )

    result = run_exertion_forecast(acts, df_all, h, metric, SURGERY)
    if not result.get("ok"):
        callout("Not enough data", result["reason"], C["yellow"], icon="⏸️")
        return

    obs, path = result["obs"], result["path"]
    fig = go.Figure()
    _add_fan(fig, path)
    fig.add_trace(go.Scatter(x=obs["date"], y=obs["y"], mode="lines+markers",
                             name=f"{result['metric_label']} (session)",
                             line=dict(color=C["muted"], width=1.5),
                             marker=dict(size=4, color=C["muted"])))
    style_figure(fig,
                 f"{result['metric_label']} — history & {h}-day forecast "
                 f"(selected: {result['selected_label']})",
                 ML_CHART_H)
    legend(fig, "left", horizontal=True)
    show(fig)

    _model_table(result)
    _model_calls(result)


# ── Section 3: interval performance forecast ─────────────────────────────────
def _render_band_forecast(iv: pd.DataFrame, df_all) -> None:
    section("🔮 Interval performance forecast")
    lo = float(st.session_state.get("ivb_lo", 3.0))
    hi = float(st.session_state.get("ivb_hi", 6.0))
    i_lo = float(st.session_state.get("ivi_lo2", 90.0))
    i_hi = float(st.session_state.get("ivi_hi2", 120.0))
    if lo > hi:
        lo, hi = hi, lo
    st.caption(
        f"Target: rolling 28-day best effort in the **same matched window as "
        f"the search tab above** ({lo:g}–{hi:g} min · {i_lo:.0f}–"
        f"{i_hi:.0f}% FTP), over the **full cached history** (a year of "
        f"intervals, not just the sidebar range) so the backtest has enough "
        f"marks. Baseline to beat: “my best stays where it is”."
    )
    h = int(st.selectbox("Forecast horizon",
                         ["4 weeks", "8 weeks", "12 weeks"],
                         index=0, key="bf_h").split()[0]) * 7

    result = run_band_forecast(iv, df_all, lo * 60, hi * 60, h,
                               i_lo=i_lo, i_hi=i_hi)
    if not result.get("ok"):
        callout("Not enough efforts in this band", result["reason"],
                C["yellow"], icon="⏸️")
        return

    obs, path = result["obs"], result["path"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=obs["date"], y=obs["y"], mode="lines+markers",
                             name="28-day best (avg W)",
                             line=dict(color=C["muted"], width=2),
                             marker=dict(size=5, color=C["muted"])))
    _add_fan(fig, path, rgb="63,185,80")     # green fan, distinct from exertion
    style_figure(fig,
                 f"Best {result['band_label']} effort — {result['n_efforts']} "
                 f"efforts, {h}-day forecast "
                 f"({result['selected_label'].split('—')[0].strip()})",
                 ML_CHART_H)
    legend(fig, "left", horizontal=True)
    show(fig)

    _model_table(result)
    _model_calls(result)


# ── Section 4: composition re-evaluated ──────────────────────────────────────
def _coef_bar(tbl: pd.DataFrame, title: str) -> go.Figure:
    vals = [float(str(v).replace("+", "")) for v in tbl["Std. weight"]]
    fig = go.Figure(go.Bar(
        x=vals, y=tbl["Feature"][::-1], orientation="h",
        marker_color=[C["green"] if v >= 0 else C["red"] for v in vals[::-1]],
        hovertemplate="%{y}: %{x:+.2f}<extra></extra>",
    ))
    style_figure(fig, title, H_STD, showlegend=False)
    fig.update_layout(margin=dict(l=210))   # room for feature names
    fig.update_xaxes(title="Standardised weight")
    return fig


def _render_composition(df_all, iv: pd.DataFrame) -> None:
    section("📊 Composition re-evaluated with real intervals")
    st.caption(
        "The old analysis asked: does **% TSS per training type** in a "
        "window predict next-window eFTP? This re-run asks the same question "
        "with **real interval composition** (minutes by duration band, "
        "work:rest, interval TSS share) — both feature sets compete in the "
        "same time-ordered gauntlet against persistence."
    )
    h = int(st.selectbox("Target horizon", ["28 days", "56 days"],
                         index=0, key="comp_h").split()[0])

    result = run_composition(df_all, iv, horizon_days=h)
    if not result.get("ok"):
        callout("Not enough data", result["reason"], C["yellow"], icon="⏸️")
        return

    rec = result["records"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec["Window end"], y=rec["Actual eFTP"],
                             mode="markers", name="Actual eFTP (verified)",
                             marker=dict(size=9, color=C["accent"])))
    fig.add_trace(go.Scatter(x=rec["Window end"], y=rec["Persistence"],
                             mode="lines", name="Persistence forecast",
                             line=dict(color=C["muted"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=rec["Window end"], y=rec["Interval model"],
                             mode="lines", name="Interval-composition model",
                             line=dict(color=C["green"], width=2.5)))
    fig.add_trace(go.Scatter(x=rec["Window end"], y=rec["Old type-% model"],
                             mode="lines", name="Old type-% model",
                             line=dict(color=C["orange"], width=2)))
    style_figure(fig,
                 f"Backtests — next-window eFTP ({h} d): what each model "
                 f"predicted vs what happened", ML_CHART_H)
    legend(fig, "left", horizontal=True)
    show(fig)

    _model_table(result)

    cc = st.columns(2)
    with cc[0]:
        show(_coef_bar(result["coef_iv"],
                       "Interval composition — weights"))
    with cc[1]:
        show(_coef_bar(result["coef_ty"],
                       "Old type-% composition — weights"))

    callout("Why this model", result["why"], C["accent"], icon="🧠")
    callout("Honest caveats", result["note"], C["yellow"], icon="⚠️")


# ── Section 5: VO₂max estimate from interval power ───────────────────────────
def _render_vo2(iv_full: pd.DataFrame) -> None:
    section("🫁 VO₂max estimate from interval power")
    st.caption(
        "Watts → oxygen cost through gross efficiency (no lab test exists, "
        "so the assumption is yours to move). Best 3–6 min power marks "
        "power-at-VO₂max (PVO₂); confidence is judged from the data itself "
        "— did the best window effort's HR reach your all-time max?"
    )

    age, sex, prof_w = None, None, None
    try:
        prof = fetch_profile()
        dob = prof.get("icu_date_of_birth") or prof.get("date_of_birth")
        if dob:
            age = int((pd.Timestamp.now().normalize()
                       - pd.Timestamp(dob)).days // 365)
        sex = prof.get("sex")
        prof_w = prof.get("icu_weight") or prof.get("weight")
    except Exception:
        pass

    cc = st.columns(2)
    with cc[0]:
        weight = float(st.number_input("Body weight (kg)", 35.0, 120.0, 57.0,
                                       step=0.5, key="vo2_weight"))
    with cc[1]:
        ge_pct = float(st.slider("Gross efficiency assumption (%)", 18.0,
                                 24.0, 21.0, step=0.5, key="vo2_ge"))
    w_note = (f" · profile weight {prof_w:.1f} kg (Strava sync)"
              if prof_w else "")
    age_s = f"{age} y old" if age is not None else "age not in profile"
    st.caption(f"Profile: {sex or '—'} · {age_s}{w_note}")

    res = run_vo2_estimate(iv_full, weight, ge_pct=ge_pct, age=age)
    if not res.get("ok"):
        callout("Not enough data", res["reason"], C["yellow"], icon="⏸️")
        return

    kc = st.columns(4)
    with kc[0]:
        metric_card("PVO₂ proxy", f"{res['pvo2_w']:.0f} W",
                    foot=f"{res['wkg_pvo2']:.2f} W/kg · "
                         f"{res['pvo2_basis']}",
                    accent=C["accent"])
    with kc[1]:
        metric_card("VO₂max (est.)", f"{res['vo2_l']:.2f} L/min",
                    foot=f"GE {res['ge_pct']:.1f} % assumed",
                    accent=C["purple"])
    with kc[2]:
        metric_card("Relative VO₂max", f"{res['vo2_mlkg']:.0f} mL/kg/min",
                    foot=res["anchor"], accent=C["green"])
    with kc[3]:
        metric_card("Effort confidence", res["conf"],
                    foot=f"{res['n_window']} efforts in 3–6 min window",
                    tone="good" if res["conf"] == "Strong" else
                    "bad" if res["conf"] == "Low" else None,
                    accent=C["green"] if res["conf"] == "Strong"
                    else C["red"] if res["conf"] == "Low" else C["yellow"])

    # Power–duration curve with an implied-VO2 axis on the right
    curve = res["curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["secs"], y=curve["watts"], mode="lines+markers",
        name="Mean-max power (avg W)",
        line=dict(color=C["accent"], width=3),
        marker=dict(size=9, color=C["accent"]),
        hovertemplate="%{y:.0f} W · %{x:.0f} s<extra></extra>"))
    winp = curve[(curve["secs"] >= 180) & (curve["secs"] <= 360)
                 & curve["watts"].notna()]
    if len(winp):
        fig.add_trace(go.Scatter(
            x=winp["secs"], y=winp["watts"], mode="markers",
            name="PVO₂ window (3–6 min)",
            marker=dict(size=14, color=C["green"],
                        line=dict(color=C["border"], width=1)),
            hovertemplate="%{y:.0f} W · %{x:.0f} s (VO₂ window)"
                          "<extra></extra>"))
    ymin, ymax = float(np.nanmin(curve["watts"])), float(
        np.nanmax(curve["watts"]))
    wt = [v for v in (100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700)
          if ymin - 50 <= v <= ymax + 50]
    style_figure(fig,
                 f"Power–duration curve → VO₂max {res['vo2_mlkg']:.0f} "
                 f"mL/kg/min @ GE {res['ge_pct']:.1f} %",
                 ML_CHART_H)
    fig.add_hline(y=res["pvo2_w"], line_dash="dot", line_width=2,
                  line_color=C["green"])
    fig.add_annotation(xref="paper", x=0.99, yref="y", y=res["pvo2_w"],
                       text=f"PVO₂ {res['pvo2_w']:.0f} W → "
                            f"{res['vo2_l']:.2f} L/min",
                       showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(color=C["green"], size=13),
                       bgcolor="rgba(13,17,23,0.85)",
                       bordercolor=C["green"], borderwidth=1,
                       borderpad=4)
    fig.update_xaxes(type="log", title="Mean-max duration (log)",
                     tickvals=list(curve["secs"]),
                     ticktext=list(curve["label"]))
    fig.update_layout(yaxis2=dict(
        overlaying="y", side="right", showgrid=False, title="L/min O₂",
        tickvals=wt,
        ticktext=[f"{vo2_l_per_min(v, ge_pct / 100):.1f}" for v in wt]))
    legend(fig, "left", horizontal=True)
    show(fig)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Uncertainty — gross-efficiency band**")
        dataframe(res["sens"], height=170)
        st.markdown("**Window durations disagree (anaerobic contribution)**")
        dataframe(res["dur_table"], height=170)
    with r2:
        st.markdown("**Evidence: strongest 3–6 min efforts vs your HR max**")
        dataframe(res["evidence"], height=330)
        hr_bits = []
        if not np.isnan(res["alltime_hr"]):
            hr_bits.append(f"all-time max HR in data "
                           f"{res['alltime_hr']:.0f} bpm")
        if not np.isnan(res["age_pred"]):
            hr_bits.append(f"age-predicted (208 − 0.7×age) "
                           f"{res['age_pred']:.0f} bpm")
        callout("Effort confidence",
                f"**{res['conf']}** — {res['conf_why']}."
                + (f" Reference: {' · '.join(hr_bits)}." if hr_bits else ""),
                C["green"] if res["conf"] == "Strong" else C["yellow"],
                icon="🫀")

    callout("How this was computed", res["formula"], C["accent"], icon="🧮")
    callout("Honest caveats", res["caveats"], C["yellow"], icon="⚠️")


# ── Page entry ───────────────────────────────────────────────────────────────
def render(head, ctx):
    df_all = ctx.df_all

    page_header(
        "🔬",
        "Intervals — like-for-like effort comparison",
        "Interval-level data straight from intervals.icu: match similar "
        "efforts, test whether exertion persists, forecast your best in a "
        "band, re-evaluate composition with real interval rows, and "
        "estimate VO₂max from the power–duration curve.",
    )

    # ML blocks (band forecast, composition) need history, not just the
    # sidebar range — sync a full year into the append-only cache, then scope
    # only the compare section to the sidebar cutoff.
    now = pd.Timestamp.now().normalize()
    range_iso = pd.Timestamp(ctx.cutoff).normalize().isoformat()
    sync_lo = min(pd.Timestamp(ctx.cutoff).normalize(),
                  now - pd.Timedelta(days=370))
    sync_iso = sync_lo.isoformat()
    try:
        with st.spinner("Syncing interval rows from intervals.icu "
                        "(first run pulls every activity in range)…"):
            bar = st.progress(0.0, text="Fetching per-activity intervals…")

            def _p(done, total):
                bar.progress(done / max(total, 1),
                             text=f"Fetching intervals {done}/{total}")

            meta = sync_intervals(sync_iso, on_progress=_p)
            bar.empty()
        acts = fetch_acts(range_iso)
        iv_full = read_intervals(sync_iso)
        iv = iv_full[iv_full["date"] >= pd.Timestamp(range_iso)].reset_index(
            drop=True)
    except Exception as exc:                      # API down / auth problem
        st.error(f"Interval sync failed: {exc}")
        st.stop()

    sync_note = (f"new: {meta['new_rows']} rows" if meta["fetched"]
                 else "up to date")
    st.caption(
        f"📡 {len(iv_full):,} interval rows cached · "
        f"{meta['window_activities']} activities with detected intervals · "
        f"{len(iv):,} rows inside the sidebar range ({sync_note}) — "
        f"append-only `data/interval_cache.csv`"
    )

    if not len(iv_full):
        callout("No interval rows yet",
                "intervals.icu has not detected intervals for activities in "
                "this range — widen the time range in the sidebar.",
                C["yellow"], icon="⏸️")
        return

    _render_compare(iv, acts, iv_full, df_all)
    _render_exertion(acts, df_all)
    _render_band_forecast(iv_full, df_all)
    _render_composition(df_all, iv_full)
    _render_vo2(iv_full)
