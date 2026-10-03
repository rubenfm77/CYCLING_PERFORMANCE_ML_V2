# views/forecast.py — Forecast page: PMC projection + validated eFTP outlook.
# (NEW FILE)  The "forecast fitness & adapt the plan" feature, done honestly:
# exact EWMA bookkeeping for the plan, walk-forward validated ML for eFTP.

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ml.ftp_forecast as fc
import ml.pmc_projection as pmc
from core.components import (callout, dataframe, legend, metric_card,
                             page_header, section, show)
from core.theme import (C, FTP_CURRENT, FTP_TARGET, H_HERO, H_STD,
                        SURGERY, WEIGHT_KG, style_figure)


def _form_words(tsb: float) -> str:
    if tsb < -30:
        return "overreached"
    if tsb < -10:
        return "deep block"
    if tsb < 10:
        return "building"
    if tsb < 25:
        return "neutral"
    if tsb < 40:
        return "fresh"
    return "very fresh / detraining risk"


def _render_pmc(df_all) -> None:
    section("🗓️ PMC projection — what your plan does to the meters")
    st.caption(
        "Exact EWMA continuation from today's actual CTL/ATL (same α as the "
        "dashboard: 2/43 and 2/8) under the TSS you plan to ride. "
        "Set the plan — see where fitness, fatigue and form end up."
    )

    wc = st.columns([2, 3, 2])
    with wc[0]:
        weekly = st.number_input(
            "Planned weekly TSS", min_value=0, max_value=2500,
            value=int(pmc.recent_weekly_tss(df_all)), step=25, key="_fc_weekly",
            help="Default = rounded mean of your last 28 days.",
        )
    with wc[1]:
        pattern = st.selectbox("Weekly pattern", list(pmc.PATTERNS),
                               key="_fc_pattern")
    with wc[2]:
        h_choice = st.selectbox("Projection horizon",
                                ["21 days", "28 days", "42 days"],
                                index=1, key="_fc_h")
    horizon = int(h_choice.split()[0])

    proj = pmc.project(df_all, weekly, horizon, pmc.PATTERNS[pattern])
    m = pmc.plan_metrics(proj, horizon)
    recent_weekly = pmc.recent_weekly_tss(df_all)

    # ── Projection chart ────────────────────────────────────────────────────
    today = pd.Timestamp.now().normalize()
    # Seam = newest ride row: where the solid line ends AND the dashes
    # begin. Anchoring the line/shading at midnight instead put it left
    # of the 12:52 ride — read as a discontinuity.
    _seam = pd.Timestamp(proj["date"].iloc[0])
    # 120-day history: the session-row EWMA reads as a proper PMC over a
    # quarter (a 35-day window looked like a seismograph), and the actual
    # line shows the climb the last month earned. vertical_spacing=0.12
    # reserves a slim band between panels — that band hosts the legend.
    hist = df_all[df_all["date"] >= today - pd.Timedelta(days=120)]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45],
        vertical_spacing=0.12)
    # Row 1 trace order is load-bearing:
    #   actuals → recent-pace corridor UNDER → plan dash ON TOP.
    # Plan defaults to your recent TSS (600 = 600), so plan and recent
    # pace carry nearly identical values — thin line over thin line and
    # one swallows the other ("lines missing"). Recent-pace is therefore
    # a WIDE translucent corridor (width 9, ~0.3 alpha): when the two
    # coincide you see a gray halo hugging the colored dash — both lines
    # visible; move the slider and the corridor visibly separates.
    # Corridor 9 / plan 4.5 leaves a 2.25px halo per side when they sit
    # on top of each other (was 6/3 = 1.5px) — bolder on both counts.
    # Every drawn line is in the legend (ATL/TSB recent included).
    # ATL actual carries the same fill band as the Fitness PMC.
    # row 1 — fitness & fatigue
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["ctl"], name="CTL actual",
                             line=dict(color=C["green"], width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["atl"], name="ATL actual",
                             line=dict(color=C["orange"], width=2.5),
                             fill="tonexty", fillcolor="rgba(240,136,62,0.1)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=proj["date"], y=proj["ctl_recent"],
                             name="CTL — recent pace", opacity=0.3,
                             line=dict(color=C["muted"], width=9)), row=1, col=1)
    fig.add_trace(go.Scatter(x=proj["date"], y=proj["atl_recent"],
                             name="ATL — recent pace", opacity=0.3,
                             line=dict(color=C["muted"], width=9)), row=1, col=1)
    fig.add_trace(go.Scatter(x=proj["date"], y=proj["ctl_plan"],
                             name="CTL — your plan",
                             line=dict(color=C["green"], width=4.5, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=proj["date"], y=proj["atl_plan"],
                             name="ATL — your plan",
                             line=dict(color=C["orange"], width=4.5, dash="dash")), row=1, col=1)
    # row 2 — form
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["tsb"], name="TSB actual",
                             line=dict(color=C["accent"], width=2.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=proj["date"], y=proj["tsb_recent"],
                             name="TSB — recent pace", opacity=0.3,
                             line=dict(color=C["muted"], width=9)), row=2, col=1)
    fig.add_trace(go.Scatter(x=proj["date"], y=proj["tsb_plan"],
                             name="TSB — your plan",
                             line=dict(color=C["accent"], width=4.5, dash="dash")), row=2, col=1)
    fig.add_hline(y=-30, line_dash="dash", line_color=C["red"],
                  annotation_text="−30 overreach", row=2, col=1)
    fig.add_hline(y=25, line_dash="dash", line_color=C["purple"],
                  annotation_text="+25 peak", row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=C["muted"], row=2, col=1)
    # BOLD seam: a solid full-height yellow wall (was a faint dashed line
    # at 0.8 opacity — easy to read as grid noise).
    fig.add_vline(x=_seam, line_dash="solid", line_width=3,
                  line_color=C["yellow"], opacity=1.0, row="all")
    # Shade everything after the last ride so the projected region is
    # unmistakable. Must run AFTER the traces: add_vrect defaults to
    # exclude_empty_subplots=True and silently drops the rect when no
    # data exists yet (plotly 6.8).
    # 0.10 alpha read as "the same chart" — 0.20 plus a dotted cyan
    # border makes the projected half a visibly separate room.
    fig.add_vrect(x0=_seam, x1=proj["date"].iloc[-1],
                  fillcolor="rgba(88,166,255,0.20)",
                  line_width=1.5, line_color="rgba(88,166,255,0.55)",
                  line_dash="dot", layer="below", row="all")
    style_figure(fig, None, H_HERO, showlegend=True)
    fig.update_annotations(font=dict(color=C["text"], size=12))
    # Right margin widened for the horizon pill labels below.
    fig.update_layout(hovermode="x unified", margin=dict(r=88))
    # Seam label sits on the same x as its line (was midnight — 4px left
    # of the actual join, reading as a gap). Now a pill so it reads as a
    # marker rather than stray text.
    fig.add_annotation(x=_seam, xref="x", y=1.0, yref="y domain",
                       text="today", showarrow=False,
                       xanchor="left", yanchor="bottom", xshift=6,
                       font=dict(color=C["yellow"], size=13),
                       bgcolor="rgba(13,17,23,0.92)",
                       bordercolor=C["yellow"], borderwidth=1, borderpad=3)
    # Quantified line ends: where each dashed line LANDS at the horizon, so
    # direction can't be misread ("they drop" → by how much, and which).
    # Delta included because the lines are nearly flat at plan ≈ recent
    # pace — "+2" tells the story the eye can't see.
    _end = proj["date"].iloc[-1]
    _ctl_end = float(proj["ctl_plan"].iloc[-1])
    _atl_end = float(proj["atl_plan"].iloc[-1])
    _tsb_end = float(proj["tsb_plan"].iloc[-1])
    _atl_now = m["anchor_ctl"] - m["anchor_tsb"]
    _atl_above = _atl_end >= _ctl_end
    # Horizon pills: bigger type + a bordered chip so the projected end
    # points are the loudest thing on their row (bare 12px text read as
    # axis furniture).
    fig.add_annotation(x=_end, y=_ctl_end, xref="x", yref="y",
                       text=f"{_ctl_end:.0f} ({_ctl_end - m['anchor_ctl']:+.0f})",
                       showarrow=False,
                       xanchor="left", xshift=12,
                       yshift=-16 if _atl_above else 16,
                       font=dict(color=C["green"], size=15),
                       bgcolor="rgba(13,17,23,0.92)",
                       bordercolor=C["green"], borderwidth=1, borderpad=4)
    fig.add_annotation(x=_end, y=_atl_end, xref="x", yref="y",
                       text=f"{_atl_end:.0f} ({_atl_end - _atl_now:+.0f})",
                       showarrow=False,
                       xanchor="left", xshift=12,
                       yshift=16 if _atl_above else -16,
                       font=dict(color=C["orange"], size=15),
                       bgcolor="rgba(13,17,23,0.92)",
                       bordercolor=C["orange"], borderwidth=1, borderpad=4)
    fig.add_annotation(x=_end, y=_tsb_end, xref="x2", yref="y2",
                       text=f"{_tsb_end:.0f} ({_tsb_end - m['anchor_tsb']:+.0f})",
                       showarrow=False,
                       xanchor="left", xshift=12,
                       font=dict(color=C["accent"], size=15),
                       bgcolor="rgba(13,17,23,0.92)",
                       bordercolor=C["accent"], borderwidth=1, borderpad=4)
    # Watermark inside the projected half of BOTH panels: the future side
    # labels itself, so "before vs after today" can't be missed even at a
    # glance. Sits below the meter lines (y domain 0.35) where nothing
    # else is drawn.
    _mid = _seam + (_end - _seam) / 2
    fig.add_annotation(x=_mid, y=0.35, xref="x", yref="y domain",
                       text="PROJECTED", textangle=-90, showarrow=False,
                       font=dict(size=30, color="rgba(88,166,255,0.5)"))
    fig.add_annotation(x=_mid, y=0.35, xref="x2", yref="y2 domain",
                       text="PROJECTED", textangle=-90, showarrow=False,
                       font=dict(size=22, color="rgba(88,166,255,0.5)"))
    for i in (1, 2):
        # Controlled date ticks: every 28 days, horizontal — auto-scaling
        # was rotating labels 90° and cramming them together. Date axes
        # take dtick in milliseconds ("D28" gets silently coerced to 1 day).
        fig.update_xaxes(gridcolor=C["grid"], row=i, col=1,
                         tickangle=0, tickformat="%d %b",
                         dtick=28 * 24 * 60 * 60 * 1000)
        fig.update_yaxes(gridcolor=C["grid"], row=i, col=1)
    # Legend: horizontal, parked in the band BETWEEN the two panels — never
    # over the data, never clipped (the failure mode that forced the
    # original inside-canvas rule).
    y1b = fig.layout.yaxis.domain[0]
    y2t = fig.layout.yaxis2.domain[1]
    fig.update_layout(legend=dict(
        orientation="h", x=0.5, xanchor="center",
        y=(y1b + y2t) / 2, yanchor="middle",
        bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d",
        font=dict(size=10, color="#c9d1d9")))
    show(fig)

    # Decode the line directions where the eye lands: "the lines drop" must
    # never be misread as fitness falling when it is fatigue clearing.
    st.caption(
        f"**Reading the lines after today** — orange **ATL** (fatigue) "
        f"{_atl_now:.0f} → {_atl_end:.0f} and blue **TSB** (form) "
        f"{m['anchor_tsb']:+.0f} → {m['tsb_end']:+.0f} tell the same story "
        f"twice: fatigue from today's ride clears and form returns (flip if "
        f"the plan pushes harder). Green **CTL** (fitness) "
        f"{m['anchor_ctl']:.0f} → {_ctl_end:.0f} ({m['ramp_word']}) because "
        f"the plan is **{weekly} TSS/wk vs your recent {recent_weekly}** — "
        f"the good month you just trained is already inside today's "
        f"{m['anchor_ctl']:.0f} (that climb is the solid line); the dashed "
        f"part only answers “if the next {horizon} days look like this "
        f"plan, where do the meters point?”. Backing off for winter? Drop "
        f"the slider — the decay is drawn just as honestly."
    )

    # ── Projection verdict cards ────────────────────────────────────────────
    cc = st.columns(4)
    with cc[0]:
        metric_card("Peak CTL in plan", f"{m['peak']:.0f}",
                    delta=f"{m['peak_delta']:+.0f} vs today",
                    tone="good" if m["peak_delta"] >= 0 else "bad",
                    foot=f"over {horizon} days · {weekly} TSS/wk",
                    accent=C["green"])
    with cc[1]:
        metric_card("CTL ramp / week", f"{m['ramp_wk']:+.1f}",
                    delta=m["ramp_word"], tone=m["tone"],
                    foot=f"{m['pct_week']:+.1f}% per week · ≤10% advised",
                    accent=C["accent"])
    with cc[2]:
        metric_card("TSB at horizon", f"{m['tsb_end']:+.0f}",
                    delta=f"{m['tsb_delta']:+.0f} vs today",
                    tone="good" if m["tsb_delta"] >= 0 else "bad",
                    foot=f"lands {_form_words(m['tsb_end'])}",
                    accent=C["purple"])
    with cc[3]:
        metric_card("Taper window (TSB ≥ +10)",
                    m["taper"].strftime("%d %b") if m["taper"] is not None
                    else "beyond horizon",
                    foot="first date form opens — under this plan",
                    accent=C["orange"])

    st.caption(
        f"Plan: **{weekly} TSS/week** ({pattern.lower()}) vs your recent "
        f"**{recent_weekly} TSS/week** · the recursion steps only on ride days "
        f"(~{m['cadence']:.1f}/week lately → "
        f"**{m['per_ride']:.0f} TSS per ride** under this plan) · gray dotted "
        f"= continuing the last 28 days unchanged · anchor = the chart's own "
        f"CTL {m['anchor_ctl']:.1f} / TSB {m['anchor_tsb']:+.1f} as of your "
        f"newest ride · window: last 120 days."
    )
    callout(
        "What this projection is — and isn't",
        "It is the dashboard's own EWMA continued forward: the recursion "
        "steps **only when a session row exists** — ride days step, rest "
        "days and any gap since your newest ride hold flat — exactly like "
        "the Fitness chart. Consequence of that same chart math: "
        "steady-state CTL ≈ mean TSS **per ride**, so the same weekly "
        "budget spread over more, shorter rides lowers this meter (fewer, "
        "bigger rides raise it) — that is the chart's behaviour, not "
        "physiology. Bookkeeping, not prediction: it does not model "
        "adaptation, freshness or race form; it answers “if I ride this "
        "plan, where will CTL/ATL/TSB point?” Ramp guidance applies to "
        "|slope| both ways: ≤ +5 CTL/week fine, ≤ 8 moderate, above = "
        "aggressive build — or fast detraining on the way down.",
        C["accent"], icon="🧮")


def _render_eftp(df_all) -> None:
    section("🎯 eFTP forecast — validated against doing nothing")
    st.caption(
        "Three candidates — persistence (do nothing), drift (recent trend), "
        "ridge (causal 28-day load features) — walk-forward tested at your "
        "chosen horizon. The most accurate one wins; if none beats “stays "
        "put”, you see that instead of a fake number."
    )

    hc = st.columns([2, 3])
    with hc[0]:
        eh = st.selectbox("Forecast horizon",
                          ["4 weeks", "8 weeks", "12 weeks"],
                          index=1, key="_fc_eh")
        days = {"4 weeks": 28, "8 weeks": 56, "12 weeks": 84}[eh]
    with hc[1]:
        st.caption(
            "Target: Intervals.icu rolling eFTP (icu_rolling_ftp / eftp). "
            "Features: your PMC + 28-day load mix, as-of each date. "
            "Coefficients, backtests and the losing models are all shown below."
        )

    res = fc.run_forecast(df_all, days, SURGERY)
    if not res["ok"]:
        callout("Not enough validated history", res["reason"],
                C["accent"], icon="ℹ️")
        return

    obs, path = res["obs"], res["path"]

    # ── History + fan chart ─────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=obs["date"], y=obs["eftp"], mode="lines+markers",
        name="eFTP (intervals.icu)",
        line=dict(color=C["accent"], width=2.5), marker=dict(size=5)))
    # Fan bands: alpha roughly doubled and each band edge given a crisp
    # stroke — a flat 0.10/0.20 wash was invisible against the panel.
    _edge_lo = "rgba(88,166,255,0.5)"
    _edge_hi = "rgba(88,166,255,0.85)"
    fig.add_trace(go.Scatter(x=path["date"], y=path["lo90"], showlegend=False,
                             line=dict(width=1, color=_edge_lo)))
    fig.add_trace(go.Scatter(x=path["date"], y=path["hi90"], name="90% band",
                             fill="tonexty", fillcolor="rgba(88,166,255,0.18)",
                             line=dict(width=1, color=_edge_lo)))
    fig.add_trace(go.Scatter(x=path["date"], y=path["lo50"], showlegend=False,
                             line=dict(width=1, color=_edge_hi)))
    fig.add_trace(go.Scatter(x=path["date"], y=path["hi50"], name="50% band",
                             fill="tonexty", fillcolor="rgba(88,166,255,0.36)",
                             line=dict(width=1, color=_edge_hi)))
    # Forecast line 2.5 → 4.5 so the projected path reads at the same
    # weight as the PMC plan dashes above it.
    fig.add_trace(go.Scatter(x=path["date"], y=path["y"],
                             name=f"Forecast ({days // 7} weeks)",
                             line=dict(color=C["accent"], width=4.5, dash="dash")))
    fig.add_vline(x=pd.Timestamp(SURGERY), line_dash="dash",
                  line_color=C["red"], opacity=0.7)
    # "today" moves to the BOTTOM of its line: at the top it collided
    # with the top-right legend (the garbled "Target F…day…" text).
    fig.add_vline(x=pd.Timestamp.now().normalize(), line_dash="solid",
                  line_width=2.5, line_color=C["yellow"], opacity=1.0,
                  annotation_text="today", annotation_position="bottom")
    fig.add_trace(go.Scatter(
        x=[res["endpoint"]], y=[res["yhat"]], mode="markers",
        name="Projection", marker=dict(color=C["accent"], size=20,
                                       symbol="star",
                                       line=dict(color="#ffffff", width=1.5)))
    )
    # Endpoint value as a bordered pill hovering over the star — bigger,
    # boxed, and lifted clear of the marker (bare trace text at 12px was
    # easy to miss and could sit on the fan).
    fig.add_annotation(x=res["endpoint"], y=res["yhat"],
                       text=f"{res['yhat']:.0f} W",
                       showarrow=False, xanchor="center", yanchor="bottom",
                       yshift=12,
                       font=dict(color=C["accent"], size=15),
                       bgcolor="rgba(13,17,23,0.92)",
                       bordercolor=C["accent"], borderwidth=1, borderpad=4)
    fig.add_hline(y=FTP_CURRENT, line_dash="dot", line_color=C["yellow"])
    fig.add_hline(y=FTP_TARGET, line_dash="dot", line_color=C["green"],
                  opacity=0.4)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color=C["yellow"], width=2, dash="dot"),
                             name=f"Current FTP ({FTP_CURRENT} W)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color=C["green"], width=2, dash="dot"),
                             name=f"Target FTP ({FTP_TARGET} W)"))
    style_figure(fig, f"eFTP — history and {days // 7}-week forecast", H_STD)
    legend(fig, "right")
    show(fig)

    # ── Forecast verdict cards ──────────────────────────────────────────────
    end_row = path.iloc[-1]
    sel, sk = res["selected"], res["skill"][res["selected"]]
    cc = st.columns(4)
    with cc[0]:
        metric_card("Current eFTP", f"{res['y_now']:.0f} W",
                    foot=f"last test {res['last_date'].strftime('%d %b %Y')} · "
                         f"{res['y_now'] / WEIGHT_KG:.2f} W/kg")
    with cc[1]:
        delta_pct = (res["yhat"] / res["y_now"] - 1) * 100 if res["y_now"] else 0
        metric_card(f"Forecast in {days // 7} weeks", f"{res['yhat']:.0f} W",
                    delta=f"{res['yhat'] - res['y_now']:+.0f} W ({delta_pct:+.1f}%)",
                    tone="good" if res["yhat"] >= res["y_now"] else "bad",
                    foot=f"{res['yhat'] / WEIGHT_KG:.2f} W/kg projected",
                    accent=C["green"])
    with cc[2]:
        metric_card("90% range",
                    f"{end_row['lo90']:.0f}–{end_row['hi90']:.0f} W",
                    foot=f"50% band: {end_row['lo50']:.0f}–{end_row['hi50']:.0f} W",
                    accent=C["purple"])
    with cc[3]:
        if sel == "persistence":
            delta, tone = "baseline won", None
        else:
            delta = f"{sk * 100:+.0f}% vs doing nothing"
            tone = "good" if sk > 0 else "bad"
        metric_card("Skill (walk-forward)", delta, tone=tone,
                    foot=f"{res['selected_label'].split('—')[0].strip()} · "
                         f"{res['n_backtests']} backtests",
                    accent=C["green"] if tone == "good" else C["accent"])

    st.caption(f"{res['why']}  \n{res['band_note']}  \n"
               f"History: {res['n_obs']} observations "
               f"({res['first_date'].strftime('%b %Y')} → "
               f"{res['last_date'].strftime('%b %Y')}).")

    # ── The audit trail: every model, every weight ──────────────────────────
    st.markdown("**Model comparison — what was tested, what won**")
    dataframe(res["model_table"], height=156)
    if res["coef_table"] is not None:
        st.markdown("**🧠 What the selected ridge model weighs "
                    "(standardised weights)**")
        dataframe(res["coef_table"], height=270)
    st.caption(res["note"])

    callout(
        "How honest is this forecast?",
        "Time-ordered only: every backtest trains on targets already in the "
        "past at issue time (embargo by target date) — no random folds, no "
        "leakage. It must beat persistence to earn its place, and its skill "
        "is shown even when negative. Bands describe historical model error "
        "(widened √time), not every future risk — an FTP test still beats "
        "any model.", C["accent"], icon="⚖️")


def _render_reconcile(df_all) -> None:
    section("🔍 eFTP reconciliation — three ways to price your fitness")
    st.caption(
        "Your best 20-minute power, Intervals' rolling model and your last "
        "validated test estimate the same thing, three different ways. The "
        "spread between them is shown as-is — no averaging, no picking the "
        "number that flatters the day."
    )

    # ── Source 1: 20-min best from the MMP power curve × 0.95 ───────────────
    derived, pc20, pc_fetched = None, None, None
    pc_path = Path("data/power_curve.csv")
    if pc_path.exists():
        try:
            pc = pd.read_csv(pc_path)
            hit = pc[pc["duration"].astype(str).str.strip() == "20 min"]
            if not hit.empty:
                pc20 = float(hit["watts"].iloc[0])
                derived = pc20 * 0.95
                pc_fetched = datetime.fromtimestamp(pc_path.stat().st_mtime)
        except Exception:
            derived = None  # unreadable curve → card says how to refresh

    # ── Source 2: rolling eFTP as of the latest ride ────────────────────────
    eftp = pd.to_numeric(df_all.get("eftp"), errors="coerce")
    hist = (pd.DataFrame({"date": df_all["date"], "eftp": eftp})
            .dropna().sort_values("date"))
    e_now = float(hist["eftp"].iloc[-1]) if len(hist) else None
    e_date = pd.Timestamp(hist["date"].iloc[-1]) if len(hist) else None

    cc = st.columns(4)
    with cc[0]:
        metric_card(
            "20-min best × 0.95",
            f"{derived:.0f} W" if derived else "no curve",
            foot=(f"from {pc20:.1f} W best · curve fetched "
                  f"{pc_fetched:%d %b %Y}" if derived
                  else "no 20-min best in the file yet"),
            accent=C["yellow"])
    with cc[1]:
        metric_card(
            "Rolling eFTP (intervals.icu)",
            f"{e_now:.0f} W" if e_now else "—",
            foot=(f"as of {e_date:%d %b %Y} · recent-effort model"
                  if e_now else "no eFTP history"),
            accent=C["accent"])
    with cc[2]:
        metric_card(
            "Validated FTP",
            f"{FTP_CURRENT:.0f} W",
            foot="from a real test — the number you actually race on",
            accent=C["green"])

    vals = [v for v in (derived, e_now, float(FTP_CURRENT)) if v]
    with cc[3]:
        if len(vals) >= 2:
            spread = max(vals) - min(vals)
            metric_card(
                "Spread (max − min)", f"{spread:.0f} W",
                delta=f"{spread / FTP_CURRENT * 100:.0f}% of FTP",
                tone="warn" if spread > 10 else "good",
                foot="distance between the three estimates",
                accent=C["orange"])
        else:
            metric_card("Spread (max − min)", "—",
                        foot="not enough sources to compare",
                        accent=C["orange"])

    if derived and e_now:
        gap = derived - e_now
        st.caption(
            f"Reading it: **20-min best ×0.95** ({derived:.0f} W) sits "
            f"**{abs(gap):.0f} W {('above' if gap >= 0 else 'below')}** the "
            f"rolling eFTP ({e_now:.0f} W); validated FTP {FTP_CURRENT:.0f} W. "
            "The ×0.95 step is a coaching convention, not a measurement, and "
            "the power curve holds your **best of the trailing 365 days** — "
            "so a gap above the rolling model means the best is old or recent "
            "fitness dipped. Only a fresh 20-min test collapses the spread."
        )


def _render_method(df_all) -> None:
    section("🧮 How these numbers are made")
    n_eftp = int(pd.to_numeric(df_all.get("eftp"), errors="coerce").notna().sum())
    st.markdown(
        f"""
- **PMC projection** — the chart's own EWMA continued forward (α CTL = 2/43, α ATL = 2/8),
  warm-started from the newest session row's CTL/ATL; it **steps only on ride days**
  (spaced by your recent cadence — rest days and gaps hold flat), so steady-state
  CTL ≈ TSS **per ride**: the Fitness chart's session-row behaviour, replicated exactly.
  Bookkeeping, not physiology: it shows where the **load meters** point if you hit the
  planned TSS — not how you will feel or perform.
- **eFTP target** — {n_eftp} real Intervals.icu rolling-FTP observations; **no synthetic
  `IF² × duration` proxy** (that self-referential leakage is exactly what the old
  `src/` experiments suffered from).
- **Validation** — walk-forward only, embargo by target date; models compete against
  *persistence* (eFTP staying put) and skill is reported even when negative.
- **Uncertainty** — 50%/90% bands = empirical quantiles of those backtest errors, widened √time.
- **Reconciliation** — three FTP estimates side by side (20-min best ×0.95, rolling eFTP,
  validated FTP) from `data/power_curve.csv` + full history; the spread is displayed,
  never averaged away, and ×0.95 is a convention rather than a measurement.
- **Not here (yet)** — plan generation/adaptation. Next-week coaching rules stay on
  Overview, where every line is readable — this page only projects, never prescribes.
- **Filters** — the left-hand range/type filters do not move either baseline;
  forecasts always start from full history, like the fitness state badge.
""")


def render(head, ctx) -> None:
    df_all = ctx.df_all
    page_header(
        "🔮",
        "Forecast — plan the meters, project the FTP",
        "PMC projection = exact EWMA under assumed TSS · eFTP forecast = "
        "walk-forward validated against doing nothing · always full history",
    )
    _render_pmc(df_all)
    _render_eftp(df_all)
    _render_reconcile(df_all)
    _render_method(df_all)
