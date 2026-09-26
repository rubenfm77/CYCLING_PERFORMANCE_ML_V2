# views/overview.py — Overview page: hero, KPI ribbon, fatigue signals. (NEW FILE)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.components import badge, callout, legend, metric_card, section, show
from core.theme import (C, FTP_CURRENT, FTP_TARGET, H_STD, HOT_TEMP_C, STATE_GUIDE,
                        WEIGHT_KG, style_figure)


def render(head, ctx):
    date_range = ctx.date_range

    # ── Hero ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="cm-hero">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;">'
        f'<div><h1>Cycling Performance</h1>'
        f'<div class="cm-sub">{date_range} · {len(ctx.df)} sessions in view · '
        f'FTP {FTP_CURRENT}W → target {FTP_TARGET}W · {WEIGHT_KG:.0f} kg climber</div></div>'
        f'<div class="cm-right"><div style="font-size:1.05rem;font-weight:700;color:{head.state_color}">'
        f'{head.state}</div>'
        f'<div class="cm-sub">CTL {head.ctl_now:.1f} · ATL {head.atl_now:.1f} · TSB {head.tsb_now:+.1f}</div></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── KPI ribbon ──────────────────────────────────────────────────────────
    k = st.columns(6)
    with k[0]:
        metric_card("Weekly TSS", f"{head.tw_tss:,.0f}",
                    delta=f"{head.tw_tss - head.lw_tss:+,.0f} vs last week",
                    tone="good" if head.tw_tss >= head.lw_tss else "bad",
                    icon="📊")
    with k[1]:
        metric_card("CTL · Fitness", f"{head.ctl_now:.1f}",
                    delta=f"{head.ctl_now - float(head.prev['ctl']):+.1f} vs 7d ago",
                    tone="good" if head.ctl_now >= float(head.prev["ctl"]) else "bad",
                    accent=C["green"],
                    spark=head.df_all["ctl"].tail(40), spark_color=C["green"])
    with k[2]:
        metric_card("ATL · Fatigue", f"{head.atl_now:.1f}",
                    delta=f"{head.atl_now - float(head.prev['atl']):+.1f} vs 7d ago",
                    tone="bad" if head.atl_now > float(head.prev["atl"]) else "good",
                    accent=C["orange"],
                    spark=head.df_all["atl"].tail(40), spark_color=C["orange"])
    with k[3]:
        metric_card("TSB · Form", f"{head.tsb_now:+.1f}",
                    delta=f"{head.tsb_now - float(head.prev['tsb']):+.1f} vs 7d ago",
                    tone="good" if head.tsb_now >= float(head.prev["tsb"]) else "bad",
                    accent=head.state_color,
                    spark=head.df_all["tsb"].tail(40), spark_color=head.state_color)
    with k[4]:
        metric_card("eFTP (rolling)", f"{head.eftp_val:.0f}W" if head.eftp_val else "—",
                    delta=f"{head.eftp_delta:+.0f}W" if head.eftp_delta else None,
                    foot=f"manual FTP {FTP_CURRENT}W · target {FTP_TARGET}W",
                    accent=C["purple"])
    with k[5]:
        metric_card("W/kg (NP, 10-session)", f"{head.wkg_val:.2f}",
                    foot=f"at {WEIGHT_KG:.0f} kg · FTP W/kg {FTP_CURRENT/WEIGHT_KG:.2f}",
                    accent=C["teal"])

    # ── Fatigue state + next-week advice ───────────────────────────────────
    st.markdown("")
    state_icon, state_advice = STATE_GUIDE.get(head.state, ("ℹ️", ""))
    badge(f"{state_icon} {head.state}",
          f"{state_advice} — TSB {head.tsb_now:+.1f}, CTL {head.ctl_now:.1f}, ATL {head.atl_now:.1f}.",
          head.state_color,
          meta="Derived from the full PMC history (time/type filters don't change it).")

    callout(head.r_title, head.r_body, head.r_color, icon="🗓️")
    st.caption(
        f"CTL {head.ctl_now:.1f} · ATL {head.atl_now:.1f} · TSB {head.tsb_now:+.1f}"
        + (f" · W/BPM vs baseline {head.eff_delta:+.1f}%"
           if head.eff_delta and not np.isnan(head.eff_delta) else "")
    )

    # ── 1 — This week vs last week ─────────────────────────────────────────
    section("📅 This week vs last week")
    c = st.columns(5)
    with c[0]:
        metric_card("Weekly TSS", f"{head.tw_tss:,.0f}",
                    delta=f"{head.tw_tss - head.lw_tss:+,.0f}",
                    tone="good" if head.tw_tss >= head.lw_tss else "bad")
    with c[1]:
        metric_card("Sessions", f"{len(head.this_week)}",
                    delta=f"{len(head.this_week) - len(head.last_week):+d}",
                    tone="good" if len(head.this_week) >= len(head.last_week) else "bad")
    with c[2]:
        metric_card("Avg IF", f"{head.tw_if:.3f}" if head.tw_if > 0 else "—",
                    delta=f"{head.tw_if - head.lw_if:+.3f}"
                    if head.tw_if > 0 and head.lw_if > 0 else None)
    with c[3]:
        metric_card("Avg power", f"{head.tw_pwr:.0f} W" if head.tw_pwr > 0 else "—",
                    delta=f"{head.tw_pwr - head.lw_pwr:+.0f} W"
                    if head.tw_pwr > 0 and head.lw_pwr > 0 else None)
    with c[4]:
        metric_card("Elevation", f"{head.tw_elev:,.0f} m",
                    delta=f"{head.tw_elev - head.lw_elev:+,.0f} m",
                    tone="good" if head.tw_elev >= head.lw_elev else "warn",
                    foot="target ≥ 3,000 m/week")

    # ── 2 — Objective fatigue signals ──────────────────────────────────────
    section("🔬 Objective fatigue signals")
    st.caption(
        "Power/HR efficiency is more reliable than RPE for you — perception is "
        "contaminated by heat, sleep and motivation. These metrics are not."
    )
    recent_eff, curr_eff, eff_delta = head.recent_eff, head.curr_eff, head.eff_delta

    c = st.columns(3)
    with c[0]:
        color = (C["green"] if eff_delta and eff_delta > 0
                 else C["red"] if eff_delta and eff_delta < -5 else C["yellow"])
        metric_card("W/BPM vs 28d avg",
                    f"{curr_eff:.2f}" if not np.isnan(curr_eff) else "—",
                    delta=f"{eff_delta:+.1f}%" if eff_delta and not np.isnan(eff_delta) else None,
                    tone="good" if eff_delta and eff_delta > 0 else "bad" if eff_delta and eff_delta < -5 else "warn",
                    accent=color)
    with c[1]:
        n_quality = int(head.this_week["is_quality"].sum())
        metric_card("True threshold sessions",
                    f"{n_quality} / {len(head.this_week)}",
                    delta="✅ Norwegian target" if n_quality >= 2 else "⚠️ need 2 quality sessions",
                    tone="good" if n_quality >= 2 else "warn",
                    foot="IF ≥ 0.85 this week")
    with c[2]:
        z3n = int(head.this_week["is_z3_drift"].sum())
        metric_card("Z3 drift sessions", f"{z3n}",
                    delta="✅ clean" if z3n == 0 else f"⚠️ {z3n} in the grey zone",
                    tone="good" if z3n == 0 else "bad",
                    foot="avoid IF 0.75–0.85")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent_eff["date"], y=recent_eff["eff_28d"], mode="lines",
        name="28d rolling avg", line=dict(color=C["accent"], width=2, dash="dash"),
        opacity=0.7))
    fig.add_trace(go.Scatter(
        x=recent_eff["date"], y=recent_eff["eff_28d"] * 1.05, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=recent_eff["date"], y=recent_eff["eff_28d"] * 0.95, mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor="rgba(88,166,255,0.08)",
        name="±5% normal range", hoverinfo="skip"))
    dot_colors = [
        C["muted"] if (pd.isna(r["efficiency"]) or pd.isna(r["eff_28d"]))
        else C["green"] if r["efficiency"] >= r["eff_28d"] * 1.05
        else C["red"] if r["efficiency"] <= r["eff_28d"] * 0.95
        else C["yellow"]
        for _, r in recent_eff.iterrows()
    ]
    fig.add_trace(go.Scatter(
        x=recent_eff["date"], y=recent_eff["efficiency"], mode="markers",
        name="Session W/BPM", marker=dict(color=dot_colors, size=8, opacity=0.85),
        customdata=np.column_stack([
            recent_eff["efficiency"].fillna(0),
            recent_eff["eff_28d"].fillna(0),
            recent_eff["temp_avg"].fillna(-1),
        ]),
        hovertemplate="%{x|%d %b}<br>W/BPM %{customdata[0]:.2f}"
                      "<br>Baseline %{customdata[1]:.2f}"
                      "<br>Temp %{customdata[2]:.0f}°C<extra></extra>"))
    hot = recent_eff[recent_eff["is_hot_session"] == True]
    if len(hot) > 0:
        fig.add_trace(go.Scatter(
            x=hot["date"], y=hot["efficiency"], mode="markers",
            name=f"Hot session (>{HOT_TEMP_C:.0f}°C)",
            marker=dict(color=C["orange"], size=14, symbol="circle-open", line=dict(width=2)),
            hoverinfo="skip"))
    style_title = ("Cardiac efficiency (W/BPM) — objective fatigue tracker"
                   "<br><sup>🟢 above baseline = adapting · 🔴 below = fatigued · "
                   "🟠 ring = hot session > 28 °C</sup>")
    style_figure(fig, style_title, H_STD)
    legend(fig, "left", horizontal=True)
    show(fig)

    if eff_delta and not np.isnan(eff_delta):
        if eff_delta < -5:
            callout("Fatigue alert",
                    f"W/BPM is {abs(eff_delta):.1f}% below your 28-day baseline. "
                    "Reduce intensity this week regardless of how you feel subjectively.",
                    C["red"], icon="🚨")
        elif eff_delta > 5:
            callout("Adaptation signal",
                    f"W/BPM is {eff_delta:.1f}% above your 28-day baseline. "
                    "Your body is responding well — a good week to push quality.",
                    C["green"], icon="✅")
