# views/sessions.py — Sessions page: log, wellness & methodology. (NEW FILE)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.components import callout, page_header, section, show
from core.theme import (C, FTP_CURRENT, FTP_TARGET, H_CARD, HOT_TEMP_C, IF_THRESHOLD,
                        IF_VO2, IF_Z2_MAX, WEIGHT_KG, style_figure)


def render(head, ctx):
    df_all = ctx.df_all

    page_header(
        "📅",
        "Sessions — log, wellness & methodology",
        "The full 20-session audit table (ignores filters), recovery signals "
        "and exactly how every number on this dashboard is computed.",
    )

    # ── Recent sessions ────────────────────────────────────────────────────
    section("📅 Recent sessions")
    display_cols = [c for c in ["date", "training_type", "tss", "if_score", "power_avg",
                                "duration_h", "hr_avg", "w_per_kg", "elevation", "temp_avg",
                                "quality_score", "fatigue_state"] if c in df_all.columns]
    recent = df_all[display_cols].tail(20).sort_values("date", ascending=False).copy()
    recent["date"] = recent["date"].dt.strftime("%d %b %Y")
    for col in ["tss", "if_score", "power_avg", "duration_h", "hr_avg",
                "w_per_kg", "elevation", "temp_avg", "quality_score"]:
        if col in recent.columns:
            recent[col] = pd.to_numeric(recent[col], errors="coerce").round(2)
    if "training_type" in recent.columns:
        recent["training_type"] = recent["training_type"].fillna("—")
    recent = recent.rename(columns={
        "date": "Date", "training_type": "Type", "tss": "TSS", "if_score": "IF",
        "power_avg": "Power (W)", "duration_h": "Hours", "hr_avg": "Avg HR",
        "w_per_kg": "W/kg", "elevation": "Elev (m)", "temp_avg": "Temp °C",
        "quality_score": "Quality", "fatigue_state": "State"})
    st.dataframe(
        recent, width="stretch", height=420,
        column_config={
            "TSS": st.column_config.ProgressColumn("TSS", min_value=0, max_value=300, format="%d"),
            "IF": st.column_config.NumberColumn("IF", format="%.3f"),
            "W/kg": st.column_config.NumberColumn("W/kg", format="%.2f"),
            # "localized" prints 1,250 rather than 1250 — elevation is the one
            # column here big enough for the separator to earn its place
            "Elev (m)": st.column_config.NumberColumn("Elev (m)", format="localized", help="metres climbed"),
            "Quality": st.column_config.ProgressColumn("Quality", min_value=0, max_value=100, format="%.0f"),
        })
    st.caption("All sessions, ignoring the range filter — so you can always audit what happened.")

    # ── Wellness & HRV ─────────────────────────────────────────────────────
    section("💚 Wellness & recovery")
    st.caption("HRV and resting HR are the earliest objective signs of accumulated "
               "fatigue — they respond before CTL/ATL do.")
    wellness = ctx.wellness
    has_hrv = (not wellness.empty and "hrv" in wellness.columns
               and wellness["hrv"].notna().sum() > 5)
    has_rhr = (not wellness.empty and "resting_hr" in wellness.columns
               and wellness["resting_hr"].notna().sum() > 5
               and float(wellness["resting_hr"].std() or 0) > 0)
    wellness_recent = (wellness[wellness["date"] >= pd.Timestamp(ctx.cutoff)]
                       if not wellness.empty else wellness)
    wc2 = st.columns(2)
    with wc2[0]:
        if has_hrv:
            fig_h = go.Figure(go.Scatter(
                x=wellness_recent["date"], y=wellness_recent["hrv"],
                mode="lines+markers", line=dict(color=C["green"], width=2),
                fill="tozeroy", fillcolor="rgba(63,185,80,0.1)"))
            style_figure(fig_h, "HRV — higher = more recovered", H_CARD, showlegend=False)
            show(fig_h)
        else:
            callout("HRV not available",
                    "HRV has never been logged in your Intervals.icu wellness data "
                    "(0 days present), so no HRV chart can be drawn — this is a data "
                    "gap, not a bug. Log HRV daily in Intervals.icu and it will "
                    "appear here automatically.",
                    C["accent"], icon="ℹ️")
    with wc2[1]:
        if has_rhr:
            fig_r = go.Figure(go.Scatter(
                x=wellness_recent["date"], y=wellness_recent["resting_hr"],
                mode="lines+markers", line=dict(color=C["orange"], width=2),
                fill="tozeroy", fillcolor="rgba(240,136,62,0.08)"))
            style_figure(fig_r, "Resting HR — lower = more recovered", H_CARD, showlegend=False)
            show(fig_r)
        else:
            callout("Resting HR not available",
                    "No resting-HR values in wellness data for this range.",
                    C["muted"], icon="ℹ️")

    # ── Methodology & caveats ──────────────────────────────────────────────
    section("Methodology & caveats")
    with st.expander("How every number on this dashboard is computed", expanded=False):
        st.markdown(f"""
**Metrics**
- **CTL / ATL / TSB** — exponential weighted means of TSS with 42 / 7-day spans (`ewm(span, adjust=False)`), TSB = CTL − ATL.
- **TSS / IF** — from Intervals.icu. IF values > 2 are treated as percentages and divided by 100.
- **W/kg** — normalized power ÷ weight; falls back to average power where NP is missing. **One weight value everywhere: {WEIGHT_KG} kg** (app.py mixed 57.0 / 56.8).
- **Efficiency (W/BPM)** — NP ÷ average HR; 28-day rolling mean is the personal baseline. Hot rides (> {HOT_TEMP_C:.0f} °C) are ringed on the chart because heat, not fitness, moves this number.
- **Quality score** — 0–100 blend: efficiency vs 28d baseline (50%), TSS (30%), IF (20%).
- **VO2max** — ACSM leg-cycling: `10.8 × W/kg + 7` using Critical Power (icu_pm_cp), not ramp-test power → likely undershoots; trend only.
- **FTP** — manual value **{FTP_CURRENT} W** (validated June 2026, matches athlete-profile.md). app.py's FTP charts disagreed (235 vs 240); this dashboard uses one value. Target {FTP_TARGET} W (pre-injury).
- **Est. FTP / FTP proxy** — best monthly NP × 0.95; a proxy, not a test result.

**Norwegian method flags**
- Pure Z2: IF < {IF_Z2_MAX} · Z3 drift (avoid): {IF_Z2_MAX} ≤ IF < {IF_THRESHOLD} · True threshold: IF ≥ {IF_THRESHOLD} · VO2max: IF ≥ {IF_VO2}.

**Known data gaps**
- HRV has never been logged (0 days) — the HRV panel says so honestly instead of rendering an empty chart.
- Only **{head.all_time_q}** threshold and **{head.all_time_vo2}** VO2max sessions exist all-time — the "2 quality/week" target is aspirational, not observed.
- ~72 monthly observations for composition correlation → hypothesis-generating only.
- Elevation/duration/temperature come from the Strava↔Intervals.icu join; API-only days may lack elevation.

**Fixes vs app.py (original untouched)**
- Training-type filter in the sidebar was **dead code** in app.py — here it is wired to every range-scoped chart.
- FTP 235/240 and weight 56.8/57.0 mismatches reconciled to single values in `core/theme.py`.
- Empty states are explicit (HRV, power curve) instead of silently blank charts.
        """)
    st.caption("*Everything here is read from your synced training file · "
               "click 🔄 Refresh data to clear the cache and re-sync it*")
