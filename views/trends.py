# views/trends.py — Trends page: year over year, efficiency, elevation. (NEW FILE)

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.components import callout, legend, metric_card, page_header, section, show
from core.data import safe_mean, safe_sum
from core.theme import (C, FTP_CURRENT, FTP_TARGET, H_PAIR, H_STD, SURGERY, WEIGHT_KG,
                        style_figure)


def render(head, ctx):
    df, df_all = ctx.df, ctx.df_all
    date_range = ctx.date_range

    page_header(
        "📊",
        "Trends — year over year, efficiency & elevation",
        "Long-view comparison against last year, cardiac-efficiency trend and "
        "your climbing volume (target ≥ 3,000 m/week).",
    )

    # ── This year vs same period last year ────────────────────────────────
    section("📊 This year vs same period last year")
    current_year, current_month = datetime.now().year, datetime.now().month
    today_md = datetime.now().strftime("%m-%d")
    this_yr = df_all[df_all["year"] == current_year]
    last_yr = df_all[(df_all["year"] == current_year - 1) &
                     (df_all["date"].dt.strftime("%m-%d") <= today_md)]

    def yr_stats(d):
        return {
            "sessions": len(d),
            "total_tss": safe_sum(d["tss"]),
            "avg_power": safe_mean(d["power_avg"]),
            "avg_wkg": safe_mean(d["w_per_kg"]),
            "elevation": safe_sum(d["elevation"]),
            "hours": safe_sum(d["duration_h"]),
        }

    ty, ly = yr_stats(this_yr), yr_stats(last_yr)
    yc = st.columns(6)
    specs = [("Sessions", ty["sessions"], ly["sessions"], "", 0),
             ("Total TSS", ty["total_tss"], ly["total_tss"], "", 0),
             ("Avg power", ty["avg_power"], ly["avg_power"], " W", 1),
             ("Avg W/kg", ty["avg_wkg"], ly["avg_wkg"], "", 2),
             ("Elevation", ty["elevation"], ly["elevation"], " m", 0),
             ("Hours", ty["hours"], ly["hours"], " h", 1)]
    for col, (label, tv, lv, unit, dec) in zip(yc, specs):
        # whole numbers get thousands separators — a season's Total TSS or
        # Elevation is six figures, and "135000" is unreadable where
        # "135,000" is instant
        fmt = (f"{{:,.0f}}{unit}" if dec == 0 else f"{{:.{dec}f}}{unit}")
        d_fmt = (f"{{:+,.0f}}{unit}" if dec == 0 else f"{{:+.{dec}f}}{unit}")
        with col:
            delta_tone = "good" if (lv and tv >= lv) else "bad" if lv else None
            metric_card(f"{label} {current_year}",
                        fmt.format(tv) if tv and not np.isnan(float(tv)) else "—",
                        delta=d_fmt.format(tv - lv) + f" vs {current_year - 1}"
                        if lv and not np.isnan(float(lv)) else None,
                        tone=delta_tone,
                        foot=f"same period {current_year - 1}: {fmt.format(lv) if lv else '—'}")

    cc = st.columns(2)
    with cc[0]:
        mc = [{"month": pd.Timestamp(f"{current_year}-{m:02d}-01").strftime("%b"),
               "this_year": (safe_sum(df_all[(df_all["year"] == current_year) &
                                             (df_all["date"].dt.month == m)]["tss"])
                             if m <= current_month else np.nan),
               "last_year": safe_sum(df_all[(df_all["year"] == current_year - 1) &
                                            (df_all["date"].dt.month == m)]["tss"])}
              for m in range(1, 13)]
        mc_df = pd.DataFrame(mc)
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(x=mc_df["month"], y=mc_df["last_year"],
                                 name=str(current_year - 1), marker_color=C["muted"], opacity=0.6))
        fig_cmp.add_trace(go.Bar(x=mc_df["month"], y=mc_df["this_year"],
                                 name=str(current_year), marker_color=C["accent"], opacity=0.85))
        fig_cmp.update_layout(barmode="group")
        style_figure(fig_cmp, f"Monthly TSS — {current_year} vs {current_year - 1}", H_PAIR)
        legend(fig_cmp, "left", horizontal=True)
        show(fig_cmp)
    with cc[1]:
        wc = [{"month": pd.Timestamp(f"{current_year}-{m:02d}-01").strftime("%b"),
               "this_year": (safe_mean(df_all[(df_all["year"] == current_year) &
                                              (df_all["date"].dt.month == m)]["w_per_kg"])
                             if m <= current_month else np.nan),
               "last_year": safe_mean(df_all[(df_all["year"] == current_year - 1) &
                                             (df_all["date"].dt.month == m)]["w_per_kg"])}
              for m in range(1, 13)]
        wc_df = pd.DataFrame(wc)
        fig_wcmp = go.Figure()
        fig_wcmp.add_trace(go.Scatter(x=wc_df["month"], y=wc_df["last_year"],
                                      name=str(current_year - 1),
                                      line=dict(color=C["muted"], width=2, dash="dash"),
                                      mode="lines+markers"))
        fig_wcmp.add_trace(go.Scatter(x=wc_df["month"], y=wc_df["this_year"],
                                      name=str(current_year),
                                      line=dict(color=C["purple"], width=2.5),
                                      mode="lines+markers"))
        style_figure(fig_wcmp, f"Monthly W/kg — {current_year} vs {current_year - 1}", H_PAIR)
        legend(fig_wcmp, "left", horizontal=True)
        show(fig_wcmp)

    # ── Power & efficiency trend ───────────────────────────────────────────
    section("⚡ Power & efficiency trend")
    cc = st.columns(2)
    with cc[0]:
        monthly = df.resample("ME", on="date").agg(
            avg_wkg=("w_per_kg", "mean"), max_wkg=("w_per_kg", "max")).reset_index()
        fig_mw = go.Figure()
        fig_mw.add_trace(go.Scatter(x=monthly["date"], y=monthly["avg_wkg"],
                                    mode="lines+markers", name="Avg W/kg",
                                    line=dict(color=C["purple"], width=2.5),
                                    marker=dict(size=6)))
        fig_mw.add_trace(go.Scatter(x=monthly["date"], y=monthly["max_wkg"],
                                    mode="lines", name="Max W/kg",
                                    line=dict(color=C["accent"], width=1.5, dash="dash"),
                                    opacity=0.6))
        fig_mw.add_hline(y=round(FTP_CURRENT / WEIGHT_KG, 2), line_dash="dot",
                         line_color=C["yellow"],
                         annotation_text=f"Current FTP target ({FTP_CURRENT / WEIGHT_KG:.2f} W/kg)")
        fig_mw.add_hline(y=round(FTP_TARGET / WEIGHT_KG, 2), line_dash="dot",
                         line_color=C["purple"], opacity=0.4,
                         annotation_text=f"Pre-injury target ({FTP_TARGET / WEIGHT_KG:.2f} W/kg)")
        style_figure(fig_mw, "Monthly W/kg trend", H_PAIR)
        legend(fig_mw, "left", horizontal=True)
        show(fig_mw)
    with cc[1]:
        monthly_eff = df.resample("ME", on="date").agg(
            avg_eff=("efficiency", "mean")).reset_index().dropna()
        fig_me = go.Figure(go.Scatter(
            x=monthly_eff["date"], y=monthly_eff["avg_eff"], mode="lines+markers",
            name="W/BPM", line=dict(color=C["orange"], width=2.5), marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(240,136,62,0.1)"))
        style_figure(fig_me, "Cardiac efficiency (W per BPM) — higher is better",
                     H_PAIR, showlegend=False)
        show(fig_me)

    # ── Elevation (climber stats) ──────────────────────────────────────────
    section("🏔️ Elevation — climber stats")
    st.caption("57 kg climber — elevation per week is your key volume metric.")
    total_elev = float(df_all["elevation"].sum())
    span_weeks = max((df_all["date"].max() - df_all["date"].min()).days / 7, 1)
    ec = st.columns(3)
    with ec[0]:
        metric_card("Career elevation", f"{total_elev:,.0f} m", accent=C["green"])
    with ec[1]:
        metric_card("Avg elevation / week", f"{total_elev / span_weeks:,.0f} m",
                    foot="target ≥ 3,000 m", accent=C["accent"])
    with ec[2]:
        metric_card("Best single session", f"{float(df_all['elevation'].max()):,.0f} m",
                    accent=C["yellow"])
    cc = st.columns(2)
    with cc[0]:
        weekly_elev = df.resample("W", on="date").agg(
            elevation=("elevation", "sum")).reset_index()
        weekly_elev["elevation"] = weekly_elev["elevation"].fillna(0)
        fig_el = go.Figure(go.Bar(
            x=weekly_elev["date"], y=weekly_elev["elevation"],
            marker_color=[C["green"] if e >= 3000 else C["yellow"] if e >= 1500 else C["muted"]
                          for e in weekly_elev["elevation"]],
            opacity=0.85, name="Weekly elevation (m)"))
        fig_el.add_hline(y=3000, line_dash="dot", line_color=C["green"],
                         annotation_text="3,000 m/week target")
        fig_el.add_hline(y=1500, line_dash="dot", line_color=C["yellow"],
                         annotation_text="1,500 m minimum", opacity=0.5)
        style_figure(fig_el, f"Weekly elevation — {date_range}", H_PAIR, showlegend=False)
        show(fig_el)
    with cc[1]:
        annual_elev = df_all.groupby("year").agg(total_elev=("elevation", "sum")).reset_index()
        fig_ae = go.Figure(go.Bar(
            x=annual_elev["year"].astype(str), y=annual_elev["total_elev"],
            marker_color=[C["red"] if y == 2025 else C["accent"] for y in annual_elev["year"]],
            opacity=0.85, text=[f"{v / 1000:.1f} K" for v in annual_elev["total_elev"]],
            textposition="outside"))
        style_figure(fig_ae, "Total elevation per year (red = surgery year)",
                     H_PAIR, showlegend=False)
        show(fig_ae)
