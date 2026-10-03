# views/fitness.py — Fitness page: PMC, VO2max, power curve, FTP. (NEW FILE)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from plotly.subplots import make_subplots

from core.components import callout, legend, metric_card, page_header, section, show
from core.theme import (C, FTP_CURRENT, FTP_TARGET, H_CARD, H_HERO, H_PAIR, H_STD,
                        STATE_COLORS, SURGERY, WEIGHT_KG, style_figure)
from ml import interval_watts as iw
from views.interval_watts import fitness_section


def render(head, ctx):
    df, df_all = ctx.df, ctx.df_all

    page_header(
        "📈",
        "Fitness — fitness vs fatigue, VO2max & power",
        f"PMC follows the selected range ({ctx.date_range}) · VO2max and FTP "
        f"development use full history · FTP {FTP_CURRENT}W → target {FTP_TARGET}W",
    )

    # ── Performance management chart ───────────────────────────────────────
    section("📈 Performance management chart")
    fig_pmc = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.35, 0.25],
        subplot_titles=["CTL vs ATL — fitness vs fatigue",
                        "TSB — form / freshness",
                        "Weekly TSS"])
    fig_pmc.add_trace(go.Scatter(x=df["date"], y=df["ctl"], name="CTL fitness",
                                 line=dict(color=C["green"], width=2.5)), row=1, col=1)
    fig_pmc.add_trace(go.Scatter(x=df["date"], y=df["atl"], name="ATL fatigue",
                                 line=dict(color=C["orange"], width=2.5),
                                 fill="tonexty", fillcolor="rgba(240,136,62,0.1)"), row=1, col=1)
    for sname, scolor in STATE_COLORS.items():
        mask = df["fatigue_state"] == sname
        if mask.sum() > 0:
            fig_pmc.add_trace(go.Scatter(
                x=df.loc[mask, "date"], y=df.loc[mask, "tsb"], mode="markers",
                name=sname, showlegend=False,  # colours stay, hover keeps the label
                marker=dict(color=scolor, size=5, opacity=0.7)), row=2, col=1)
    fig_pmc.add_hline(y=-30, line_dash="dash", line_color=C["red"],
                      annotation_text="−30 overreach", row=2, col=1)
    fig_pmc.add_hline(y=25, line_dash="dash", line_color=C["purple"],
                      annotation_text="+25 peak", row=2, col=1)
    fig_pmc.add_hline(y=0, line_dash="dot", line_color=C["muted"], row=2, col=1)
    weekly_tss = df.resample("W", on="date")["tss"].sum().reset_index()
    fig_pmc.add_trace(go.Bar(
        x=weekly_tss["date"], y=weekly_tss["tss"],
        marker_color=[C["red"] if t >= 700 else C["green"] if t >= 400 else C["yellow"]
                      for t in weekly_tss["tss"]],
        name="Weekly TSS", opacity=0.75), row=3, col=1)
    style_figure(fig_pmc, None, H_HERO, showlegend=True)
    fig_pmc.update_annotations(font=dict(color=C["text"], size=12))
    for i in range(1, 4):
        fig_pmc.update_xaxes(gridcolor=C["grid"], row=i, col=1)
        fig_pmc.update_yaxes(gridcolor=C["grid"], row=i, col=1)
    legend(fig_pmc, "right")
    show(fig_pmc)

    # ── VO2max estimate ────────────────────────────────────────────────────
    section("🩺 VO2max estimate · power-curve based")
    st.caption(
        "ACSM leg-cycling equation — VO2max = 10.8 × W/kg + 7 — using each "
        "session's Critical Power (icu_pm_cp) and logged body weight. CP is a "
        "sustainable effort, not the short maximal ramp-test power the formula "
        "was built around, so this likely undershoots true VO2max — treat it as "
        "a directional trend, not an absolute number. All-time, ignores filters."
    )
    vo2_data = df_all[df_all["vo2max_est"].notna()].sort_values("date")
    if len(vo2_data) > 0:
        vc = st.columns([1, 3])
        with vc[0]:
            latest_vo2 = float(vo2_data.iloc[-1]["vo2max_est"])
            first_vo2 = float(vo2_data.iloc[0]["vo2max_est"])
            metric_card("Latest est. VO2max", f"{latest_vo2:.1f}",
                        delta=f"{latest_vo2 - first_vo2:+.1f} vs window start",
                        tone="good" if latest_vo2 >= first_vo2 else "bad",
                        foot=f"ml/kg/min · {len(vo2_data)} modelled sessions",
                        accent=C["green"])
        with vc[1]:
            fig_vo2 = go.Figure(go.Scatter(
                x=vo2_data["date"], y=vo2_data["vo2max_est"],
                mode="lines+markers", name="Est. VO2max",
                line=dict(color=C["green"], width=2.5), marker=dict(size=6),
                fill="tozeroy", fillcolor="rgba(63,185,80,0.1)"))
            style_figure(fig_vo2, "Estimated VO2max over time (ml/kg/min)",
                         H_CARD, showlegend=False)
            show(fig_vo2)
    else:
        callout("No power-curve model data",
                "No icu_pm_cp values yet — this populates once Intervals.icu fits "
                "a critical-power model to recent activities.", C["accent"], icon="ℹ️")

    # ── Power curve & best efforts ─────────────────────────────────────────
    section("⚡ Power curve & best efforts")
    # Source order: the mean-maximal CSV if some pipeline actually produced it,
    # otherwise the curve computed from the file itself. The CSV was never
    # reachable on a deploy (nothing in this repo writes it), so the old code
    # showed "run python src/intervals_api.py" — a script that does not exist.
    pc_df = pd.DataFrame()
    pc_src = None
    if Path("data/power_curve.csv").exists():
        try:
            _csv = pd.read_csv("data/power_curve.csv")
            if {"watts", "secs", "duration"}.issubset(_csv.columns):
                pc_df, pc_src = _csv, "csv"
        except Exception:
            pc_df = pd.DataFrame()

    if pc_df.empty or "watts" not in pc_df.columns:
        pc_df = iw.power_curve(df_all)
        pc_src = "file" if len(pc_df) else None

    if pc_src is None or pc_df.empty:
        callout("Power curve unavailable",
                "No detected effort in your file carries both a duration and a "
                "watt figure, so there is nothing to plot. This curve is built "
                "from the efforts already in your training data — no separate "
                "download is needed.",
                C["accent"], icon="ℹ️")
    else:
        pc_df = pc_df.sort_values("secs").reset_index(drop=True)
        if "date" in pc_df.columns:
            pc_df["date"] = pd.to_datetime(pc_df["date"], errors="coerce")
        pc_df["wkg"] = pc_df["watts"] / WEIGHT_KG   # one weight value, from theme
        _hover_ok = {"date", "n", "duration"}.issubset(pc_df.columns)
        cc = st.columns(2)
        with cc[0]:
            fig_pc = go.Figure(go.Scatter(
                x=pc_df["secs"], y=pc_df["watts"], mode="lines+markers+text",
                line=dict(color=C["purple"], width=3),
                marker=dict(size=10, color=C["purple"]),
                text=[""] + [f"{w:.0f} W" for w in pc_df["watts"].iloc[1:]],
                textposition=["top center" if i % 2 == 0 else "bottom center"
                              for i in range(len(pc_df))],
                name="Best power (W)",
                customdata=(pc_df[["duration", "n", "date"]].astype(object).values
                            if _hover_ok else None),
                hovertemplate=(
                    "duration <b>%{customdata[0]}</b><br>"
                    "%{customdata[1]} effort(s) in the window<br>"
                    "best on %{customdata[2]|%d %b %Y}<extra>power curve</extra>"
                ) if _hover_ok else None))
            fig_pc.add_hline(y=FTP_CURRENT, line_dash="dot", line_color=C["yellow"])
            fig_pc.add_hline(y=FTP_TARGET, line_dash="dot", line_color=C["green"], opacity=0.4)
            fig_pc.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                        line=dict(color=C["yellow"], width=2, dash="dot"),
                                        name=f"Current FTP ({FTP_CURRENT} W)"))
            fig_pc.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                        line=dict(color=C["green"], width=2, dash="dot"),
                                        name=f"Target FTP ({FTP_TARGET} W)"))
            fig_pc.update_xaxes(type="log", tickvals=pc_df["secs"].tolist(),
                                ticktext=pc_df["duration"].tolist())
            style_figure(fig_pc, "Power curve — best mean-maximal power", H_PAIR)
            legend(fig_pc, "right")
            show(fig_pc)
        with cc[1]:
            ftp_wkg = round(FTP_CURRENT / WEIGHT_KG, 2)
            tgt_wkg = round(FTP_TARGET / WEIGHT_KG, 2)
            fig_wkg = go.Figure(go.Bar(
                x=pc_df["duration"], y=pc_df["wkg"],
                marker_color=[C["purple"] if w >= 8.0 else C["red"] if w >= 5.0
                              else C["orange"] if w >= 4.0 else C["yellow"] if w >= 3.5
                              else C["accent"] for w in pc_df["wkg"]],
                text=[f"{w:.2f}" for w in pc_df["wkg"]],
                textposition="outside", opacity=0.85, showlegend=False))
            fig_wkg.add_hline(y=ftp_wkg, line_dash="dot", line_color=C["yellow"])
            fig_wkg.add_hline(y=tgt_wkg, line_dash="dot", line_color=C["green"], opacity=0.4)
            fig_wkg.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                         line=dict(color=C["yellow"], width=2, dash="dot"),
                                         name=f"Current FTP ({ftp_wkg} W/kg)"))
            fig_wkg.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                         line=dict(color=C["green"], width=2, dash="dot"),
                                         name=f"Target ({tgt_wkg} W/kg)"))
            fig_wkg.update_yaxes(range=[0, round(float(pc_df["wkg"].max()) * 1.15, 1)])
            style_figure(fig_wkg, f"W/kg at each duration — {WEIGHT_KG:.0f} kg climber", H_PAIR)
            legend(fig_wkg, "right")
            show(fig_wkg)

        if pc_src == "file":
            # Disclose exactly where every point came from — no silent sourcing.
            M_pc = iw.measured(df_all)
            dcol = pd.to_datetime(pc_df["date"], errors="coerce").dropna()
            span = (f"{dcol.min():%d %b %Y} → {dcol.max():%d %b %Y}"
                    if len(dcol) else "—")
            st.caption(
                f"**Source: your own training file.** {len(pc_df)} durations, each "
                f"the best effort within ±12% of that length, chosen from "
                f"{len(M_pc):,} detected efforts ({span}); the date and the number "
                f"of efforts behind every point are on the hover. Nothing is "
                f"interpolated between points and nothing is extrapolated past the "
                f"longest effort you have ridden — a duration you never held inside "
                f"the window is simply not drawn."
            )
        elif pc_src == "csv":
            st.caption(
                "**Source: `data/power_curve.csv`** — a mean-maximal export "
                "already sitting in your data folder. Every point is read from "
                "that file as it is; nothing on this chart is recomputed."
            )

    # ── Interval watts over time + power-duration law ─────────────────────
    # Full history (df_all), not the sidebar range: the sidebar defaults to
    # six months and would delete every year compared here.
    fitness_section(df_all)

    # ── FTP progression ────────────────────────────────────────────────────
    section("📈 FTP progression — month by month")
    fc = st.columns(2)
    with fc[0]:
        monthly_ftp = (
            df_all[df_all["power_np"].fillna(0) > 0]
            .groupby(df_all["date"].dt.to_period("M"))
            .agg(best_np=("power_np", "max"), sessions=("tss", "count"))
            .reset_index()
        )
        monthly_ftp["month_dt"] = monthly_ftp["date"].apply(lambda p: p.start_time)
        monthly_ftp["ftp_est"] = monthly_ftp["best_np"] * 0.95
        fig_fp = go.Figure()
        fig_fp.add_trace(go.Scatter(
            x=monthly_ftp["month_dt"], y=monthly_ftp["ftp_est"],
            mode="lines+markers", name="Est. FTP (best NP × 0.95)",
            line=dict(color=C["purple"], width=2.5), marker=dict(size=5),
            fill="tozeroy", fillcolor="rgba(188,140,255,0.08)"))
        fig_fp.add_trace(go.Scatter(
            x=monthly_ftp["month_dt"], y=monthly_ftp["ftp_est"].rolling(3).mean(),
            mode="lines", name="3-month trend",
            line=dict(color=C["accent"], width=2, dash="dash")))
        fig_fp.add_vline(x=pd.Timestamp(SURGERY), line_dash="dash",
                         line_color=C["red"], opacity=0.7)
        # One FTP value only — theme says 240 (app.py drew 235 here, 240 elsewhere)
        fig_fp.add_hline(y=FTP_CURRENT, line_dash="dot", line_color=C["yellow"])
        fig_fp.add_hline(y=FTP_TARGET, line_dash="dot", line_color=C["green"], opacity=0.4)
        style_figure(fig_fp, "Monthly FTP progression 2019–2026", H_STD)
        legend(fig_fp, "left", horizontal=True)
        show(fig_fp)
        st.caption(f"Yellow ── current FTP {FTP_CURRENT} W · green ── pre-injury "
                   f"{FTP_TARGET} W · red │ surgery {pd.Timestamp(SURGERY).strftime('%b %Y')}")
    with fc[1]:
        annual_ftp = (
            df_all[df_all["power_np"].fillna(0) > 0]
            .groupby("year")
            .agg(peak_ftp=("power_np", lambda x: x.max() * 0.95),
                 avg_wkg=("w_per_kg", "mean"))
            .reset_index()
        )
        fig_af = go.Figure(go.Bar(
            x=annual_ftp["year"].astype(str), y=annual_ftp["peak_ftp"],
            marker_color=[C["red"] if y == 2025 else C["purple"] for y in annual_ftp["year"]],
            opacity=0.85, text=[f"{v:.0f} W" for v in annual_ftp["peak_ftp"]],
            textposition="outside", name="Peak FTP estimate"))
        style_figure(fig_af, "Peak estimated FTP by year (red = surgery year)",
                     H_STD, showlegend=False)
        show(fig_af)
