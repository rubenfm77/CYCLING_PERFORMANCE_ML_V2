# views/training.py — Training page: method compliance & composition. (NEW FILE)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.components import (callout, legend, metric_card, page_header, section, show,
                             stat_block)
from core.theme import (C, FTP_CURRENT, FTP_TARGET, FTP_DRIVERS, H_CARD, H_PAIR, H_STD,
                        IF_THRESHOLD, IF_VO2, IF_Z2_MAX, MAIN_TYPES, SURGERY, TYPE_COLORS,
                        ZONES, style_figure)


def render(head, ctx):
    df, df_main, df_main_all, df_all = ctx.df, ctx.df_main, ctx.df_main_all, ctx.df_all
    date_range = ctx.date_range

    page_header(
        "🏋️",
        "Training — method compliance & composition",
        "Is the weekly load actually Norwegian-compliant, and does the "
        "quality/base mix predict FTP? Thresholds: IF 0.75 / 0.85 / 0.95.",
    )

    # ── Norwegian method compliance ────────────────────────────────────────
    section("🇳🇴 Norwegian method compliance")
    st.caption(
        "Double threshold: 2 sessions/week at true threshold (IF ≥ 0.85). "
        "Avoid Z3 drift (IF 0.75–0.85). Keep Z2 pure (IF < 0.75). "
        "No VO2max unless specifically planned."
    )

    # Distribution-first: the real numbers behind the "2/week target"
    n_total = len(df)
    n_q = int(df["is_quality"].sum())
    n_z3 = int(df["is_z3_drift"].sum())
    n_z2 = int(df["is_true_z2"].sum())
    pct = lambda n: f"{n / n_total * 100:.1f}%" if n_total else "—"
    mc = st.columns(3)
    with mc[0]:
        stat_block(f"{n_z2}", "Pure Z2 sessions (IF < 0.75)", pct(n_z2), C["green"])
    with mc[1]:
        stat_block(f"{n_z3}", "Z3 drift — grey zone (0.75–0.85)", pct(n_z3), C["orange"])
    with mc[2]:
        stat_block(f"{n_q}", "True threshold (IF ≥ 0.85)", pct(n_q), C["red"])

    st.caption(
        f"All-time: **{head.all_time_q}** threshold sessions and **{head.all_time_vo2}** "
        "VO2max sessions exist across the full history — the “2 quality "
        "sessions per week” Norwegian target has rarely if ever been met in "
        "sustained blocks. The numbers above show where you actually sit."
    )

    # Weekly quality distribution + IF histogram
    cc = st.columns(2)
    with cc[0]:
        weekly_compliance = (
            df.groupby(pd.Grouper(key="date", freq="W"))
            .apply(lambda x: pd.Series({
                "quality_sessions": int(x["if_score"].ge(IF_THRESHOLD).sum()),
                "z3_drift": int((x["if_score"].ge(IF_Z2_MAX) & x["if_score"].lt(IF_THRESHOLD)).sum()),
                "pure_z2": int(x["if_score"].lt(IF_Z2_MAX).sum()),
                "total": len(x),
            }), include_groups=False)
            .reset_index()
        )
        fig_n = go.Figure()
        fig_n.add_trace(go.Bar(
            x=weekly_compliance["date"], y=weekly_compliance["quality_sessions"],
            name="True threshold (IF ≥ 0.85)", marker_color=C["green"], opacity=0.85))
        fig_n.add_trace(go.Bar(
            x=weekly_compliance["date"], y=weekly_compliance["z3_drift"],
            name="Z3 drift — avoid", marker_color=C["orange"], opacity=0.85))
        fig_n.add_hline(y=2, line_dash="dot", line_color=C["green"],
                        annotation_text="Norwegian target (2)")
        fig_n.update_layout(barmode="stack")
        style_figure(fig_n, "Weekly session quality distribution"
                     "<br><sup>green line = Norwegian target of 2 threshold "
                     "sessions per week</sup>", H_PAIR)
        legend(fig_n, "left", horizontal=True)
        show(fig_n)
    with cc[1]:
        if_data = df[df["if_score"].notna() & (df["if_score"] > 0.3)]
        fig_if = go.Figure(go.Histogram(
            x=if_data["if_score"], nbinsx=30, marker_color=C["accent"],
            opacity=0.75, name="Sessions", showlegend=False))
        for x, col, label in [(IF_Z2_MAX, C["yellow"], f"{IF_Z2_MAX} — Z2/Z3 boundary"),
                              (IF_THRESHOLD, C["green"], f"{IF_THRESHOLD} — threshold start"),
                              (IF_VO2, C["red"], f"{IF_VO2} — VO2max zone")]:
            fig_if.add_vline(x=x, line_dash="solid", line_color=col, opacity=0.8)
            fig_if.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                        line=dict(color=col, width=2), name=label))
        style_figure(fig_if, f"IF distribution — {date_range}"
                     "<br><sup>ideal: big spike at &lt;0.75 (Z2) + smaller spike "
                     "at 0.85–0.95 (threshold)</sup>", H_PAIR)
        legend(fig_if, "right")
        show(fig_if)

    # ── FTP stimulus & quality/base TSS split ──────────────────────────────
    section("🎯 FTP development — is the mix actually driving FTP?")
    st.caption("Is your training mix actually pushing FTP toward 275 W?")
    cc = st.columns(2)
    with cc[0]:
        if len(df_main) > 0:
            stim = (df_main.groupby("training_type")["ftp_stimulus"]
                    .mean().reset_index().sort_values("ftp_stimulus", ascending=True))
            stim["is_driver"] = stim["training_type"].isin(FTP_DRIVERS)
            fig_s = go.Figure(go.Bar(
                x=stim["ftp_stimulus"], y=stim["training_type"], orientation="h",
                marker_color=[C["green"] if d else C["muted"] for d in stim["is_driver"]],
                opacity=0.85, text=[f"{v:.1f}" for v in stim["ftp_stimulus"]],
                textposition="outside"))
            style_figure(fig_s, f"FTP stimulus score — {date_range}",
                         H_PAIR, showlegend=False)
            show(fig_s)
        else:
            callout("No labelled training types",
                    "No sessions with a MAIN_TYPES label in this period.",
                    C["muted"], icon="ℹ️")
    with cc[1]:
        if len(df_main) > 0:
            driver_tss = float(df_main.loc[df_main["training_type"].isin(FTP_DRIVERS), "tss"].sum())
            base_tss = float(df_main.loc[~df_main["training_type"].isin(FTP_DRIVERS), "tss"].sum())
            fig_pie = go.Figure(go.Pie(
                labels=["FTP drivers", "Base volume"], values=[driver_tss, base_tss],
                marker_colors=[C["green"], C["muted"]], hole=0.5,
                textinfo="label+percent"))
            style_figure(fig_pie, "TSS split — quality vs base volume",
                         H_PAIR, showlegend=False)
            show(fig_pie)
        else:
            callout("No labelled training types",
                    "No sessions with a MAIN_TYPES label in this period.",
                    C["muted"], icon="ℹ️")

    # ── Training composition analysis ──────────────────────────────────────
    section("🔀 Training composition — does the mix predict FTP?")
    st.caption(
        "Each bar is % TSS by training type per month. Months are independent "
        "observations but the *count* is small, so treat anything below as "
        "hypothesis-generating, not confirmatory. This section always uses your "
        "full history, ignoring the sidebar range."
    )

    # Composition analysis deliberately uses the FULL history (df_all) rather than
    # the sidebar window: "does the mix predict FTP?" is a long-term question, so it
    # gets a long-term answer regardless of the date-range picker.
    _comp_all = df_all[df_all["training_type"].isin(MAIN_TYPES)].copy()

    if _comp_all.empty:
        callout("No labelled training types",
                "No session carries one of the MAIN_TYPES labels, so there is no "
                "composition to analyse.",
                C["yellow"], icon="🔍")
        _monthly_type_tss = pd.DataFrame(
            columns=["month_dt", "training_type", "type_tss", "pct_tss"])
    else:
        _comp_all["_month"] = _comp_all["date"].dt.to_period("M")
        # % of that month's TSS. groupby.transform() gives the monthly denominator
        # in place, so no second frame and no merge.
        _mt = (_comp_all.groupby(["_month", "training_type"], as_index=False)["tss"]
               .sum().rename(columns={"tss": "type_tss"}))
        _mt["month_tss"] = _mt.groupby("_month")["type_tss"].transform("sum")
        _mt["pct_tss"] = (_mt["type_tss"]
                          / _mt["month_tss"].replace(0, np.nan) * 100).fillna(0.0)
        _mt["month_dt"] = _mt["_month"].dt.to_timestamp()
        _monthly_type_tss = _mt

    # Per-month training "pattern" summary — the regressor for the FTP-gain tables.
    _outcome_rows = []
    for period, grp in _comp_all.groupby("_month"):
        total_tss = float(grp["tss"].sum())
        if total_tss <= 0:
            continue
        tss_by_type = grp.groupby("training_type")["tss"].sum()
        q_by_type = grp[grp["training_type"].isin(FTP_DRIVERS)].groupby("training_type")["tss"].sum()
        q_tss = float(q_by_type.sum())
        if q_tss > 0:
            qdom = q_by_type.idxmax()
            pattern = (f"Single: {qdom}" if float(q_by_type.max()) / q_tss * 100 > 50
                       else "Mixed: " + "+".join(sorted(q_by_type.nlargest(2).index)))
        else:
            pattern = "No quality sessions"
        _outcome_rows.append({
            "period": str(period),
            "total_tss": total_tss,
            "quality_pct": q_tss / total_tss * 100,
            "pattern": pattern,
            "combo": "+".join(sorted(tss_by_type.nlargest(3).index)),
        })
    _outcome_df = pd.DataFrame(_outcome_rows, columns=["period", "total_tss",
                                                       "quality_pct", "pattern", "combo"])
    if not _outcome_df.empty:
        _outcome_df["_month"] = pd.PeriodIndex(_outcome_df["period"], freq="M")

    # FTP proxy: best normalised power of the month × 0.95. Labelled a proxy
    # everywhere — it is not a measured threshold test.
    _pwr = pd.to_numeric(df_all["power_np"].fillna(df_all["power_avg"]), errors="coerce")
    _px = df_all[["date"]].assign(_pwr=_pwr)
    _px["_month"] = _px["date"].dt.to_period("M")
    _proxy_monthly = (_px[_px["_pwr"] > 50]
                      .groupby("_month", as_index=False)["_pwr"].max()
                      .rename(columns={"_pwr": "best_pwr"}))
    _proxy_monthly["ftp_proxy"] = _proxy_monthly["best_pwr"] * 0.95
    _proxy_monthly["month_dt"] = _proxy_monthly["_month"].dt.to_timestamp()
    _proxy_monthly["ftp_trend"] = _proxy_monthly["ftp_proxy"].rolling(3, min_periods=2).mean()
    _proxy_monthly["ftp_gain"] = _proxy_monthly["ftp_proxy"].diff()

    if not _outcome_df.empty:
        _outcome_df = _outcome_df.merge(
            _proxy_monthly[["_month", "ftp_proxy", "ftp_gain"]],
            on="_month", how="left")
    else:
        _outcome_df["ftp_proxy"] = pd.Series(dtype="float64")
        _outcome_df["ftp_gain"] = pd.Series(dtype="float64")

    n_months = int(_outcome_df["period"].nunique()) if len(_outcome_df) else 0
    st.caption(f"Full history: {len(_comp_all):,} labelled sessions across "
               f"{n_months} months ({df_all['date'].min():%b %Y} → "
               f"{df_all['date'].max():%b %Y}).")

    # Chart 1: Monthly composition — only if data exists
    if len(_monthly_type_tss) > 0:
        fig_comp = go.Figure()
        for t in MAIN_TYPES:
            t_data = _monthly_type_tss[_monthly_type_tss["training_type"] == t].sort_values("month_dt")
            if len(t_data) == 0:
                continue
            fig_comp.add_trace(go.Bar(
                x=t_data["month_dt"], y=t_data["pct_tss"], name=t,
                marker_color=TYPE_COLORS.get(t, C["muted"]), opacity=0.85,
                hovertemplate=f"<b>{t}</b><br>%{{y:.1f}}% of TSS<br>%{{x|%b %Y}}<extra></extra>"))
        fig_comp.update_layout(barmode="stack")
        style_figure(fig_comp, "Monthly composition — % TSS by type (all time)", H_STD)
        legend(fig_comp, "left", horizontal=True, size=10)
        show(fig_comp)
    else:
        callout("No data", "No training types from MAIN_TYPES found in this range.",
                C["muted"], icon="ℹ️")

    # Chart 2: FTP proxy — only if data exists
    if len(_proxy_monthly) > 0:
        fig_ref = go.Figure()
        fig_ref.add_trace(go.Scatter(
            x=_proxy_monthly["month_dt"], y=_proxy_monthly["ftp_proxy"], mode="lines+markers",
            name="FTP proxy (best NP × 0.95)", line=dict(color=C["purple"], width=2.5),
            marker=dict(size=4), fill="tozeroy", fillcolor="rgba(188,140,255,0.08)"))
        fig_ref.add_trace(go.Scatter(
            x=_proxy_monthly["month_dt"], y=_proxy_monthly["ftp_trend"], mode="lines",
            name="3-month trend", line=dict(color=C["accent"], width=2, dash="dash")))
        fig_ref.add_hline(y=FTP_TARGET, line_dash="dot", line_color=C["green"], opacity=0.5,
                          annotation_text=f"{FTP_TARGET} W peak")
        fig_ref.add_hline(y=FTP_CURRENT, line_dash="dot", line_color=C["yellow"],
                          annotation_text=f"{FTP_CURRENT} W current")
        fig_ref.add_vline(x=pd.Timestamp(SURGERY), line_dash="dash",
                          line_color=C["red"], opacity=0.7, annotation_text="Surgery")
        style_figure(fig_ref, "FTP proxy — correlate with the composition bars above", H_CARD)
        legend(fig_ref, "left", horizontal=True)
        show(fig_ref)
    else:
        callout("No data", "No power data > 50 W found for FTP proxy.",
                C["muted"], icon="ℹ️")

    section("Pattern vs next-month FTP change")
    cc = st.columns(2)
    with cc[0]:
        st.caption(
            "Single-dominant: one type >50% of quality TSS. Mixed: ≥2 types share "
            "it. FTP Δ = next month's proxy minus this month's."
        )
        _valid = _outcome_df[_outcome_df["ftp_gain"].notna()]
        if len(_valid) > 0:
            _pat = (_valid.groupby("pattern")
                    .agg(Months=("ftp_gain", "count"), Avg_gain=("ftp_gain", "mean"),
                         Median_gain=("ftp_gain", "median"), Avg_TSS=("total_tss", "mean"))
                    .reset_index().sort_values("Avg_gain", ascending=False).round(1))
            _pat.columns = ["Pattern", "Months", "Avg FTP Δ (W)", "Median Δ (W)", "Avg TSS"]
            _pat["Reliable"] = _pat["Months"].apply(lambda n: "✓" if n >= 5 else "⚠ n<5")
            st.dataframe(_pat, width="stretch", hide_index=True,
                         column_config={
                             "Avg FTP Δ (W)": st.column_config.NumberColumn(format="%.1f"),
                             "Median Δ (W)": st.column_config.NumberColumn(format="%.1f"),
                         })
            st.caption(f"⚠️ {len(_valid)} independent monthly observations — "
                       "directional, not statistically robust.")
        else:
            callout("Not enough data", "No monthly FTP gains yet in this range.",
                    C["muted"], icon="ℹ️")
    with cc[1]:
        st.caption("Top-3 types by TSS share per month, ranked by avg next-month FTP gain.")
        if len(_valid) > 0:
            _combo = (_valid.groupby("combo")
                      .agg(Months=("ftp_gain", "count"), Avg_gain=("ftp_gain", "mean"),
                           Avg_TSS=("total_tss", "mean"))
                      .reset_index().sort_values("Avg_gain", ascending=False).head(15).round(1))
            _combo.columns = ["Combination", "Months", "Avg FTP Δ (W)", "Avg TSS"]
            _combo["Reliable"] = _combo["Months"].apply(lambda n: "✓" if n >= 3 else "⚠ n<3")
            st.dataframe(_combo, width="stretch", hide_index=True,
                         column_config={
                             "Avg FTP Δ (W)": st.column_config.NumberColumn(format="%.1f"),
                         })
            st.caption("Most combinations appear only 1–2 times; only ✓ rows "
                       "(n ≥ 3) are directional.")
        else:
            callout("Not enough data", "No combinations to rank yet.",
                    C["muted"], icon="ℹ️")

    # ── Training type analysis (all time) ──────────────────────────────────
    section("🏋️ Training type analysis")
    st.caption("All-time data regardless of filters.")
    cc = st.columns(2)
    with cc[0]:
        type_counts = df_main_all["training_type"].value_counts().reset_index()
        type_counts.columns = ["type", "count"]
        fig_tc = px.bar(type_counts, x="count", y="type", orientation="h",
                        color="type", title="Sessions per training type (all time)",
                        color_discrete_map=TYPE_COLORS)
        style_figure(fig_tc, None, H_STD, showlegend=False)
        show(fig_tc)
    with cc[1]:
        type_if = (df_main_all.groupby("training_type")["if_score"]
                   .mean().reset_index().sort_values("if_score", ascending=True))
        fig_tif = go.Figure(go.Bar(
            x=type_if["if_score"], y=type_if["training_type"], orientation="h",
            marker_color=[C["accent"] if v < IF_Z2_MAX else C["yellow"] if v < IF_THRESHOLD
                          else C["red"] for v in type_if["if_score"]],
            text=[f"{v:.3f}" for v in type_if["if_score"]], textposition="outside"))
        fig_tif.add_vline(x=IF_Z2_MAX, line_dash="solid", line_color=C["orange"], opacity=0.8)
        fig_tif.add_vline(x=IF_THRESHOLD, line_dash="solid", line_color=C["red"], opacity=0.8)
        style_figure(fig_tif, "Avg intensity factor by type (all time)",
                     H_STD, showlegend=False)
        show(fig_tif)
        st.caption("Orange | 0.75 — SST boundary · red | 0.85 — FTP threshold")

    # ── Power zones ────────────────────────────────────────────────────────
    section("🎯 Power zones (FTP-relative)")
    zc = st.columns(4)
    for i, (zname, (lo, hi, zlabel, zcolor)) in enumerate(ZONES.items()):
        with zc[i % 4]:
            metric_card(zname,
                        f"{lo * FTP_CURRENT:.0f}–{min(hi, 9.99) * FTP_CURRENT:.0f} W"
                        if hi < 9.99 else f"< {lo * FTP_CURRENT:.0f} W",
                        foot=f"{zlabel} · {int(lo * 100)}–{int(min(hi, 9.99) * 100)}% FTP",
                        accent=zcolor)
