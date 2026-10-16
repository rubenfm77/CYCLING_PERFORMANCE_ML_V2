# views/training.py — Training page: method compliance & composition. (NEW FILE)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.components import (callout, dataframe, legend, metric_card, page_header,
                              section, show, stat_block)
from core.data import (FTP_AUTO_FRAC, FTP_AUTO_MAX_S, FTP_AUTO_MIN_S,
                       FTP_AUTO_SHARE)
from core.theme import (C, FTP_CURRENT, FTP_TARGET, FTP_DRIVERS, H_CARD, H_PAIR, H_STD,
                        IF_THRESHOLD, IF_VO2, IF_Z2_MAX, MAIN_TYPES, SURGERY, TYPE_COLORS,
                        ZONES, style_figure)


def _num(df, col):
    """Numeric view of a column that may be absent (API rows are sparse)."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _unlabelled_rows(unl: pd.DataFrame) -> pd.DataFrame:
    """One row per unlabelled session: why no rule claims it.

    Mirrors the auto-rule thresholds (core.data FTP_AUTO_*), so the reason
    is the failing condition itself, not a paraphrase: a best window under
    18:00, over 25:00, under 95% of the ride's own eFTP, or under 15% of
    the ride. Those rides stay unlabelled — hand-label them (SST/TEMPO by
    IF) rather than loosening the rule, which would also mislabel the
    surge rides the 15% share was built to refuse.
    """
    rows = []
    _pm_w = _num(unl, "icu_pm_ftp_watts")
    _pm_s = _num(unl, "icu_pm_ftp_secs")
    _ftp = _num(unl, "eftp")
    _dur = _num(unl, "duration_s")
    _if = _num(unl, "if_score")
    _tss = _num(unl, "tss").fillna(0.0)

    def _fmt(s):
        try:
            s = float(s)
        except (TypeError, ValueError):
            return "—"
        if pd.isna(s) or s <= 0:
            return "—"
        return f"{s:.0f} s" if s < 60 else f"{s / 60:.0f} min"

    for i in unl.index:
        w, s = _pm_w.get(i), _pm_s.get(i)
        try:
            w, s = float(w), float(s)
        except (TypeError, ValueError):
            w, s = float("nan"), float("nan")
        ftp, dur = _ftp.get(i), _dur.get(i)
        if pd.isna(w) or pd.isna(s) or s <= 0:
            why = "no sustained threshold window recorded by the meter"
            win = "—"
        elif s < FTP_AUTO_MIN_S:
            why = (f"best window {_fmt(s)} under the "
                   f"{FTP_AUTO_MIN_S / 60:.0f}:00 auto-FTP floor")
            win = f"{w:.0f} W / {_fmt(s)}"
        elif s > FTP_AUTO_MAX_S:
            why = (f"best window {_fmt(s)} over the "
                   f"{FTP_AUTO_MAX_S / 60:.0f}:00 auto-FTP ceiling")
            win = f"{w:.0f} W / {_fmt(s)}"
        elif pd.isna(ftp) or w < FTP_AUTO_FRAC * float(ftp):
            why = (f"{w:.0f} W under 95% of this ride's own eFTP"
                   if pd.notna(ftp) else "no eFTP reference on this ride")
            win = f"{w:.0f} W / {_fmt(s)}"
        elif pd.isna(dur) or s < FTP_AUTO_SHARE * float(dur):
            why = (f"only {s / float(dur):.0%} of the ride — under the 15% "
                   f"share a threshold session must own")
            win = f"{w:.0f} W / {_fmt(s)}"
        else:
            why = ("passes the window checks — the detector found no "
                   "matching sustained block (threshold corroboration)")
            win = f"{w:.0f} W / {_fmt(s)}"
        try:
            _d = pd.Timestamp(unl.at[i, "date"]).date().isoformat()
        except (TypeError, ValueError, KeyError):
            _d = "—"
        try:
            _ifv = float(_if.get(i))
            _ifs = f"{_ifv:.2f}" if pd.notna(_ifv) else "—"
        except (TypeError, ValueError):
            _ifs = "—"
        rows.append({"Date": _d, "TSS": round(float(_tss.get(i, 0.0))),
                     "Duration": _fmt(dur), "IF": _ifs,
                     "Best window": win, "Why unlabelled": why})
    return pd.DataFrame(rows,
                        columns=["Date", "TSS", "Duration", "IF",
                                 "Best window", "Why unlabelled"])


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
            # Honest denominator: unlabelled TSS is load too. Dropping it from
            # the pie silently inflates the driver share — the exact complaint
            # behind "impossible" percentages. It gets its own slice instead.
            unl = df.loc[~df["training_type"].isin(MAIN_TYPES)]
            unl_tss = float(pd.to_numeric(unl["tss"], errors="coerce").fillna(0).sum())
            _slices = [("FTP drivers", driver_tss, C["green"]),
                       ("Base volume", base_tss, C["muted"])]
            if unl_tss > 0:
                _slices.append(("Unlabelled", unl_tss, C["yellow"]))
            fig_pie = go.Figure(go.Pie(
                labels=[s[0] for s in _slices],
                values=[s[1] for s in _slices],
                marker_colors=[s[2] for s in _slices], hole=0.5,
                textinfo="label+percent"))
            style_figure(fig_pie, "TSS split — quality vs base volume",
                         H_PAIR, showlegend=False)
            show(fig_pie)
            # Per-source audit: how much of the driver slice the athlete filed
            # themselves vs the auto-rule. The older model had no auto-rule,
            # so any gap between the two numbers lives here, stated plainly.
            if "label_source" in df_main.columns:
                _w = float(df_main.loc[df_main["training_type"].isin(FTP_DRIVERS)
                                        & (df_main["label_source"] == "workout"),
                                        "tss"].sum())
                _a = float(df_main.loc[df_main["training_type"].isin(FTP_DRIVERS)
                                        & (df_main["label_source"] == "auto-ftp"),
                                        "tss"].sum())
                _b = float(df_main.loc[df_main["training_type"].isin(FTP_DRIVERS)
                                        & (df_main["label_source"] == "auto-billat"),
                                        "tss"].sum())
                st.caption(f"Driver TSS: **{_w:,.0f} athlete-labelled** + "
                           f"**{_a:,.0f} auto-ftp** + **{_b:,.0f} auto-billat** "
                           f"(every auto catch is "
                           f"listed with its numbers below). Unlabelled TSS "
                           f"excluded from driver/base: **{unl_tss:,.0f}**.")
                # The unlabelled slice, accounted for ride by ride: each row
                # names the exact auto-rule condition it fails, so "why is
                # this unlabelled" is answered by the numbers, not by trust.
                if len(unl):
                    with st.expander(
                            f"Why {len(unl):,} session(s) stay unlabelled"):
                        st.caption(
                            "One row per unlabelled session in this period. "
                            "The reason is the failing auto-rule condition "
                            "itself — window outside 18–25 min, under 95% of "
                            "the ride's own eFTP, under the 15% share, or no "
                            "sustained window at all. Billat-pattern reps are "
                            "counted in the Intervals detail, not here.")
                        dataframe(_unlabelled_rows(unl), height=300)
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
        "full history, ignoring the sidebar range. Monthly bars are % of "
        "labelled TSS only — the pie's Unlabelled slice is excluded from "
        "them, not hidden by them."
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

    # ── Threshold power: real measurements only ────────────────────────────
    # icu_pm_ftp_watts = intervals.icu's peak-meter FTP for the ride, measured over
    # icu_pm_ftp_secs. The window is whatever the ride contained, so it is NOT
    # comparable across rides — we keep only ~20-minute efforts (18–25 min), which
    # is the protocol actually being trained, and never mix windows.
    # eftp = intervals.icu rolling eFTP (a step function; it only moves when the
    # load justifies it).
    #
    # DO NOT reintroduce the old `power_np x 0.95` proxy: power_np is null on all
    # 188 of the 2026 sessions, so that silently plotted AVERAGE ride power and
    # called it FTP. It looked plausible and meant nothing.
    #
    # Both of these fields exist only from 2025 onward. There is no honest
    # pre-2025 threshold number in this dataset — the series starts late on
    # purpose rather than being back-filled with a proxy.
    PM_LO, PM_HI = FTP_AUTO_MIN_S, FTP_AUTO_MAX_S
    _pm_w = _num(df_all, "icu_pm_ftp_watts")
    _pm_s = _num(df_all, "icu_pm_ftp_secs")
    _eftp = _num(df_all, "eftp")
    _pm_ok = _pm_w.notna() & _pm_s.between(PM_LO, PM_HI)

    _px = df_all[["date"]].assign(pm_w=_pm_w, eftp=_eftp, pm_ok=_pm_ok)
    _px["_month"] = _px["date"].dt.to_period("M")
    _proxy_monthly = _px[_px["pm_ok"]].groupby("_month", as_index=False).agg(
        pm_ftp=("pm_w", "max"),          # best ~20-min expression that month
        pm_n=("pm_w", "size"),          # how many supported it
    )
    # eFTP is a rolling value — take the last reading in the month, not the max.
    _eftp_m = (_px.dropna(subset=["eftp"]).groupby("_month", as_index=False)["eftp"]
               .last().rename(columns={"eftp": "eftp_m"}))
    if len(_proxy_monthly) and len(_eftp_m):
        _proxy_monthly = _proxy_monthly.merge(_eftp_m, on="_month", how="outer")
    elif len(_eftp_m):
        _proxy_monthly = _eftp_m.rename(columns={"eftp_m": "pm_ftp"}).assign(pm_n=0)
    _proxy_monthly = _proxy_monthly.sort_values("_month")
    _proxy_monthly["month_dt"] = _proxy_monthly["_month"].dt.to_timestamp()
    _proxy_monthly["ftp_trend"] = _proxy_monthly["pm_ftp"].rolling(3, min_periods=2).mean()
    _proxy_monthly["ftp_gain"] = _proxy_monthly["pm_ftp"].diff()

    if not _outcome_df.empty:
        _outcome_df = _outcome_df.merge(
            _proxy_monthly[["_month", "pm_ftp", "ftp_gain"]],
            on="_month", how="left")
    else:
        _outcome_df["pm_ftp"] = pd.Series(dtype="float64")
        _outcome_df["ftp_gain"] = pd.Series(dtype="float64")

    n_months = int(_outcome_df["period"].nunique()) if len(_outcome_df) else 0
    n_pm = int(_outcome_df["pm_ftp"].notna().sum()) if len(_outcome_df) else 0
    st.caption(f"Composition: {len(_comp_all):,} labelled sessions across "
               f"{n_months} months ({df_all['date'].min():%b %Y} → "
               f"{df_all['date'].max():%b %Y}). "
               f"Threshold power: {n_pm} of those months have a ~20-min effort to "
               f"measure — peak-meter FTP and eFTP only exist from 2025, so the "
               f"tables below use {n_pm} months, not {n_months}. Fewer rows, but "
               f"they are measured watts instead of average power.")

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

    # Chart 2: measured threshold power — peak-meter FTP + eFTP
    if len(_proxy_monthly) > 0:
        _eftp_now = _num(df_all, "eftp").dropna()
        _eftp_last = float(_eftp_now.iloc[-1]) if len(_eftp_now) else None
        _pm = _proxy_monthly.dropna(subset=["pm_ftp"])
        fig_ref = go.Figure()
        if len(_pm):
            fig_ref.add_trace(go.Scatter(
                x=_pm["month_dt"], y=_pm["pm_ftp"], mode="lines+markers",
                name="Best ~20-min effort (peak-meter FTP)",
                line=dict(color=C["purple"], width=2.5), marker=dict(size=5),
                customdata=_pm["pm_n"],
                hovertemplate=("<b>%{y:.0f} W</b> · best 18–25 min effort"
                               "<br>%{x|%b %Y} · from %{customdata} effort(s)"
                               "<extra></extra>")))
            fig_ref.add_trace(go.Scatter(
                x=_pm["month_dt"], y=_pm["ftp_trend"], mode="lines",
                name="3-month trend", line=dict(color=C["accent"], width=2, dash="dash"),
                hovertemplate="%{y:.0f} W<extra></extra>"))
        _ef = _proxy_monthly.dropna(subset=["eftp_m"])
        if len(_ef):
            fig_ref.add_trace(go.Scatter(
                x=_ef["month_dt"], y=_ef["eftp_m"], mode="lines+markers",
                name="eFTP (intervals.icu rolling)",
                line=dict(color=C["green"], width=2, dash="dot"), marker=dict(size=4),
                hovertemplate="eFTP %{y:.0f} W<extra></extra>"))
        fig_ref.add_hline(y=FTP_TARGET, line_dash="dot", line_color=C["green"], opacity=0.4,
                          annotation_text=f"{FTP_TARGET} W pre-injury reference")
        if _eftp_last:
            fig_ref.add_hline(y=_eftp_last, line_dash="dot", line_color=C["yellow"],
                              opacity=0.7, annotation_text=f"eFTP now {_eftp_last:.0f} W")
        fig_ref.add_vline(x=pd.Timestamp(SURGERY), line_dash="dash",
                          line_color=C["red"], opacity=0.7, annotation_text="Surgery")
        style_figure(fig_ref, "Measured threshold power — read with the composition bars above", H_CARD)
        legend(fig_ref, "left", horizontal=True)
        show(fig_ref)
        st.caption(
            f"Both series start in {_pm['month_dt'].min():%b %Y} — intervals.icu only "
            f"reports peak-meter FTP and eFTP from 2025. {len(_pm)} months carry a "
            f"~20-min effort; a month backed by a single effort is marked in the "
            f"hover. Not comparable with anything before 2025, because no threshold "
            f"measurement exists there."
        )
    else:
        callout("No threshold data",
                "No ~20-min peak-meter effort found. intervals.icu reports this "
                "from 2025 onward only.",
                C["muted"], icon="ℹ️")

    # Removed: invented pattern/combo taxonomy (does not match the 11 agreed types).
    # The training-type analysis below uses the athlete's own labels only.

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

    # ── Labelling audit: show exactly how each type was decided ─────────────
    if "label_source" in df_all.columns:
        _src = df_all["label_source"].value_counts()
        _n_auto = int(_src.get("auto-ftp", 0))
        _n_lab = int(_src.get("workout", 0))
        _n_unl = int(_src.get("unlabelled", 0))
        _n_bil = int(_src.get("auto-billat", 0))
        _n_bas = int(_src.get("auto-base", 0))
        _src_bits = [f"**{_n_lab:,}** carry a workout name from intervals.icu"]
        if _n_auto:
            _src_bits.append(f"**{_n_auto:,}** were labelled FTP by the rule "
                             f"below")
        if _n_bil:
            _src_bits.append(f"**{_n_bil:,}** matched the Billat 30/30 row "
                             f"pattern and were filed BILLAT")
        if _n_bas:
            _src_bits.append(f"**{_n_bas:,}** easy or sensor-less rides were "
                             f"filed AEROBIC BASE")
        st.caption("Types come from " + ", ".join(_src_bits) + ". " +
                   (f"**{_n_unl:,}** sessions are still unlabelled and appear "
                    f"in no type-based view — that is a real gap in the data, "
                    f"not a rounding error." if _n_unl else
                    "Nothing is unlabelled."))
        _de = int(df_all.attrs.get("dupe_rows_dropped", 0) or 0)
        if _de:
            st.caption(
                f"Sessions counted: **{len(df_all):,}**. That is "
                f"**{_de:,} fewer** than the raw import, because the same ride was "
                f"often present twice — once from the head unit and once as a manual "
                f"upload. Each physical ride is counted once."
            )
        _lr = int(df_all.attrs.get("labels_recovered", 0) or 0)
        _lc = int(df_all.attrs.get("labels_carried", 0) or 0)
        if _lr or _lc:
            _bits = []
            if _lc:
                _bits.append(
                    f"**{_lc:,}** label(s) from the last 60 days were carried over "
                    f"from the CSV onto their intervals.icu row, which can never "
                    f"supply one"
                )
            if _lr:
                _bits.append(
                    f"**{_lr:,}** label(s) survived a duplicate-pair merge because the "
                    f"copy that survived was the unlabelled twin"
                )
            st.caption(
                "Without these two steps the most recent two months of training — "
                "and any categorised session that happened to be double-imported — "
                "would silently drop out of every type-based view. Recovered: "
                + "; ".join(_bits) + "."
            )
        with st.expander("🔍 How the FTP auto-label works — and every session it caught"):
            st.markdown(
                f"**Rule.** A session with *no* workout name is labelled **FTP** when "
                f"it contains a peak-meter effort of "
                f"**{FTP_AUTO_MIN_S // 60}–{FTP_AUTO_MAX_S // 60} min** at "
                f"**≥ {FTP_AUTO_FRAC:.0%} of that session's own eFTP** *and* that "
                f"window spans **≥ {FTP_AUTO_SHARE:.0%} of the ride itself**.\n\n"
                f"The first three conditions say the effort was threshold work. The "
                f"share test says the RIDE was built around it — it is what keeps a "
                f"3-hour endurance ride with one 21-min surge inside from being typed "
                f"FTP. Every matched session below prints both of its numbers.\n\n"
                f"Each session is compared against the eFTP recorded *on that ride*, "
                f"not today's — eFTP is a rolling value, so using the current one on "
                f"an old session would be anachronistic.\n\n"
                f"The window is the same one the measured-threshold chart above uses, "
                f"so the label a session gets and the number that chart plots always "
                f"come from the same definition. Nothing is overwritten silently: the "
                f"original value is kept in `training_type_raw`."
            )
            if _n_auto:
                _a = df_all[df_all["label_source"] == "auto-ftp"].copy()
                _a["effort"] = (_a["auto_ftp_effort_s"] / 60).round(1)
                _a["min"] = _a["effort"].map(lambda m: f"{m:.0f} min")
                _a["effort"] = _a["auto_ftp_effort_w"].round(0)
                _a["pct"] = (_a["auto_ftp_effort_w"] / _a["auto_ftp_threshold_w"]).round(3)
                if "duration_s" in _a.columns:
                    _d = pd.to_numeric(_a["duration_s"], errors="coerce")
                    _a["share"] = (_a["auto_ftp_effort_s"] / _d).round(3)
                else:
                    _a["share"] = np.nan
                _a = _a[["date", "effort", "min", "pct", "share"]].sort_values(
                    "date", ascending=False)
                _a["date"] = _a["date"].dt.strftime("%Y-%m-%d")
                _a["pct"] = _a["pct"].map(lambda p: f"{p:.1%}")
                _a["share"] = _a["share"].map(
                    lambda s: f"{s:.0%}" if pd.notna(s) else "—")
                _a.columns = ["Date", "Effort (W)", "Window", "% of eFTP",
                              "Share of ride"]
                st.dataframe(_a, width="stretch", hide_index=True,
                             column_config={
                                 "Effort (W)": st.column_config.NumberColumn(format="%.0f"),
                                 "% of eFTP": st.column_config.TextColumn(),
                                 "Share of ride": st.column_config.TextColumn(),
                             })
                st.caption(f"{_n_auto} session(s) matched. A % of eFTP near 100 % is "
                           f"threshold work; far above it means a hard outlier effort. "
                           f"Share of ride is the window over the whole session, and "
                           f"must clear {FTP_AUTO_SHARE:.0%} for the label to apply.")
                _rej = int(df_all.attrs.get("auto_ftp_rejected", 0) or 0)
                if _rej:
                    st.caption(f"**{_rej} further session(s)** tripped the power gate "
                               f"but the interval detector holds no sustained block "
                               f"there — a surge inside another ride, not a threshold "
                               f"session — so they stayed unlabelled instead of "
                               f"becoming FTP.")
            else:
                callout("No sessions matched", "The rule caught nothing in this dataset.",
                        C["muted"], icon="ℹ️")

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
