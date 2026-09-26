# core/metrics.py — headline numbers computed once per run. (NEW FILE)
#
# Everything the persistent chrome (top bar, KPI ribbon, fatigue state,
# next-week advice) needs, derived from the same logic app.py used so the
# multipage dashboard cannot drift from it.

from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

from core.data import safe_sum, safe_mean
from core.theme import C, IF_VO2, STATE_COLORS


def headline(ctx) -> SimpleNamespace:
    """Compute the numbers shared by the router chrome and the Overview page."""
    df_all, df_all_f = ctx.df_all, ctx.df_all_f

    # ── This week vs last week (respects the sidebar type filter, like app.py)
    now = pd.Timestamp.now()
    week_start = now - timedelta(days=now.weekday())
    this_week = df_all_f[df_all_f["date"] >= week_start]
    last_week = df_all_f[(df_all_f["date"] >= week_start - timedelta(days=7))
                         & (df_all_f["date"] < week_start)]

    tw_tss, lw_tss = safe_sum(this_week["tss"]), safe_sum(last_week["tss"])
    tw_elev, lw_elev = safe_sum(this_week["elevation"]), safe_sum(last_week["elevation"])
    tw_if, lw_if = safe_mean(this_week["if_score"]), safe_mean(last_week["if_score"])
    tw_pwr, lw_pwr = safe_mean(this_week["power_avg"]), safe_mean(last_week["power_avg"])

    # ── Current fitness state (always full history — filters must not move it)
    latest = df_all.iloc[-1]
    prev = df_all.iloc[-8] if len(df_all) >= 8 else df_all.iloc[0]
    ctl_now, atl_now, tsb_now = float(latest["ctl"]), float(latest["atl"]), float(latest["tsb"])
    state = latest["fatigue_state"]
    state_color = STATE_COLORS.get(state, C["accent"])

    # eFTP (rolling estimate from Intervals.icu)
    eftp_val, eftp_delta = None, None
    if "eftp" in df_all.columns:
        eftp_s = pd.to_numeric(df_all["eftp"], errors="coerce").dropna()
        if len(eftp_s) > 0:
            eftp_val = float(eftp_s.iloc[-1])
            if len(eftp_s) > 1:
                eftp_delta = eftp_val - float(eftp_s.iloc[-2])

    wkg_s = df_all["w_per_kg"].dropna()
    wkg_val = float(wkg_s.tail(10).mean()) if len(wkg_s) >= 10 else float(wkg_s.mean())

    # ── Cardiac-efficiency delta (objective fatigue — replaces RPE) ─────────
    _eff_cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
    recent_eff = df_all[(df_all["efficiency"].notna())
                        & (df_all["date"] >= _eff_cutoff)].copy()
    if len(recent_eff) < 10:
        recent_eff = df_all[df_all["efficiency"].notna()].tail(35).copy()
    curr_eff = recent_eff["efficiency"].iloc[-1] if len(recent_eff) > 0 else np.nan
    avg_eff_28d = recent_eff["eff_28d"].iloc[-1] if len(recent_eff) > 0 else np.nan
    eff_delta = ((curr_eff - avg_eff_28d) / avg_eff_28d * 100) if avg_eff_28d else np.nan

    # ── Next-week recommendation (same rule table as app.py) ────────────────
    fatigue_override = bool(eff_delta and not np.isnan(eff_delta) and eff_delta < -5)
    if fatigue_override:
        r_color, r_title, r_body = C["red"], "🚨 Physiological fatigue detected", (
            f"W/BPM is {abs(eff_delta):.1f}% below baseline despite TSB = {tsb_now:+.1f}. "
            "Your body is more fatigued than TSB suggests. Reduce load this week."
        )
    elif tsb_now < -30:
        r_color, r_title, r_body = C["red"], "🚨 Rest or very easy week", (
            f"TSB = {tsb_now:.1f} — overreached. Max 3 easy Z2 sessions. No quality until TSB > −20."
        )
    elif tsb_now < -10:
        r_color, r_title, r_body = C["orange"], "💪 Continue hard block", (
            f"TSB = {tsb_now:.1f} — deep build. 2 quality sessions: Wednesday FTP + Saturday PIRAMIDAL."
        )
    elif tsb_now < 5:
        r_color, r_title, r_body = C["yellow"], "✅ Standard build week", (
            f"TSB = {tsb_now:.1f} — build phase. Wednesday FTP/SST + Saturday PIRAMIDAL with TEMPO blocks."
        )
    elif tsb_now < 20:
        r_color, r_title, r_body = C["green"], "🟢 Push hard this week", (
            f"TSB = {tsb_now:.1f} — fresh. 4×10min FTP Wednesday + long PIRAMIDAL Saturday."
        )
    else:
        r_color, r_title, r_body = C["purple"], "⚡ Peak form — race or FTP test", (
            f"TSB = {tsb_now:.1f} — peak form. 20-min FTP test or hardest session of the block."
        )

    # ── All-time Norwegian counts (methodology + training pages) ────────────
    all_time_q = int(df_all["is_quality"].sum())
    all_time_vo2 = int((df_all["if_score"] >= IF_VO2).sum())

    return SimpleNamespace(**locals())
