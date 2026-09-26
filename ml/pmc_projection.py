# ml/pmc_projection.py — deterministic PMC projection. (NEW FILE)
"""
Project the Performance Management Chart forward under an assumed TSS plan.

This is NOT machine learning and does not pretend to be. The PMC is an
exactly specified EWMA — core.data builds it with pandas
ewm(span=S, adjust=False), i.e. y_t = y_(t-1) + alpha * (x_t - y_(t-1))
with alpha = 2 / (S + 1). Given a TSS plan, where the meters will point is
pure bookkeeping: no fitting, no leakage, nothing to overfit.

Projection rules:
  * warm start = the dashboard's own current CTL/ATL (the same numbers in
    the top bar), so projection and reality cannot disagree at t0;
  * the dashboard's PMC advances ONLY on a session row (ewm over rows), so
    rest days and any gap since the newest ride HOLD state flat — the
    projection therefore steps ONLY on ride days, spaced by your recent
    cadence (session rows per week over the last 28 days), each carrying
    weekly/cadence TSS (one ride per day assumed; multi-ride days are a
    <0.2% effect);
  * consequence of that same chart math: steady-state CTL ≈ mean TSS PER
    RIDE, so the same weekly budget spread over more, shorter rides lowers
    this chart's meter (fewer, bigger rides raise it). The projection
    reproduces the chart's behaviour faithfully; it adds no physiology;
  * what it CANNOT do: predict adaptation, freshness or race performance.
    It shows where the LOAD meters point IF you hit the planned TSS.
"""

import pandas as pd

# pandas ewm(span=S, adjust=False) uses alpha = 2 / (S + 1)
ALPHA_CTL = 2 / (42 + 1)   # == core.data's ctl EWMA
ALPHA_ATL = 2 / (7 + 1)    # == core.data's atl EWMA

# UI label -> internal pattern code (kept here so the view stays thin)
PATTERNS = {
    "Flat — usual cadence, even TSS": "flat",
    "6 on / 1 rest day": "6on1off",
    "3 build weeks + 1 recovery week": "periodized",
}


def recent_cadence(df_all, days: int = 28) -> float:
    """Session rows per week over the last `days` days — the chart's own
    step frequency (its EWMA advances once per row). Clamped to 3–7 so a
    thin window can't produce an absurd schedule."""
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    tss = pd.to_numeric(df_all.loc[df_all["date"] >= cutoff, "tss"],
                        errors="coerce")
    n = float(tss.notna().sum()) * 7.0 / days
    return float(min(7.0, max(3.0, n)))


def recent_weekly_tss(df_all, days: int = 28) -> int:
    """Rounded mean weekly TSS of the last `days` calendar days (default)."""
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    total = float(df_all.loc[df_all["date"] >= cutoff, "tss"].sum())
    weekly = total * 7.0 / days
    return int(max(50, round(weekly / 25) * 25))


def _raw_recent_weekly(df_all, days: int = 28) -> float:
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    total = float(df_all.loc[df_all["date"] >= cutoff, "tss"].sum())
    return total * 7.0 / days


def _daily_tss(weekly: float, n_days: int, pattern: str,
               cadence: float) -> list:
    """Per-day TSS for the session-row recursion (0.0 = rest day).

    Ride days are spaced by `cadence` (rides/week, fractional, on a 28-day
    cycle) and each ride carries `weekly / cadence` TSS, so the weekly
    budget is preserved *per ride* — the unit the chart's row-EWMA
    actually converges to. `6on1off` overrides cadence with 6 rides;
    `periodized` halves the budget in every 4th week.
    """
    if pattern == "6on1off":
        return [0.0 if i % 7 == 6 else weekly / 6.0
                for i in range(n_days)]
    n28 = cadence * 4.0  # rides per 28-day cycle (multiples of 0.25 exact)
    out = []
    for i in range(n_days):
        rides = int((i + 1) * n28 / 28.0) - int(i * n28 / 28.0)
        budget = weekly * (0.5 if (pattern == "periodized"
                                   and (i // 7) % 4 == 3) else 1.0)
        out.append(budget / cadence if rides else 0.0)
    return out


def project(df_all, weekly_tss: float, horizon_days: int = 28,
            pattern: str = "flat") -> pd.DataFrame:
    """Walk the EWMA forward from the dashboard's current state.

    Steps happen only on ride days — rest days and any gap since the
    newest ride hold the meters flat, exactly like the chart's
    session-row EWMA. Returns a frame with one row per day: date, kind
    (anchor/gap/plan), tss, and both scenarios — `*_plan` (the slider)
    and `*_recent` (last 28 days continued unchanged, for comparison).
    """
    latest = df_all.iloc[-1]
    ctl = float(latest["ctl"])
    atl = float(latest["atl"])
    raw = latest.get("tss", 0.0)
    tss_now = 0.0 if pd.isna(raw) else float(raw)
    last_ride = pd.Timestamp(latest["date"])
    today = pd.Timestamp.now().normalize()
    cadence = recent_cadence(df_all)

    cp, ap = ctl, atl     # plan scenario
    cr, ar = ctl, atl     # recent-pace scenario
    rows = []

    def _snap(d, kind, tss):
        rows.append({
            "date": d, "kind": kind, "tss": tss,
            "ctl_plan": cp, "atl_plan": ap, "tsb_plan": cp - ap,
            "ctl_recent": cr, "atl_recent": ar, "tsb_recent": cr - ar,
        })

    # Anchor on the newest session row — the exact point the chart's line
    # ends, so actual and projected lines connect with no jump.
    _snap(last_ride, "anchor", tss_now)

    # Any gap since that ride does NOT move the meters: the chart's
    # session-level EWMA only steps on a row. Draw it flat.
    d = last_ride.normalize() + pd.Timedelta(days=1)
    while d <= today:
        _snap(d, "gap", 0.0)
        d += pd.Timedelta(days=1)

    # Apply the plan (and the status-quo comparison) forward. Ride days
    # step the recursion; rest days repeat the previous state (tss = 0).
    start = max(today, last_ride.normalize()) + pd.Timedelta(days=1)
    plan_tss = _daily_tss(weekly_tss, horizon_days, pattern, cadence)
    recent_tss = _daily_tss(_raw_recent_weekly(df_all), horizon_days,
                            "flat", cadence)
    for i in range(horizon_days):
        tp, tr = plan_tss[i], recent_tss[i]
        if tp > 0.0:
            cp += ALPHA_CTL * (tp - cp)
            ap += ALPHA_ATL * (tp - ap)
        if tr > 0.0:
            cr += ALPHA_CTL * (tr - cr)
            ar += ALPHA_ATL * (tr - ar)
        _snap(start + pd.Timedelta(days=i), "plan", tp)

    return pd.DataFrame(rows)


def plan_metrics(proj: pd.DataFrame, horizon_days: int) -> dict:
    """Coach-facing summary numbers of the projection (all exact, no fit)."""
    anchor = proj[proj["kind"] == "anchor"].iloc[-1]
    plan = proj[proj["kind"] == "plan"]
    anchor_ctl = float(anchor["ctl_plan"])
    anchor_tsb = float(anchor["tsb_plan"])

    peak = float(plan["ctl_plan"].max())
    ramp_wk = (float(plan["ctl_plan"].iloc[-1]) - anchor_ctl) / (horizon_days / 7.0)
    pct_week = (ramp_wk / anchor_ctl * 100.0) if anchor_ctl > 0 else 0.0
    # Ramp guidance on |slope| (a fast collapse matters as much as a fast build):
    # <= 5 CTL/week fine, <= 8 moderate, above = aggressive / fast detraining.
    mag = abs(ramp_wk)
    tone = "good" if mag <= 5 else ("warn" if mag <= 8 else "bad")
    if ramp_wk >= 0:
        ramp_word = {"good": "conservative build", "warn": "moderate build",
                     "bad": "aggressive build"}[tone]
    else:
        ramp_word = {"good": "gentle decay", "warn": "moderate decay",
                     "bad": "fast detraining"}[tone]

    tsb_end = float(plan["tsb_plan"].iloc[-1])
    fresh = plan[plan["tsb_plan"] >= 10]
    taper = fresh["date"].iloc[0] if len(fresh) else None

    # Ride-day structure actually applied (the row-EWMA's real unit is
    # TSS per ride, so surface it instead of hiding it).
    rides = plan[plan["tss"] > 0]
    cadence = len(rides) * 7.0 / horizon_days if horizon_days else 0.0
    per_ride = float(rides["tss"].mean()) if len(rides) else 0.0

    return {
        "anchor_ctl": anchor_ctl, "anchor_tsb": anchor_tsb,
        "cadence": cadence, "per_ride": per_ride,
        "peak": peak, "peak_delta": peak - anchor_ctl,
        "ramp_wk": ramp_wk, "pct_week": pct_week, "tone": tone,
        "ramp_word": ramp_word,
        "tsb_end": tsb_end, "tsb_delta": tsb_end - anchor_tsb,
        "taper": taper,
    }
