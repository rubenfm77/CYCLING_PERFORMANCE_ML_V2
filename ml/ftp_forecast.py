# ml/ftp_forecast.py — honest, validated eFTP forecasting. (NEW FILE)
"""
Walk-forward validated eFTP forecast for the Forecast page.

Design rules, learned by auditing the original src/ experiments:

1. NEVER predict a formula from its own ingredients. (src/ftp_analysis.py
   predicted IF^2 x duration from features containing IF and duration and
   then "discovered" TSS — pure target leakage.) Here the target is
   Intervals.icu's own rolling eFTP estimate; features are past load only.
2. TIME-ORDERED VALIDATION ONLY. To predict eFTP at t+H the model trains
   on pairs (state at s -> eFTP at s+H) whose target date never exceeds
   the test cutoff (embargo by target date). No random folds on a series.
3. COMPETE AGAINST THE TOUGHEST HONEST BASELINE — persistence ("eFTP
   stays where it is"). If nothing beats it, say so and show it.
   Negative skill is reported, never hidden.
4. SHOW THE COEFFICIENTS. The coach sees exactly what the model weighs.

Models: persistence (baseline) . drift (trailing linear trend) .
ridge (causal 28-day load features, standardised, L2-regularised).

Uncertainty: empirical quantiles of walk-forward errors at the chosen
horizon, widened with sqrt(time) — labelled as such in the UI.
"""

import math

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["trend_y", "post_op_d", "tss_28", "hi_share_28", "if_28",
            "ctl_ramp_28", "ctl_level"]

FEATURE_META = {
    "trend_y":     ("Calendar trend (years)", "Long-term direction of the series"),
    "post_op_d":   ("Days since surgery", "Recovery slope after Jun 2025"),
    "tss_28":      ("TSS, last 28 days", "How much load was absorbed recently"),
    "hi_share_28": ("Quality share (IF >= 0.85)", "Portion of recent load that was threshold or harder"),
    "if_28":       ("Mean IF, last 28 days", "Average intensity of the block"),
    "ctl_ramp_28": ("CTL change, 4 weeks", "Fitness trend the PMC already measured"),
    "ctl_level":   ("CTL level", "Current chronic-load baseline"),
}

MODEL_LABELS = {
    "persistence": "Persistence — eFTP stays put",
    "drift": "Drift — trailing trend",
    "ridge": "Ridge — causal load features",
}

DRIFT_K = 6      # trailing observations used by the drift model
MIN_TRAIN = 30   # minimum backtest training pairs to fit on


# ── Data prep ────────────────────────────────────────────────────────────────
def _observations(df_all) -> pd.DataFrame:
    """Clean eFTP observation series: one value per day, sorted, sanity-capped."""
    if "eftp" not in df_all.columns:
        return pd.DataFrame(columns=["date", "eftp"])
    obs = df_all[["date", "eftp"]].copy()
    obs["eftp"] = pd.to_numeric(obs["eftp"], errors="coerce")
    obs = obs.dropna(subset=["eftp"])
    obs = obs[(obs["eftp"] > 100) & (obs["eftp"] < 700)]
    obs["date"] = obs["date"].dt.normalize()
    obs = obs.sort_values("date").drop_duplicates("date", keep="last")
    return obs.reset_index(drop=True)


def _daily_frame(df_all) -> pd.DataFrame:
    """Daily causal training-load features from the session history."""
    g = df_all.sort_values("date").set_index("date")
    hi = g.loc[g["is_quality"].fillna(False), "tss"]
    daily = pd.DataFrame({
        "tss_d": g["tss"].resample("1D").sum(),
        "if_d": g["if_score"].resample("1D").mean(),
        "hi_d": hi.resample("1D").sum(),
        "ctl_d": g["ctl"].resample("1D").last(),
    })
    daily["hi_d"] = daily["hi_d"].fillna(0.0)
    daily["ctl_d"] = daily["ctl_d"].ffill()
    daily["tss_28"] = daily["tss_d"].rolling(28, min_periods=7).sum()
    hi_28 = daily["hi_d"].rolling(28, min_periods=7).sum()
    daily["hi_share_28"] = (hi_28 / daily["tss_28"].replace(0.0, np.nan)).clip(0, 1)
    daily["if_28"] = daily["if_d"].rolling(28, min_periods=7).mean()
    daily["ctl_ramp_28"] = daily["ctl_d"] - daily["ctl_d"].shift(28)
    return daily.rename_axis("date").reset_index()


def _align(daily: pd.DataFrame, dates, first, surg) -> pd.DataFrame:
    """As-of (backward) feature lookup: every date sees only data <= itself."""
    left = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates)).dt.normalize()})
    left = left.sort_values("date").reset_index(drop=True)
    m = pd.merge_asof(left, daily, on="date", direction="backward")
    m["trend_y"] = (m["date"] - first).dt.days / 365.25
    m["post_op_d"] = (m["date"] - surg).dt.days.clip(lower=0).astype(float)
    m["ctl_level"] = m["ctl_d"]
    return m


def _make_ridge() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])


# ── Validation ───────────────────────────────────────────────────────────────
def _pair_indices(dates: np.ndarray, horizon: int):
    """(i, k) pairs: forecast issued at obs i, verified at first obs >= i + H."""
    need = dates + np.timedelta64(horizon, "D")
    ks = np.searchsorted(dates, need, side="left")
    return [(int(i), int(k)) for i, k in enumerate(ks) if k < len(dates) and k > i]


def _walk_forward(X, y, dates, pairs):
    """Signed residuals (actual - predicted) per model, strictly time-ordered.

    Embargo: the ridge training set only contains pairs whose TARGET date is
    already <= the test issue date — the model never trains on an eFTP value
    that had not been observed yet at forecast time.
    """
    pi = np.array([p[0] for p in pairs], dtype=int)
    pk = np.array([p[1] for p in pairs], dtype=int)
    k_dates = dates[pk]
    test_start = max(int(len(y) * 0.55), 40)
    errs = {"persistence": [], "drift": [], "ridge": []}

    for i, k in pairs:
        if i < test_start:
            continue
        n_train = int(np.searchsorted(k_dates, dates[i], side="right"))
        if n_train < MIN_TRAIN:
            continue

        # persistence — the forecast is "last known value, unchanged"
        r_pers = float(y[k] - y[i])

        # drift — OLS line through the trailing DRIFT_K observations
        lo = max(0, i - DRIFT_K + 1)
        dx = ((dates[lo:i + 1] - dates[i]) / np.timedelta64(1, "D")).astype(float)
        slope, intercept = np.polyfit(dx, y[lo:i + 1], 1)
        xk = float((dates[k] - dates[i]) / np.timedelta64(1, "D"))
        r_drift = float(y[k] - (slope * xk + intercept))

        # ridge — train only on pairs whose target was already observed
        model = _make_ridge()
        model.fit(X[pi[:n_train]], y[pk[:n_train]])
        r_ridge = float(y[k] - model.predict(X[[i]])[0])

        errs["persistence"].append(r_pers)
        errs["drift"].append(r_drift)
        errs["ridge"].append(r_ridge)

    return errs


# ── Public entry point ───────────────────────────────────────────────────────
def run_forecast(df_all, horizon_days: int, surgery_date: str,
                 min_obs: int = 60) -> dict:
    """Backtest all three models at `horizon_days`, pick the honest winner,
    fit it on everything, and forecast one horizon step beyond today.

    Returns a plain dict the view renders (ok/reason on failure paths).
    """
    obs = _observations(df_all)
    if len(obs) < min_obs:
        return {"ok": False,
                "reason": f"Only {len(obs)} usable eFTP observations "
                          f"(need ≥ {min_obs}) — not enough history to "
                          f"validate a {horizon_days}-day forecast."}

    daily = _daily_frame(df_all)
    first = obs["date"].iloc[0]
    surg = pd.Timestamp(surgery_date)

    X = _align(daily, obs["date"], first, surg)[FEATURES].to_numpy(dtype=float)
    y = obs["eftp"].to_numpy(dtype=float)
    dates = obs["date"].to_numpy()

    pairs = _pair_indices(dates, horizon_days)
    if len(pairs) < 25:
        return {"ok": False,
                "reason": f"Only {len(pairs)} valid {horizon_days}-day "
                          f"backtest pairs — too few to validate a forecast."}

    errs = _walk_forward(X, y, dates, pairs)
    n_backtests = len(errs["persistence"])
    if n_backtests < 8:
        return {"ok": False,
                "reason": f"Only {n_backtests} valid backtests at "
                          f"{horizon_days} days — too few to trust a band."}

    mae = {m: float(np.mean(np.abs(v))) for m, v in errs.items()}
    mae_p = max(mae["persistence"], 1e-9)
    skill = {m: (0.0 if m == "persistence" else 1.0 - mae[m] / mae_p)
             for m in mae}
    # ties resolve to persistence (prefer the baseline over a marginal model)
    selected = min(mae, key=lambda m: (round(mae[m], 3), m != "persistence"))
    label = MODEL_LABELS[selected]

    # ── Fit the winner on all pairs and forecast beyond today ───────────────
    today = pd.Timestamp.now().normalize()
    y_now = float(y[-1])
    pi = np.array([p[0] for p in pairs], dtype=int)
    pk = np.array([p[1] for p in pairs], dtype=int)
    coef_table, note, yhat = None, "", y_now

    if selected == "ridge":
        model = _make_ridge()
        model.fit(X[pi], y[pk])
        x_today = _align(daily, pd.Series([today]), first, surg)[FEATURES]
        yhat = float(model.predict(x_today.to_numpy(dtype=float))[0])
        coefs = model.named_steps["ridge"].coef_
        rows = []
        for j in np.argsort(-np.abs(coefs)):
            name, blurb = FEATURE_META[FEATURES[j]]
            c = float(coefs[j])
            rows.append({"Feature": name, "Std. weight": f"{c:+.2f}",
                         "Effect": "↑ raises forecast" if c >= 0 else "↓ lowers forecast",
                         "What it is": blurb})
        coef_table = pd.DataFrame(rows)
        note = (f"Ridge fitted on {len(pairs)} (state → eFTP {horizon_days}d "
                f"later) pairs, α = 1.0, features standardised — bigger "
                f"|weight| = more influence on the forecast.")
    elif selected == "drift":
        lo = max(0, len(y) - DRIFT_K)
        dx = ((dates[lo:] - dates[-1]) / np.timedelta64(1, "D")).astype(float)
        slope, intercept = np.polyfit(dx, y[lo:], 1)
        x_end = float((today + pd.Timedelta(days=horizon_days)
                       - obs["date"].iloc[-1]) / np.timedelta64(1, "D"))
        yhat = float(slope * x_end + intercept)
        note = (f"Drift: straight line through your last {len(y) - lo} eFTP "
                f"tests — {slope * 7:+.1f} W/week — extrapolated. "
                f"No features, just the recent trend.")
    else:
        note = (f"No model beat the flat line: over {horizon_days} days your "
                f"eFTP has moved less than model error, so the honest "
                f"forecast is “stays where it is”.")

    why = (f"Selected **{label}** — lowest walk-forward MAE "
           f"({mae[selected]:.1f} W) across {n_backtests} backtests "
           f"{horizon_days} days ahead. Each backtest trained only on targets "
           f"already known at issue time (embargo by target date). "
           f"Doing nothing scored {mae['persistence']:.1f} W.")
    band_note = (f"Band = 50% / 90% quantiles of those {n_backtests} "
                 f"backtest errors, widened with √time.")

    # ── Path + bands (interpolated to the endpoint, √time widening) ─────────
    q05, q25, q75, q95 = (float(np.quantile(errs[selected], p))
                           for p in (0.05, 0.25, 0.75, 0.95))
    endpoint = today + pd.Timedelta(days=horizon_days)
    rows = [{"date": today, "y": y_now, "lo50": y_now, "hi50": y_now,
             "lo90": y_now, "hi90": y_now}]
    steps = sorted(set(range(7, horizon_days + 1, 7)) | {horizon_days})
    for h in steps:
        frac = h / horizon_days
        center = y_now + (yhat - y_now) * frac
        s = math.sqrt(frac)
        rows.append({"date": today + pd.Timedelta(days=h), "y": center,
                     "lo50": center + q25 * s, "hi50": center + q75 * s,
                     "lo90": center + q05 * s, "hi90": center + q95 * s})
    path = pd.DataFrame(rows)

    # ── Model comparison table (everything visible, nothing hidden) ─────────
    mt_rows = []
    for m in ("persistence", "drift", "ridge"):
        mt_rows.append({
            "Model": MODEL_LABELS[m],
            "MAE (W)": round(mae[m], 1),
            "Skill vs doing nothing": ("baseline" if m == "persistence"
                                       else f"{skill[m] * 100:+.0f}%"),
            "Verdict": "✓ selected" if m == selected else "—",
        })

    return {
        "ok": True, "obs": obs, "path": path,
        "y_now": y_now, "yhat": yhat, "endpoint": endpoint,
        "horizon_days": horizon_days,
        "n_obs": int(len(obs)),
        "first_date": obs["date"].iloc[0], "last_date": obs["date"].iloc[-1],
        "n_backtests": n_backtests, "mae": mae, "skill": skill,
        "selected": selected, "selected_label": label,
        "model_table": pd.DataFrame(mt_rows), "coef_table": coef_table,
        "why": why, "note": note, "band_note": band_note,
    }
