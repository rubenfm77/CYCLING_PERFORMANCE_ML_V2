# ml/exertion_forecast.py — honest exertion persistence test. (NEW FILE)
"""
Walk-forward validation of "does exertion persist?" for the Intervals page.

Same four rules as ml/ftp_forecast.py:

1. NEVER predict a formula from its own ingredients.
2. TIME-ORDERED VALIDATION ONLY — embargo by target date: the ridge training
   set contains only pairs whose target was already observed at issue time.
3. COMPETE AGAINST PERSISTENCE ("yesterday's hard ride feels like today's").
   Negative skill is reported, never hidden.
4. SHOW THE COEFFICIENTS.

Target choice (the user has NEVER logged RPE — icu_rpe is empty for all
1,604 activities), selectable in the UI:

  strain      — intervals.icu strain_score (the default proxy; independent
                of power-based TSS, r = 0.24)
  strain_rate — strain_score per hour of riding (duration-normalised)
  trimp       — HR-based TRIMP (dense since Apr 2026; behaves like load)
  icu_rpe     — true RPE; works the moment the athlete starts logging it

Units differ per metric; the model table reports MAE in that metric's units.
"""

import math

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["trend_y", "post_op_d", "ex_7", "ex_28", "ex_trend",
            "tss_28", "hi_share_28", "ctl_level"]

FEATURE_META = {
    "trend_y":       ("Calendar trend (years)", "Long-term direction"),
    "post_op_d":     ("Days since surgery", "Recovery slope after Jun 2025"),
    "ex_7":          ("Exertion mean, 7 days", "How hard the last week felt"),
    "ex_28":         ("Exertion mean, 28 days", "Recent block exertion level"),
    "ex_trend":      ("Exertion trend (7d − 28d)", "Harder or easier than the block"),
    "tss_28":        ("TSS, last 28 days", "How much load was absorbed"),
    "hi_share_28":   ("Quality share (IF ≥ 0.85)", "Portion of recent load that was threshold+"),
    "ctl_level":     ("CTL level", "Current chronic-load baseline"),
}

MODEL_LABELS = {
    "persistence": "Persistence — exertion stays put",
    "drift":       "Drift — trailing trend",
    "ridge":       "Ridge — causal exertion/load features",
}

METRIC_META = {
    "strain":      ("Strain score", "strain_score", "strain points"),
    "strain_rate": ("Strain per hour", None, "strain/h"),
    "trimp":       ("TRIMP (HR-based)", "trimp", "TRIMP"),
    "icu_rpe":     ("RPE (your logged entries)", "icu_rpe", "RPE"),
}

DRIFT_K = 6
MIN_TRAIN = 30
MIN_BACKTESTS = 8

# Sanity caps per metric — nothing outside a physiological range enters the model.
_CAPS = {"strain": (1, 1500), "strain_rate": (1, 500),
         "trimp": (1, 1000), "icu_rpe": (1, 10)}


def metric_counts(acts: pd.DataFrame) -> dict:
    """How many observations each selectable metric actually has."""
    out = {}
    for key, (_, col, _) in METRIC_META.items():
        if key == "strain_rate":
            s = pd.to_numeric(acts.get("strain_score"), errors="coerce")
            hrs = pd.to_numeric(acts.get("moving_time"), errors="coerce") / 3600
            v = (s / hrs.replace(0, np.nan))
        else:
            v = pd.to_numeric(acts.get(col), errors="coerce")
        out[key] = int((v.notna() & (v > 0)).sum())
    return out


def _observations(acts: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One exertion value per day (last session of each day), sanity-capped."""
    _, col, _ = METRIC_META[metric]
    lo, hi = _CAPS[metric]
    df = acts[["date"]].copy()
    if metric == "strain_rate":
        s = pd.to_numeric(acts["strain_score"], errors="coerce")
        hrs = pd.to_numeric(acts["moving_time"], errors="coerce") / 3600
        df["y"] = (s / hrs.replace(0, np.nan))
    else:
        df["y"] = pd.to_numeric(acts[col], errors="coerce")
    df = df.dropna(subset=["y"])
    df = df[(df["y"] >= lo) & (df["y"] <= hi)]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    return df.reset_index(drop=True)


def _features_daily(acts: pd.DataFrame, df_all, metric: str) -> pd.DataFrame:
    ex = _observations(acts, metric).set_index("date")["y"]
    ex = ex.rename("ex_d")

    g = df_all.sort_values("date").set_index("date")
    hi = g.loc[g["is_quality"].fillna(False), "tss"]
    load = pd.DataFrame({
        "tss_d": g["tss"].resample("1D").sum(),
        "hi_d": hi.resample("1D").sum(),
        "ctl_d": g["ctl"].resample("1D").last(),
    })
    load["hi_d"] = load["hi_d"].fillna(0.0)
    load["ctl_d"] = load["ctl_d"].ffill()

    daily = load.join(ex, how="left")
    daily["ex_7"] = daily["ex_d"].rolling(7, min_periods=2).mean()
    daily["ex_28"] = daily["ex_d"].rolling(28, min_periods=4).mean()
    daily["ex_trend"] = daily["ex_7"] - daily["ex_28"]
    daily["tss_28"] = daily["tss_d"].rolling(28, min_periods=7).sum()
    hi_28 = daily["hi_d"].rolling(28, min_periods=7).sum()
    daily["hi_share_28"] = (hi_28 / daily["tss_28"].replace(0.0, np.nan)).clip(0, 1)
    daily["ctl_level"] = daily["ctl_d"]
    return daily.reset_index()


def _align(daily: pd.DataFrame, dates, first, surg) -> pd.DataFrame:
    left = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates)).dt.normalize()})
    left = left.sort_values("date").reset_index(drop=True)
    m = pd.merge_asof(left, daily, on="date", direction="backward")
    m["trend_y"] = (m["date"] - first).dt.days / 365.25
    m["post_op_d"] = (m["date"] - surg).dt.days.clip(lower=0).astype(float)
    return m


def _make_ridge() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])


def _pair_indices(dates: np.ndarray, horizon: int):
    need = dates + np.timedelta64(horizon, "D")
    ks = np.searchsorted(dates, need, side="left")
    return [(int(i), int(k)) for i, k in enumerate(ks) if k < len(dates) and k > i]


def _walk_forward(X, y, dates, pairs):
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
        errs["persistence"].append(float(y[k] - y[i]))

        lo = max(0, i - DRIFT_K + 1)
        dx = ((dates[lo:i + 1] - dates[i]) / np.timedelta64(1, "D")).astype(float)
        slope, intercept = np.polyfit(dx, y[lo:i + 1], 1)
        xk = float((dates[k] - dates[i]) / np.timedelta64(1, "D"))
        errs["drift"].append(float(y[k] - (slope * xk + intercept)))

        model = _make_ridge()
        model.fit(X[pi[:n_train]], y[pk[:n_train]])
        errs["ridge"].append(float(y[k] - model.predict(X[[i]])[0]))

    return errs


def run_exertion_forecast(acts: pd.DataFrame, df_all, horizon_days: int,
                          metric: str, surgery_date: str,
                          min_obs: int = 60) -> dict:
    """Backtest persistence/drift/ridge for exertion at horizon_days ahead."""
    if metric not in METRIC_META:
        return {"ok": False, "reason": f"Unknown metric {metric!r}."}
    _, _, unit = METRIC_META[metric]
    label = METRIC_META[metric][0]

    obs = _observations(acts, metric)
    if len(obs) < min_obs:
        extra = ""
        if metric == "icu_rpe":
            extra = (" — you have never logged RPE in intervals.icu. "
                     "Log it on a few activities and this model lights up.")
        return {"ok": False,
                "reason": f"Only {len(obs)} usable {label} observations "
                          f"(need ≥ {min_obs}){extra}."}

    daily = _features_daily(acts, df_all, metric)
    first = obs["date"].iloc[0]
    surg = pd.Timestamp(surgery_date)

    X = _align(daily, obs["date"], first, surg)[FEATURES].to_numpy(dtype=float)
    y = obs["y"].to_numpy(dtype=float)
    dates = obs["date"].to_numpy()

    pairs = _pair_indices(dates, horizon_days)
    if len(pairs) < 25:
        return {"ok": False,
                "reason": f"Only {len(pairs)} valid {horizon_days}-day pairs — "
                          f"too few to validate an exertion forecast."}

    errs = _walk_forward(X, y, dates, pairs)
    n_backtests = len(errs["persistence"])
    if n_backtests < MIN_BACKTESTS:
        return {"ok": False,
                "reason": f"Only {n_backtests} valid backtests at "
                          f"{horizon_days} days — too few to trust."}

    mae = {m: float(np.mean(np.abs(v))) for m, v in errs.items()}
    mae_p = max(mae["persistence"], 1e-9)
    skill = {m: (0.0 if m == "persistence" else 1.0 - mae[m] / mae_p)
             for m in mae}
    selected = min(mae, key=lambda m: (round(mae[m], 3), m != "persistence"))
    sel_label = MODEL_LABELS[selected]

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
        note = (f"Ridge fitted on {len(pairs)} (state → {label} "
                f"{horizon_days}d later) pairs, α = 1.0, features standardised.")
    elif selected == "drift":
        lo = max(0, len(y) - DRIFT_K)
        dx = ((dates[lo:] - dates[-1]) / np.timedelta64(1, "D")).astype(float)
        slope, intercept = np.polyfit(dx, y[lo:], 1)
        x_end = float((today + pd.Timedelta(days=horizon_days)
                       - obs["date"].iloc[-1]) / np.timedelta64(1, "D"))
        yhat = float(slope * x_end + intercept)
        note = (f"Drift: line through your last {len(y) - lo} sessions "
                f"({slope * 7:+.1f} {unit}/week), extrapolated.")
    else:
        note = (f"No model beat the flat line: at {horizon_days} days your "
                f"{label.lower()} is best predicted by it simply staying "
                f"where it is.")

    why = (f"Selected **{sel_label}** — lowest walk-forward MAE "
           f"({mae[selected]:.1f} {unit}) across {n_backtests} backtests "
           f"{horizon_days} days ahead, embargo by target date. "
           f"Doing nothing scored {mae['persistence']:.1f} {unit}.")
    band_note = (f"Band = 50%/90% quantiles of those {n_backtests} backtest "
                 f"errors, widened with √time.")

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

    mt_rows = []
    for m in ("persistence", "drift", "ridge"):
        mt_rows.append({
            "Model": MODEL_LABELS[m],
            f"MAE ({unit})": round(mae[m], 1),
            "Skill vs doing nothing": ("baseline" if m == "persistence"
                                       else f"{skill[m] * 100:+.0f}%"),
            "Verdict": "✓ selected" if m == selected else "—",
        })

    return {
        "ok": True, "obs": obs, "path": path,
        "y_now": y_now, "yhat": yhat, "endpoint": endpoint,
        "horizon_days": horizon_days, "unit": unit, "metric_label": label,
        "n_obs": int(len(obs)),
        "n_backtests": n_backtests, "mae": mae, "skill": skill,
        "selected": selected, "selected_label": sel_label,
        "model_table": pd.DataFrame(mt_rows), "coef_table": coef_table,
        "why": why, "note": note, "band_note": band_note,
    }
