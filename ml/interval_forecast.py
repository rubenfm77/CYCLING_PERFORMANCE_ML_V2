# ml/interval_forecast.py — honest forecasting of best efforts in a band. (NEW FILE)
"""
Walk-forward validated forecast of "my best effort in this duration band"
for the Intervals page. Same four rules as ml/ftp_forecast.py:

1. Target = the observed rolling 28-day best of REAL interval efforts
   (intervals.icu's own detected WORK intervals) — not a formula derived
   from the features.
2. TIME-ORDERED VALIDATION ONLY, embargo by target date.
3. COMPETE AGAINST PERSISTENCE ("my best stays where it is").
4. SHOW THE COEFFICIENTS.

Series: on each day with a band effort, D(t) = max power among band efforts
in the trailing 28 days. Observations are sampled at effort days (D only
changes then). Pairs: issue at effort i, verify at first effort >= i + H.
"""

import math

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["d_level", "d_ramp_28", "n_eff_28", "iv_secs_28",
            "tss_28", "hi_share_28", "ctl_level"]

FEATURE_META = {
    "d_level":     ("Current 28-day best", "The level being forecast"),
    "d_ramp_28":   ("Best change, 4 weeks", "Improving or fading"),
    "n_eff_28":    ("Band efforts, 28 days", "How often you practise this duration"),
    "iv_secs_28":  ("Band volume, 28 days (s)", "Total seconds spent in the band"),
    "tss_28":      ("TSS, last 28 days", "Overall load absorbed"),
    "hi_share_28": ("Quality share (IF ≥ 0.85)", "Portion of recent load that was threshold+"),
    "ctl_level":   ("CTL level", "Current chronic-load baseline"),
}

MODEL_LABELS = {
    "persistence": "Persistence — best effort stays put",
    "drift":       "Drift — trailing trend",
    "ridge":       "Ridge — causal band/load features",
}

DRIFT_K = 6
MIN_TRAIN = 25
MIN_BACKTESTS = 8


def _band_efforts(iv: pd.DataFrame, lo_s: float, hi_s: float,
                  metric: str, i_lo=None, i_hi=None) -> pd.DataFrame:
    """WORK intervals inside the duration band: date, value, secs."""
    if iv is None or not len(iv):
        return pd.DataFrame(columns=["date", "value", "secs"])
    w = iv[(iv["iv_type"] == "WORK") &
           (iv["secs"] >= lo_s) & (iv["secs"] <= hi_s)].copy()
    if i_lo is not None:
        w = w[pd.to_numeric(w["intensity"], errors="coerce") >= i_lo]
    if i_hi is not None:
        w = w[pd.to_numeric(w["intensity"], errors="coerce") <= i_hi]
    w = w.dropna(subset=["date", metric])
    w = w[w[metric] > 0]
    if not len(w):
        return pd.DataFrame(columns=["date", "value", "secs"])
    w["date"] = pd.to_datetime(w["date"]).dt.normalize()
    day = w.groupby("date").agg(value=(metric, "max"), secs=("secs", "sum"))
    return day.reset_index().sort_values("date").reset_index(drop=True)


def _obs_frame(day: pd.DataFrame) -> pd.DataFrame:
    """Rolling 28-day best sampled at effort days."""
    full = pd.date_range(day["date"].min(), day["date"].max(), freq="D")
    daily = day.set_index("date")["value"].reindex(full)
    d28 = daily.rolling(28, min_periods=1).max()
    obs = d28.loc[day["date"]].dropna().reset_index()
    obs.columns = ["date", "y"]
    return obs


def _features_daily(day: pd.DataFrame, df_all) -> pd.DataFrame:
    full = pd.date_range(day["date"].min(), day["date"].max(), freq="D")
    n_eff = day.set_index("date")["secs"].reindex(full)
    frame = pd.DataFrame(index=full)
    frame["n_eff_28"] = (n_eff.notna().astype(float)
                         .rolling(28, min_periods=1).sum())
    frame["iv_secs_28"] = n_eff.fillna(0.0).rolling(28, min_periods=1).sum()
    # current 28-day best (same definition as the target series)
    daily_val = day.set_index("date")["value"].reindex(full)
    frame["d_level"] = daily_val.rolling(28, min_periods=1).max()

    g = df_all.sort_values("date").set_index("date")
    hi = g.loc[g["is_quality"].fillna(False), "tss"]
    load = pd.DataFrame({
        "tss_d": g["tss"].resample("1D").sum(),
        "hi_d": hi.resample("1D").sum(),
        "ctl_d": g["ctl"].resample("1D").last(),
    })
    load["hi_d"] = load["hi_d"].fillna(0.0)
    load["ctl_d"] = load["ctl_d"].ffill()
    frame = frame.join(load, how="left")
    frame["tss_28"] = frame["tss_d"].rolling(28, min_periods=7).sum()
    hi_28 = frame["hi_d"].rolling(28, min_periods=7).sum()
    frame["hi_share_28"] = (hi_28 / frame["tss_28"].replace(0.0, np.nan)).clip(0, 1)
    frame["ctl_level"] = frame["ctl_d"].ffill()
    return frame.reset_index().rename(columns={"index": "date"})


def _align(daily: pd.DataFrame, dates) -> pd.DataFrame:
    left = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates)).dt.normalize()})
    left = left.sort_values("date").reset_index(drop=True)
    return pd.merge_asof(left, daily, on="date", direction="backward")


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
    test_start = max(int(len(y) * 0.55), 30)
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


def run_band_forecast(iv: pd.DataFrame, df_all, lo_s: float, hi_s: float,
                      horizon_days: int, metric: str = "avg_w",
                      min_efforts: int = 15, i_lo=None, i_hi=None) -> dict:
    """Forecast the 28-day best effort inside [lo_s, hi_s] seconds."""
    unit = "W"
    day = _band_efforts(iv, lo_s, hi_s, metric, i_lo, i_hi)
    if len(day) < min_efforts:
        return {"ok": False,
                "reason": f"Only {len(day)} efforts in the "
                          f"{lo_s / 60:g}–{hi_s / 60:g} min band "
                          f"(need ≥ {min_efforts}) — widen the band or "
                          f"extend the time range."}

    obs = _obs_frame(day)
    if len(obs) < 45:
        return {"ok": False,
                "reason": f"Only {len(obs)} dated best-effort observations "
                          f"(need ≥ 45) — not enough history to validate a "
                          f"forecast."}

    daily = _features_daily(day, df_all)
    # d_ramp_28: how much the 28-day best moved over the last 4 weeks
    daily = daily.sort_values("date")
    daily["d_ramp_28"] = daily["d_level"] - daily["d_level"].shift(28)

    X = _align(daily, obs["date"])[FEATURES].to_numpy(dtype=float)
    y = obs["y"].to_numpy(dtype=float)
    dates = obs["date"].to_numpy()

    pairs = _pair_indices(dates, horizon_days)
    if len(pairs) < 25:
        return {"ok": False,
                "reason": f"Only {len(pairs)} valid {horizon_days}-day pairs "
                          f"in this band — too few to validate."}

    errs = _walk_forward(X, y, dates, pairs)
    n_backtests = len(errs["persistence"])
    if n_backtests < MIN_BACKTESTS:
        return {"ok": False,
                "reason": f"Only {n_backtests} valid backtests at "
                          f"{horizon_days} days — too few to trust a band. "
                          f"Widen the band for more data."}

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
        x_today = _align(daily, pd.Series([today]))[FEATURES]
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
        note = (f"Ridge fitted on {len(pairs)} (band state → best {horizon_days}d "
                f"later) pairs, α = 1.0, features standardised.")
    elif selected == "drift":
        lo = max(0, len(y) - DRIFT_K)
        dx = ((dates[lo:] - dates[-1]) / np.timedelta64(1, "D")).astype(float)
        slope, intercept = np.polyfit(dx, y[lo:], 1)
        x_end = float((today + pd.Timedelta(days=horizon_days)
                       - obs["date"].iloc[-1]) / np.timedelta64(1, "D"))
        yhat = float(slope * x_end + intercept)
        note = (f"Drift: line through your last {len(y) - lo} best-effort "
                f"marks ({slope * 7:+.1f} W/week), extrapolated.")
    else:
        note = (f"No model beat the flat line: over {horizon_days} days your "
                f"best {lo_s / 60:g}–{hi_s / 60:g} min effort has moved "
                f"less than model error — the honest forecast is “stays "
                f"where it is”.")

    why = (f"Selected **{sel_label}** — lowest walk-forward MAE "
           f"({mae[selected]:.1f} W) across {n_backtests} backtests "
           f"{horizon_days} days ahead, embargo by target date. "
           f"Doing nothing scored {mae['persistence']:.1f} W.")
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

    band_lbl = f"{lo_s / 60:g}–{hi_s / 60:g} min"
    return {
        "ok": True, "obs": obs, "path": path,
        "y_now": y_now, "yhat": yhat, "endpoint": endpoint,
        "horizon_days": horizon_days, "unit": unit,
        "band_label": band_lbl, "n_efforts": int(len(day)),
        "n_obs": int(len(obs)),
        "n_backtests": n_backtests, "mae": mae, "skill": skill,
        "selected": selected, "selected_label": sel_label,
        "model_table": pd.DataFrame(mt_rows), "coef_table": coef_table,
        "why": why, "note": note, "band_note": band_note,
    }
