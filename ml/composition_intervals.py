# ml/composition_intervals.py — honest composition re-evaluation. (NEW FILE)
"""
Re-runs the old monthly composition analysis with REAL interval data and
puts both feature sets in the same time-ordered gauntlet.

Old question (src/monthly_composition_analysis.py): does the % of TSS per
training TYPE in a window predict next-window FTP?
New question: does the composition of REAL interval work — minutes by
duration band, work:rest, interval TSS share — predict next-window eFTP
better, and does either beat persistence?

Honesty rules (same as ml/ftp_forecast.py):
1. Target = Intervals.icu's own rolling eFTP observations (never derived
   from the features).
2. TIME-ORDERED VALIDATION, embargo by target date.
3. Persistence ("eFTP stays put") is the baseline; negative skill shown.
4. Coefficients of BOTH composition models are surfaced.

Caveat surfaced in the UI: windows are 28 days stepped 7 days, so backtests
overlap — treat skill as exploratory, not as independent trials.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.theme import FTP_DRIVERS, MAIN_TYPES

HORIZON_DEFAULT = 28
WINDOW_DAYS = 28
STEP_DAYS = 7
MIN_TRAIN = 10
MIN_BACKTESTS = 8

IV_BANDS = [
    ("Sprints <1 min", 0, 60),
    ("VO2 1–3 min", 60, 180),
    ("VO2/threshold 3–6 min", 180, 360),
    ("Threshold 6–12 min", 360, 720),
    ("Sweet spot 12–20 min", 720, 1200),
    ("Endurance 20 min+", 1200, 10 ** 9),
]

IV_FEATURES = ["iv_min_sprint", "iv_min_1_3", "iv_min_3_6", "iv_min_6_12",
               "iv_min_12_20", "iv_min_20p", "work_rest", "mean_intensity",
               "iv_tss_share", "n_work"]

IV_FEATURE_META = {
    "iv_min_sprint":  ("Interval min <1 min", "Neuromuscular/sprint practice"),
    "iv_min_1_3":     ("Interval min 1–3 min", "VO2max repeatability work"),
    "iv_min_3_6":     ("Interval min 3–6 min", "Classic VO2max sets"),
    "iv_min_6_12":    ("Interval min 6–12 min", "Threshold/sweet-spot sets"),
    "iv_min_12_20":   ("Interval min 12–20 min", "Long threshold work"),
    "iv_min_20p":     ("Interval min 20+", "Long endurance efforts"),
    "work_rest":      ("Work : rest (log)", "Density of the interval work"),
    "mean_intensity": ("Mean interval IF", "Average intensity of WORK efforts"),
    "iv_tss_share":   ("Interval share of TSS", "Portion of load from intervals"),
    "n_work":         ("WORK efforts", "How many efforts were done"),
}

TY_FEATURES = [f"pct_{t}" for t in MAIN_TYPES] + ["quality_pct"]

TY_FEATURE_META = {f"pct_{t}": (f"TSS share — {t}", "Old-style type composition")
                   for t in MAIN_TYPES}
TY_FEATURE_META["quality_pct"] = ("Quality TSS share", "FTP-driver types (old style)")

MODEL_LABELS = {
    "persistence":    "Persistence — eFTP stays put",
    "ridge_types":    "Ridge — old type-% composition",
    "ridge_intervals": "Ridge — real interval composition",
}


def _make_ridge() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])


# ── Windows ──────────────────────────────────────────────────────────────────
def _window_ends(df_all) -> pd.Series:
    eftp = _eftp(df_all)
    if not len(eftp):
        return pd.Series(dtype="datetime64[ns]")
    start = max(eftp["date"].min() + pd.Timedelta(days=WINDOW_DAYS),
                df_all["date"].min() + pd.Timedelta(days=WINDOW_DAYS))
    end = eftp["date"].max()
    return pd.date_range(start.normalize(), end, freq=f"{STEP_DAYS}D")


def _eftp(df_all) -> pd.DataFrame:
    e = df_all[["date", "eftp"]].copy()
    e["eftp"] = pd.to_numeric(e["eftp"], errors="coerce")
    e = e.dropna(subset=["eftp"])
    e = e[(e["eftp"] > 100) & (e["eftp"] < 700)]
    e["date"] = pd.to_datetime(e["date"]).dt.normalize()
    return e.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _iv_features_at(iv_work: pd.DataFrame, iv_rec_secs: pd.DataFrame,
                    start, end) -> dict:
    w = iv_work[(iv_work["date"] > start) & (iv_work["date"] <= end)]
    feats = {}
    secs = pd.to_numeric(w["secs"], errors="coerce").fillna(0)
    for i, (_, lo, hi) in enumerate(IV_BANDS):
        col = IV_FEATURES[i]
        m = (secs >= lo) & (secs < hi)
        feats[col] = float(secs[m].sum() / 60.0)
    work_s = float(secs.sum())
    r = iv_rec_secs[(iv_rec_secs["date"] > start) & (iv_rec_secs["date"] <= end)]
    rec_s = float(pd.to_numeric(r["secs"], errors="coerce").fillna(0).sum())
    feats["work_rest"] = float(np.log1p(work_s / max(rec_s, 1.0)))
    ints = pd.to_numeric(w["intensity"], errors="coerce").dropna()
    feats["mean_intensity"] = float(ints.mean()) if len(ints) else np.nan
    loads = pd.to_numeric(w["load"], errors="coerce").fillna(0)
    feats["_iv_tss"] = float(loads.sum())
    feats["n_work"] = int(len(w))
    return feats


def _ty_features_at(df_all, start, end) -> dict:
    w = df_all[(df_all["date"] > start) & (df_all["date"] <= end)]
    tss = pd.to_numeric(w["tss"], errors="coerce").fillna(0)
    total = float(tss.sum())
    feats = {}
    for t in MAIN_TYPES:
        share = float(tss[w["training_type"] == t].sum())
        feats[f"pct_{t}"] = (share / total) if total > 0 else 0.0
    q = float(tss[w["training_type"].isin(FTP_DRIVERS)].sum())
    feats["quality_pct"] = (q / total) if total > 0 else 0.0
    feats["_tss"] = total
    return feats


def _build_windows(df_all, iv: pd.DataFrame):
    """Windowed feature matrix + eFTP levels at each window end."""
    ends = _window_ends(df_all)
    if not len(ends):
        return None
    eftp = _eftp(df_all)

    iv = iv if iv is not None and len(iv) else pd.DataFrame(
        columns=["date", "iv_type", "secs", "intensity", "load"])
    iv = iv.copy()
    iv["date"] = pd.to_datetime(iv["date"]).dt.normalize()
    iv_work = iv[iv["iv_type"] == "WORK"] if len(iv) else iv
    iv_rec = iv[iv["iv_type"] == "RECOVERY"] if len(iv) else iv

    rows_iv, rows_ty, y_now = [], [], []
    for end in ends:
        start = end - pd.Timedelta(days=WINDOW_DAYS)
        f_iv = _iv_features_at(iv_work, iv_rec, start, end)
        share = f_iv.pop("_iv_tss")
        f_ty = _ty_features_at(df_all, start, end)
        tss_total = f_ty.pop("_tss")
        f_iv["iv_tss_share"] = (share / tss_total) if tss_total > 0 else 0.0
        rows_iv.append(f_iv)
        rows_ty.append(f_ty)

    X_iv = pd.DataFrame(rows_iv, columns=IV_FEATURES)
    X_ty = pd.DataFrame(rows_ty, columns=TY_FEATURES)
    m = pd.DataFrame({"date": ends})
    m = pd.merge_asof(m.sort_values("date"), eftp, on="date", direction="backward")
    y = m["eftp"].to_numpy(dtype=float)
    valid = ~np.isnan(y)
    return (X_iv[valid].reset_index(drop=True),
            X_ty[valid].reset_index(drop=True),
            y[valid],
            pd.DatetimeIndex(ends[valid]))


# ── Validation ───────────────────────────────────────────────────────────────
def _pair_indices(dates: np.ndarray, horizon: int):
    need = dates + np.timedelta64(horizon, "D")
    ks = np.searchsorted(dates, need, side="left")
    return [(int(i), int(k)) for i, k in enumerate(ks) if k < len(dates) and k > i]


def run_composition(df_all, iv: pd.DataFrame,
                    horizon_days: int = HORIZON_DEFAULT) -> dict:
    built = _build_windows(df_all, iv)
    if built is None:
        return {"ok": False, "reason": "No eFTP observations in the history — "
                                       "cannot score a composition model."}
    X_iv, X_ty, y, dates = built
    dates = pd.DatetimeIndex(dates).to_numpy()   # datetime64 array (harness style)
    n_windows = len(y)
    if n_windows < 20:
        return {"ok": False,
                "reason": f"Only {n_windows} evaluable windows (need ≥ 20) — "
                          f"not enough eFTP history for a composition test."}

    pairs = _pair_indices(dates, horizon_days)
    if len(pairs) < 15:
        return {"ok": False,
                "reason": f"Only {len(pairs)} valid {horizon_days}-day pairs — "
                          f"too few to compare composition models."}

    pi = np.array([p[0] for p in pairs], dtype=int)
    pk = np.array([p[1] for p in pairs], dtype=int)
    k_dates = dates[pk]
    test_start = max(int(n_windows * 0.55), 15)

    errs = {"persistence": [], "ridge_types": [], "ridge_intervals": []}
    records = []

    for i, k in pairs:
        if i < test_start:
            continue
        n_train = int(np.searchsorted(k_dates, dates[i], side="right"))
        if n_train < MIN_TRAIN:
            continue

        r_pers = float(y[k] - y[i])
        errs["persistence"].append(r_pers)

        pred = {"persistence": y[i] + r_pers}   # = y[i]
        for key, X in (("ridge_types", X_ty), ("ridge_intervals", X_iv)):
            model = _make_ridge()
            model.fit(X.to_numpy(dtype=float)[pi[:n_train]], y[pk[:n_train]])
            p = float(model.predict(X.to_numpy(dtype=float)[[i]])[0])
            errs[key].append(float(y[k] - p))
            pred[key] = p

        records.append({"Window end": dates[i], "Actual eFTP": float(y[k]),
                        "Persistence": pred["persistence"],
                        "Old type-% model": pred["ridge_types"],
                        "Interval model": pred["ridge_intervals"]})

    n_backtests = len(errs["persistence"])
    if n_backtests < MIN_BACKTESTS:
        return {"ok": False,
                "reason": f"Only {n_backtests} valid backtests at "
                          f"{horizon_days} days — too few to compare models "
                          f"honestly."}

    mae = {m: float(np.mean(np.abs(v))) for m, v in errs.items()}
    mae_p = max(mae["persistence"], 1e-9)
    skill = {m: (0.0 if m == "persistence" else 1.0 - mae[m] / mae_p)
             for m in mae}
    selected = min(mae, key=lambda m: (round(mae[m], 3), m != "persistence"))

    def _coef_table(model, features, meta):
        coefs = model.named_steps["ridge"].coef_
        rows = []
        for j in np.argsort(-np.abs(coefs)):
            name, blurb = meta[features[j]]
            c = float(coefs[j])
            rows.append({"Feature": name, "Std. weight": f"{c:+.2f}",
                         "Effect": "↑ raises forecast" if c >= 0 else "↓ lowers forecast",
                         "What it is": blurb})
        return pd.DataFrame(rows)

    # refit both ridges on ALL pairs for the coefficient tables
    m_ty = _make_ridge().fit(X_ty.to_numpy(dtype=float)[pi], y[pk])
    m_iv = _make_ridge().fit(X_iv.to_numpy(dtype=float)[pi], y[pk])

    mt_rows = []
    for mkey in ("persistence", "ridge_types", "ridge_intervals"):
        mt_rows.append({
            "Model": MODEL_LABELS[mkey],
            f"MAE (W)": round(mae[mkey], 1),
            "Skill vs doing nothing": ("baseline" if mkey == "persistence"
                                       else f"{skill[mkey] * 100:+.0f}%"),
            "Verdict": "✓ selected" if mkey == selected else "—",
        })

    iv_wins = mae["ridge_intervals"] < mae["ridge_types"]
    why = (f"Selected **{MODEL_LABELS[selected]}** — lowest walk-forward MAE "
           f"({mae[selected]:.1f} W) across {n_backtests} backtests "
           f"{horizon_days} days ahead (embargo by target date; persistence "
           f"= {mae['persistence']:.1f} W). "
           + (f"Real interval composition beat the old type-% composition "
              f"({mae['ridge_intervals']:.1f} vs {mae['ridge_types']:.1f} W)."
              if iv_wins else
              f"The old type-% composition still matches/beats the new "
              f"interval composition ({mae['ridge_types']:.1f} vs "
              f"{mae['ridge_intervals']:.1f} W)."))
    note = (f"Windows: {WINDOW_DAYS} days stepped {STEP_DAYS} days — they "
            f"overlap, so backtests are correlated. Treat skill as "
            f"exploratory, not as {n_backtests} independent trials.")

    return {
        "ok": True, "n_windows": n_windows, "n_backtests": n_backtests,
        "horizon_days": horizon_days, "mae": mae, "skill": skill,
        "selected": selected,
        "model_table": pd.DataFrame(mt_rows),
        "coef_iv": _coef_table(m_iv, IV_FEATURES, IV_FEATURE_META),
        "coef_ty": _coef_table(m_ty, TY_FEATURES, TY_FEATURE_META),
        "records": pd.DataFrame(records),
        "why": why, "note": note,
    }
