# ml/protocol_reps.py — the individual intervals inside each session. (NEW FILE)
"""
The SETS table (ml.set_evolution) stores one row per set — the average of its
reps. A coach, though, reads a workout rep by rep: "which 4 of the 6 reps
died on the last day?" This module walks back to the raw interval rows for ONE
training type × duration-class series and returns

  * bars  — every single interval of every matching session in riding order
            (watts, NP, HR, IF, cadence, seconds, TSB, temperature), so each
            session can be drawn as a row of bars;
  * sessions — one row per DAY: the mean watts of that day's intervals (the
            fair like-for-like point on a time line — within-set fatigue
            depresses the later reps, so the mean is compared, never the
            single best rep), plus rep count, first/last rep, within-set
            fade, best/worst rep, measured IF and context;
  * trend — the session means vs time, the ROBUST (Theil–Sen) fit and its
            bootstrap 95 % CI band, so one all-out day cannot masquerade as
            progress and the reader can see how wide the uncertainty is.

Honesty: bars are observed intervals — nothing smoothed, filled or invented.
The session average is the plain mean of that day's matching reps only, so
two sessions are only ever compared when they contain the same kind of work
(the type × duration class chosen by the caller).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.type_comparison import robust_fit

MIN_TREND_SESSIONS = 3        # sessions needed before a trend line is drawn
SLOT_WIDTH = 0.86             # fraction of a date slot its bars may occupy
OPEN_FROM_REP = 2             # the FIRST rep of a block carries the power ramp
OPEN_N = 2                    # opening window = reps 2–3 …
CLOSE_N = 2                   # … closing window = the last 2 reps


def _open_close(g: pd.DataFrame) -> tuple[float, float]:
    """Opening and closing watts of one set, for the within-set fade.

    The first rep of a block is deliberately skipped: intervals.icu starts the
    work interval while the power is still ramping, so rep 1 reads LOW (in the
    Ronnestad data, ~250-295 W against a 295-323 W set mean) and a
    first-rep-based \"fade\" would report the opposite of what happened. With
    ≥ 3 reps the opening window is reps 2–3 and the closing window is the
    last two; a 2-rep set falls back to rep 1 vs rep 2 (nothing else exists)."""
    w = pd.to_numeric(g["avg_w"], errors="coerce").dropna()
    n = len(w)
    if n == 0:
        return np.nan, np.nan
    if n < OPEN_FROM_REP + OPEN_N:
        return float(w.iloc[0]), float(w.iloc[-1])
    op = float(w.iloc[OPEN_FROM_REP - 1:OPEN_FROM_REP - 1 + OPEN_N].mean())
    cl = float(w.iloc[-CLOSE_N:].mean())
    return op, cl


def _set_members(iv: pd.DataFrame, sets_sel: pd.DataFrame) -> pd.DataFrame:
    """Raw WORK rows belonging to the selected sets (matched on
    activity × group key; solo sets match on their own generated key)."""
    if iv is None or not len(iv) or sets_sel is None or not len(sets_sel):
        return pd.DataFrame()
    w = iv[iv["iv_type"] == "WORK"].copy()
    if not len(w):
        return w
    for c in ("secs", "avg_w", "np_w", "hr_avg", "intensity", "load",
              "cad_avg"):
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w.dropna(subset=["secs", "avg_w"])
    if "group_id" not in w.columns:
        w["group_id"] = np.nan
    w["activity_id"] = w["activity_id"].astype(str)
    w["date"] = pd.to_datetime(w["date"]).dt.normalize()

    keyed = sets_sel.copy()
    keyed["activity_id"] = keyed["activity_id"].astype(str)
    keyed["date"] = pd.to_datetime(keyed["date"]).dt.normalize()
    if "_key" not in keyed.columns:
        keyed["_key"] = pd.Series(index=keyed.index, dtype="object")
    # session context rides along with the set (temp/TSB live on the SETS
    # table, not on the raw interval rows)
    ctx = [c for c in ("temp", "tsb", "name", "reps") if c in keyed.columns]
    keyed = (keyed[["activity_id", "_key", "date"] + ctx]
             .rename(columns={"_key": "_gkey", "reps": "set_reps"}))
    m = w.merge(keyed, on=["activity_id", "date"], how="inner")

    grp = m["group_id"].astype("string")
    gkey = m["_gkey"].astype("string")
    out = m[(grp.notna() & (grp == gkey)) | (grp.isna() & (grp == gkey))]
    if not len(out):
        return pd.DataFrame()
    out = out.drop_duplicates(subset=["activity_id", "seq", "_gkey"])
    out = out.sort_values(["date", "activity_id", "seq"])
    key_cols = ["date", "activity_id", "_gkey"]
    out["rep_idx"] = out.groupby(key_cols).cumcount() + 1
    out["reps_in_set"] = out.groupby(key_cols)["seq"].transform("size")
    return out


def run_protocol_view(iv: pd.DataFrame, sets: pd.DataFrame, family: str,
                      dur_b: float) -> dict:
    """All interval-level evidence for ONE (training type × duration) series.

    Returns bars (one row per individual interval), sessions (one row per
    session, with that session's average watts) and the trend fit."""
    sel = sets[(sets["family"] == family) &
               (np.isclose(sets["dur_b"], dur_b))].copy()
    if not len(sel):
        return {"ok": False, "reason": "No sets in that series."}

    reps = _set_members(iv, sel)
    if not len(reps):
        # raw member rows unavailable (cache pruned) — fall back to set
        # averages so the time line still works; bars are then unavailable
        bars = pd.DataFrame()
        # no member rows: the SET averages are all that survive, so the
        # interval count is genuinely unknown — blank, never invented
        sess = (sel.groupby("date")
                .agg(reps=("set_w", lambda c: np.nan),
                     sets=("set_w", "size"), w=("set_w", "mean"),
                     best=("set_w", "max"), worst=("set_w", "min"),
                     if_mean=("intensity", "mean"),
                     secs_mean=("rep_secs", "mean"),
                     name=("name", "last")).reset_index()
                .sort_values("date").reset_index(drop=True))
        sess["open_w"] = np.nan
        sess["close_w"] = np.nan
        sess["if_min"] = sess["if_mean"]
        sess["if_max"] = sess["if_mean"]
        sess["temp"] = sel.groupby("date")["temp"].mean().reindex(
            sess["date"]).to_numpy()
        sess["tsb"] = sel.groupby("date")["tsb"].mean().reindex(
            sess["date"]).to_numpy()
    else:
        bars = reps[["date", "activity_id", "_gkey", "name", "seq", "rep_idx",
                     "reps_in_set", "secs", "avg_w", "np_w", "hr_avg",
                     "intensity", "cad_avg", "tsb", "temp"]].copy()
        for c in ("reps_in_set", "rep_idx"):
            bars[c] = bars[c].astype(int)
        # per set: opening/closing windows (the within-set fade a coach
        # reads, with the ramp-affected first rep skipped), then per session
        rows = []
        for key, g in bars.groupby(["date", "activity_id", "_gkey"],
                                   sort=True):
            # one group = ONE set, however many reps are in it
            if len(g) < 2:
                # a single effort has no opening and no closing rep — the
                # fade would be 0.0 % by arithmetic, not by physiology, so it
                # is left missing instead of being reported as a clean fade
                rows.append({"date": key[0], "activity_id": key[1],
                             "open_w": np.nan, "close_w": np.nan,
                             "n_sets": 1})
                continue
            op, cl = _open_close(g)
            rows.append({"date": key[0], "activity_id": key[1],
                         "open_w": op, "close_w": cl, "n_sets": 1})
        per_set = pd.DataFrame(rows)
        sess = (bars.groupby("date")
                .agg(reps=("avg_w", "size"),
                     w=("avg_w", "mean"),
                     best=("avg_w", "max"),
                     worst=("avg_w", "min"),
                     if_mean=("intensity", "mean"),
                     if_min=("intensity", "min"),
                     if_max=("intensity", "max"),
                     secs_mean=("secs", "mean"),
                     name=("name", "last")).reset_index()
                .sort_values("date").reset_index(drop=True))
        fa = per_set.groupby("date").agg(
            open_w=("open_w", "mean"), close_w=("close_w", "mean"),
            sets=("n_sets", "sum"))
        sess = sess.merge(fa, on="date", how="left")
        sess = sess.merge(
            bars.groupby("date")[["temp", "tsb"]].mean().reset_index(),
            on="date", how="left")

    open_w = pd.to_numeric(sess["open_w"], errors="coerce")
    close_w = pd.to_numeric(sess["close_w"], errors="coerce")
    # positive = faded (finished below where it opened), negative = finished
    # above the opening window
    sess["fade_%"] = np.where(
        (open_w > 0) & close_w.notna(),
        (open_w - close_w) / open_w * 100.0, np.nan)
    sess["n_sessions"] = len(sess)
    sess["trend"] = np.nan
    sess["trend_lo"] = np.nan
    sess["trend_hi"] = np.nan

    trend = None
    fit = robust_fit(sess.rename(columns={"w": "set_w"}))
    if fit["y_hat"] is not None:
        sess["trend"] = fit["y_hat"]
        if fit["ci_lo"] is not None:
            sess["trend_lo"] = fit["ci_lo"]
            sess["trend_hi"] = fit["ci_hi"]
    if len(sess):
        trend = dict(
            n=fit["n"],
            slope_m=fit["slope_w_month"],
            slope_robust=fit["slope_w_month"],
            ci=fit["ci_w_month"],
            span_days=float((pd.to_datetime(sess["date"]).iloc[-1] -
                             pd.to_datetime(sess["date"]).iloc[0]).days),
            first=float(sess["w"].iloc[0]),
            last=float(sess["w"].iloc[-1]),
            best=float(sess["w"].max()),
            best_rep=(float(bars["avg_w"].max()) if len(bars) else None),
            spread=(float(sess["w"].max() - sess["w"].min())),
            fade_med=(float(sess["fade_%"].median())
                      if sess["fade_%"].notna().any() else np.nan),
            if_mean=(float(pd.to_numeric(sess["if_mean"],
                                         errors="coerce").mean())
                     if pd.to_numeric(sess["if_mean"],
                                      errors="coerce").notna().any() else np.nan),
        )
    return {"ok": True, "bars": bars, "sessions": sess, "trend": trend,
            "n_reps": int(len(bars)), "n_sessions": int(len(sess))}
