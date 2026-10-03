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

from ml.set_evolution import DETECTED_SRC
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


BAR_COLS = ["date", "activity_id", "_gkey", "name", "seq", "rep_idx",
            "reps_in_set", "secs", "avg_w", "np_w", "hr_avg", "intensity",
            "cad_avg", "tsb", "temp", "src"]


def _uncovered_sets(sel: pd.DataFrame, reps: pd.DataFrame) -> pd.DataFrame:
    """The selected sets that own NO raw member row.

    `_set_members` returns one row per raw WORK interval the detector
    segmented, keyed by activity × group key. A set missing from it has no
    per-interval evidence in the cache: a peak-meter reading (intervals.icu
    reports the best held window of a ride outside its interval list), or a
    set whose rows were pruned. Such a set used to fall out of the bars AND
    out of the session line without a word — which is how "FTP · 20 min"
    ended up showing one session of the four that were selected.
    """
    if not len(sel) or "_key" not in sel.columns:
        return sel.iloc[0:0]
    if reps is None or not len(reps) or "_gkey" not in reps.columns:
        return sel                      # nothing was segmented: none covered
    have = set(zip(reps["activity_id"].astype(str),
                   pd.to_datetime(reps["date"]).dt.normalize(),
                   reps["_gkey"].astype(str)))
    want = zip(sel["activity_id"].astype(str),
               pd.to_datetime(sel["date"]).dt.normalize(),
               sel["_key"].astype(str))
    return sel[np.fromiter((k not in have for k in want), dtype=bool,
                           count=len(sel))]


def _solo_bars(solo: pd.DataFrame) -> pd.DataFrame:
    """One bar for each single-effort set that has no member row.

    `reps == 1` means the set IS one held window: its watts and its length
    are the measurement itself, so exactly one bar is drawn. A set of
    several reps with no rows gives no per-interval evidence and is NOT bar
    drawn here — one bar per rep would be an invention. Its day still rides
    on the time line (see `_carry_sessions`).
    """
    empty = pd.DataFrame(columns=BAR_COLS)
    if not len(solo):
        return empty
    one = solo[pd.to_numeric(solo["reps"], errors="coerce") == 1]
    if not len(one):
        return empty
    w = pd.to_numeric(one["set_w"], errors="coerce")
    s = pd.to_numeric(one["rep_secs"], errors="coerce")
    keep = w.notna() & s.notna() & (s > 0)
    one, w, s = one[keep], w[keep], s[keep]
    if not len(one):
        return empty
    # the source travels with the set: the peak fill tags every row it adds,
    # and a frame built without the fill has no peak row in it at all, so
    # falling back to the detector label cannot mislabel anything
    src = (one["Source"] if "Source" in one.columns
           else pd.Series(DETECTED_SRC, index=one.index))
    out = pd.DataFrame({
        "date": pd.to_datetime(one["date"]).dt.normalize(),
        "activity_id": one["activity_id"].astype(str),
        "_gkey": one["_key"].astype(str),
        "name": (one["name"].fillna("").astype(str)
                 if "name" in one.columns else ""),
        "seq": np.nan,
        "rep_idx": 1,
        "reps_in_set": 1,
        "secs": s.to_numpy(),
        "avg_w": w.to_numpy(),
        "np_w": np.nan,
        "hr_avg": np.nan,
        "intensity": np.nan,
        "cad_avg": np.nan,
        "tsb": (pd.to_numeric(one["tsb"], errors="coerce").to_numpy()
                if "tsb" in one.columns else np.nan),
        "temp": (pd.to_numeric(one["temp"], errors="coerce").to_numpy()
                 if "temp" in one.columns else np.nan),
        "src": src.fillna(DETECTED_SRC).to_numpy(),
    })
    return out[BAR_COLS]


def _make_bars(reps: pd.DataFrame, solo: pd.DataFrame) -> pd.DataFrame:
    """Every interval this series can actually prove: the detector's own
    rows plus the single-effort sets it never segmented."""
    parts = []
    if reps is not None and len(reps):
        p = reps[BAR_COLS[:-1]].copy()
        p["src"] = DETECTED_SRC          # a raw row IS the detector's cut
        parts.append(p)
    s = _solo_bars(solo)
    if len(s):
        parts.append(s)
    if not parts:
        return pd.DataFrame(columns=BAR_COLS)
    bars = pd.concat(parts, ignore_index=True, sort=False)
    bars = (bars.sort_values(["date", "activity_id", "seq"],
                             na_position="last")
            .reset_index(drop=True))
    for c in ("reps_in_set", "rep_idx"):
        bars[c] = bars[c].astype(int)
    return bars


def _leftover_sets(solo: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Uncovered sets that got no bar (several reps, or no measurable watts)."""
    if not len(solo):
        return solo.iloc[0:0]
    if bars is None or not len(bars):
        return solo
    drawn = set(zip(bars["activity_id"].astype(str),
                    pd.to_datetime(bars["date"]).dt.normalize(),
                    bars["_gkey"].astype(str)))
    want = zip(solo["activity_id"].astype(str),
               pd.to_datetime(solo["date"]).dt.normalize(),
               solo["_key"].astype(str))
    return solo[np.fromiter((k not in drawn for k in want), dtype=bool,
                            count=len(solo))]


def _carry_sessions(sess: pd.DataFrame, left: pd.DataFrame) -> pd.DataFrame:
    """Put the sets that gave no bar on the session line anyway.

    A set of several reps whose member rows are gone contributes no
    per-interval evidence, so it cannot be bar drawn — but dropping its DAY
    from the time line would hide a session, which is the complaint this
    fixes. Its own average watts are known, so the day is carried at that
    average (weighted by rep count when the day also has real bars), and the
    intervals it could not be split into stay blank rather than guessed.
    One row per date, always.
    """
    if not len(left) or not len(sess):
        return sess
    out = sess.copy()
    for c in ("w", "reps", "sets", "best", "worst", "secs_mean"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    left = left.copy()
    left["_d"] = pd.to_datetime(left["date"]).dt.normalize()
    out_d = pd.to_datetime(out["date"]).dt.normalize()
    for dte, g in left.groupby("_d", sort=True):
        gw = pd.to_numeric(g["set_w"], errors="coerce")
        gr = pd.to_numeric(g["reps"], errors="coerce").fillna(1.0).clip(lower=1)
        fin = gw.notna()
        w_mean = float(gw[fin].mean()) if fin.any() else np.nan
        rs = (pd.to_numeric(g["rep_secs"], errors="coerce").mean()
              if "rep_secs" in g.columns else np.nan)
        hit = out.index[out_d == dte]
        if len(hit):
            i = hit[0]
            cur_w, cur_n = out.at[i, "w"], out.at[i, "reps"]
            add_n = float(gr[fin].sum())
            if fin.any() and np.isfinite(cur_w) and np.isfinite(cur_n) and \
                    add_n > 0:
                # the day's mean stays the mean of everything on it: the
                # real bars plus these sets weighted by their rep count
                out.at[i, "w"] = (cur_w * cur_n
                                  + float((gw[fin] * gr[fin]).sum())) \
                    / (cur_n + add_n)
                out.at[i, "reps"] = cur_n + add_n
            elif np.isfinite(w_mean):
                out.at[i, "w"] = w_mean
            out.at[i, "sets"] = out.at[i, "sets"] + len(g)
            if fin.any():
                b, wo = out.at[i, "best"], out.at[i, "worst"]
                out.at[i, "best"] = (max(b, float(gw[fin].max()))
                                     if np.isfinite(b)
                                     else float(gw[fin].max()))
                out.at[i, "worst"] = (min(wo, float(gw[fin].min()))
                                      if np.isfinite(wo)
                                      else float(gw[fin].min()))
            continue
        # a day with no bars at all: the set average is all there is, so the
        # interval count stays blank — never invented (same rule as the
        # no-member-rows fallback)
        row = {c: np.nan for c in out.columns}
        row.update({"date": dte, "reps": np.nan, "sets": float(len(g)),
                    "w": w_mean,
                    "best": float(gw[fin].max()) if fin.any() else np.nan,
                    "worst": float(gw[fin].min()) if fin.any() else np.nan,
                    "if_mean": np.nan, "if_min": np.nan, "if_max": np.nan,
                    "open_w": np.nan, "close_w": np.nan,
                    "secs_mean": rs,
                    "name": (g["name"].iloc[-1] if "name" in g.columns
                             else np.nan),
                    "temp": (pd.to_numeric(g["temp"], errors="coerce").mean()
                             if "temp" in g.columns else np.nan),
                    "tsb": (pd.to_numeric(g["tsb"], errors="coerce").mean()
                            if "tsb" in g.columns else np.nan)})
        out = pd.concat([out, pd.DataFrame([row])], ignore_index=True,
                        sort=False)
    return out.sort_values("date").reset_index(drop=True)



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
    # A set that owns no raw member row — a peak-meter reading, whose single
    # held window is not in the interval list, or a set whose rows were
    # pruned — would fall out of BOTH the bars and the session line, which is
    # how "FTP · 20 min" came to show one session of the four selected. Every
    # selected set is looked for, not only the ones the detector segmented.
    solo = _uncovered_sets(sel, reps)
    bars = _make_bars(reps, solo)
    if not len(bars):
        # Nothing could be bar drawn (cache pruned, and no single-effort
        # reading either) — fall back to the set averages so the time line
        # still works. The SET averages are all that survive, so the interval
        # count is genuinely unknown — blank, never invented.
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
        # sets that gave no bar still have a day: carry them onto the line at
        # their own average watts, never as a second row for the same date
        sess = _carry_sessions(sess, _leftover_sets(solo, bars))

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
    # robust_fit drops rows with no watts, so its band is one value PER FITTED
    # SESSION — with an unmeasurable day in the frame the lengths differ, and
    # the band is left blank rather than being shifted onto the wrong dates
    if fit["y_hat"] is not None and len(fit["y_hat"]) == len(sess):
        sess["trend"] = fit["y_hat"]
        if fit["ci_lo"] is not None and len(fit["ci_lo"]) == len(sess):
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
