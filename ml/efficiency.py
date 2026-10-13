# ml/efficiency.py — the book's aerobic-efficiency checks, in English.
#
# From the "complex systems" chapter the athlete photographed: track
# Efficiency Factor (power over heart rate) and decoupling (power:HR drift)
# on sustained work over the months, and read the present moment on the
# readiness × performance nine-quadrant matrix. What the file cannot support
# is stated, not faked:
#   * stimulus bands (light/moderate/stimulant/extreme) need FIRST-HALF vs
#     SECOND-HALF power and HR inside one session; the file carries session
#     means only, so no band is assigned;
#   * RPP (power over perceived exertion) needs per-session RPE, which this
#     file does not carry at all (icu_rpe/session_rpe are empty) — the RPP
#     table lists what it would need instead of inventing it.
import numpy as np
import pandas as pd

# Book quadrants, translated: (performance band, readiness band) ->
# (number, title, subtitle). Rows read performance top to bottom
# (rising, holding, falling); columns read readiness left to right
# (low, mid, high) — the book's own layout.
QUADRANTS = {
    ("up", "low"): (1, "Load block", "Unsustainable stimulus long-term"),
    ("up", "mid"): (2, "Chronic adaptation", "Good adaptation"),
    ("up", "high"): (3, "Tapering, race prep", "Optimal"),
    ("hold", "low"): (4, "Poor training response",
                      "Plateau, excess monotony"),
    ("hold", "mid"): (5, "Maintenance", "Baseline level, habits"),
    ("hold", "high"): (6, "Good recovery", "Good stimulus assimilation"),
    ("down", "low"): (7, "Non-functional overreaching", ""),
    ("down", "mid"): (8, "Poor stimulus fit", ""),
    ("down", "high"): (9, "Detraining, lack of stimulus", ""),
}

PERF_TOL = 2.0       # native units (W or CTL points) counting as "holding"
TSB_LOW = -10.0      # below: readiness low
TSB_HIGH = 10.0      # above: readiness high


def monthly_ef(df_all: pd.DataFrame) -> pd.DataFrame:
    """Monthly median Efficiency Factor (NP/HR) + n.

    EF lives on the session (`efficiency`); the median is over sessions with
    a heart-rate reading, and every point carries its n. Note the grain
    difference to the book: the book reads EF per INTERVAL, the file only
    supports per session, so interval-level EF drift is not claimed.
    """
    df = df_all.copy()
    if "date" not in df.columns or not len(df):
        return pd.DataFrame(columns=["month", "med_ef", "n"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    ef = pd.to_numeric(df.get("efficiency"), errors="coerce")
    df = df.assign(_ef=ef.values).dropna(subset=["_ef"])
    if not len(df):
        return pd.DataFrame(columns=["month", "med_ef", "n"])
    g = df.groupby(df["date"].dt.to_period("M"))
    out = pd.DataFrame({
        "month": [pd.Timestamp(p.start_time) for p, _ in g],
        "med_ef": [float(v.median()) for _, v in g["_ef"]],
        "n": [int(v.notna().sum()) for _, v in g["_ef"]],
    })
    return out.sort_values("month").reset_index(drop=True)


def monthly_decoupling(iv) -> pd.DataFrame:
    """Monthly median aerobic decoupling (%) on detector WORK rows + n.

    intervals.icu's own drift number, read off the rows that carry it. No
    thresholds are applied: the book's bands need within-session splits the
    file does not hold, so values are shown with n and nothing is graded.
    """
    if iv is None or not len(iv) or "decoupling" not in iv.columns \
            or "date" not in iv.columns:
        return pd.DataFrame(columns=["month", "med_dec", "n"])
    w = iv[iv["iv_type"].astype(str).str.upper() == "WORK"].copy()
    w["dec"] = pd.to_numeric(w["decoupling"], errors="coerce")
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    w = w.dropna(subset=["dec", "date"])
    if not len(w):
        return pd.DataFrame(columns=["month", "med_dec", "n"])
    g = w.groupby(w["date"].dt.to_period("M"))
    out = pd.DataFrame({
        "month": [pd.Timestamp(p.start_time) for p, _ in g],
        "med_dec": [float(v.median()) for _, v in g["dec"]],
        "n": [int(v.notna().sum()) for _, v in g["dec"]],
    })
    return out.sort_values("month").reset_index(drop=True)


def perf_delta(df_all: pd.DataFrame) -> tuple:
    """Performance change over the last 28 days: (delta, source).

    eFTP first (the book's performance axis), CTL when eFTP has fewer than
    two readings in the window. The source travels with the number so a CTL
    delta is never read as watts.
    """
    df = df_all.copy()
    if df is None or not len(df) or "date" not in df.columns:
        return (np.nan, "none")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    if not len(df):
        return (np.nan, "none")
    end = df["date"].max()
    win = df[df["date"] > end - pd.Timedelta(days=28)]
    e = pd.to_numeric(win.get("eftp"), errors="coerce").dropna()
    if len(e) >= 2:
        return (float(e.iloc[-1] - e.iloc[0]), "eFTP W")
    c = pd.to_numeric(win.get("ctl"), errors="coerce").dropna()
    if len(c) >= 2:
        return (float(c.iloc[-1] - c.iloc[0]), "CTL pts")
    return (np.nan, "none")


def readiness(tsb: float, delta: float) -> dict:
    """The book's nine quadrants for one (TSB, performance-change) pair."""
    try:
        tsb = float(tsb)
    except (TypeError, ValueError):
        tsb = np.nan
    try:
        delta = float(delta)
    except (TypeError, ValueError):
        delta = np.nan
    read = ("low" if tsb < TSB_LOW else "high" if tsb > TSB_HIGH else "mid"
            if np.isfinite(tsb) else "mid")
    perf = ("up" if delta > PERF_TOL else "down" if delta < -PERF_TOL
            else "hold" if np.isfinite(delta) else "hold")
    no, title, sub = QUADRANTS[(perf, read)]
    return {"perf": perf, "readiness": read, "no": no, "title": title,
            "sub": sub}


def rpp_status(df_all: pd.DataFrame) -> dict:
    """Why there is no RPP table: RPE coverage, stated with n.

    RPP is power over perceived exertion on a repeated standard effort. It
    needs per-session RPE; the file carries none (both RPE columns empty),
    so the honest output is the gap itself plus what would fill it.
    """
    n = len(df_all) if df_all is not None else 0
    rpe = 0
    if df_all is not None and len(df_all):
        for c in ("icu_rpe", "session_rpe"):
            if c in df_all.columns:
                rpe += int(pd.to_numeric(df_all[c],
                                         errors="coerce").notna().sum())
    return {"sessions": int(n), "rpe_n": int(rpe),
            "needs": "per-session RPE (icu_rpe/session_rpe) to divide power by"}
