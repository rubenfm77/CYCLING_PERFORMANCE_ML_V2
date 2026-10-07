# ml/weekpattern.py — which weekly patterns preceded better fitness, honestly.
#
# Association only, never causation: for every past training week the module
# records the week's pattern (TSS, quality sessions by family, rest days),
# the conditions it started under (season, CTL band, TSB band) and what the
# fitness model did over the next 28 days (CTL change; eFTP change where the
# file carries eFTP, which is 2025 onward only). Similar weeks are matched on
# season + starting CTL/TSB bands, outcomes are medians with n attached, and
# small samples say so instead of ranking noise.
import numpy as np
import pandas as pd

from ml.plan import KEY_FAMS

# Conventional form reading off TSB. A reading, not a diagnosis — bands are
# stated so they can be argued with.
TSB_FRESH = 10.0
TSB_OK = -10.0
TSB_DEEP = -30.0

SEASON_WINDOW_M = 1      # same season: within one calendar month either way
CTL_BAND = 10.0          # starting CTL within ±10
TSB_BAND = 7.0           # starting TSB within ±7
OUTCOME_DAYS = 28
MIN_WEEKS_RANK = 3       # fewer similar weeks than this is not ranked


def weekly_frame(df_all: pd.DataFrame) -> pd.DataFrame:
    """One row per Monday-week: pattern, starting conditions, outcomes.

    Quality sessions are sessions filed under the key families (the athlete's
    own labels). Rest days = days of the week with no session. Outcomes look
    28 days ahead: CTL change always (the model runs every year), eFTP change
    only where both ends carry eFTP. Weeks too close to the file's end to
    have an outcome keep NaN there instead of a partial one.
    """
    df = df_all.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["tss"] = pd.to_numeric(df.get("tss"), errors="coerce").fillna(0.0)
    df["ctl"] = pd.to_numeric(df.get("ctl"), errors="coerce")
    df["tsb"] = pd.to_numeric(df.get("tsb"), errors="coerce")
    df["eftp"] = pd.to_numeric(df.get("eftp"), errors="coerce")
    lab = df["training_type"].astype(str).str.strip()
    df["quality"] = lab.isin(KEY_FAMS + ["VO2MAX"])
    df["wk"] = df["date"].dt.to_period("W-SUN").dt.start_time
    rows = []
    for wk, g in df.groupby("wk", sort=True):
        g = g.sort_values("date")
        fams = lab.loc[g.index]
        rows.append({
            "week": pd.Timestamp(wk).normalize(),
            "tss": float(g["tss"].sum()),
            "sessions": int(len(g)),
            "quality": int(g["quality"].sum()),
            "rest_days": int(7 - g["date"].dt.normalize().nunique()),
            "ftp_n": int((fams == "FTP").sum()),
            "vo2_n": int((fams == "VO2MAX").sum()),
            "billat_n": int((fams == "BILLAT").sum()),
            "sst_n": int((fams == "SST").sum()),
            "tempo_n": int((fams == "TEMPO").sum()),
            "month": int(pd.Timestamp(wk).month),
            "ctl_start": (float(g["ctl"].dropna().iloc[0])
                          if g["ctl"].notna().any() else np.nan),
            "tsb_start": (float(g["tsb"].dropna().iloc[0])
                          if g["tsb"].notna().any() else np.nan),
        })
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    end_ctl = df.set_index("date")["ctl"]
    end_eftp = df.set_index("date")["eftp"]
    dctl, deftp = [], []
    for wk in out["week"]:
        fut = end_ctl[(end_ctl.index > pd.Timestamp(wk) + pd.Timedelta(days=6))
                      & (end_ctl.index <= pd.Timestamp(wk)
                         + pd.Timedelta(days=6 + OUTCOME_DAYS))].dropna()
        dctl.append(float(fut.iloc[-1] - fut.iloc[0])
                    if len(fut) >= 2 else np.nan)
        fe = end_eftp[(end_eftp.index > pd.Timestamp(wk)
                       + pd.Timedelta(days=6))
                      & (end_eftp.index <= pd.Timestamp(wk)
                         + pd.Timedelta(days=6 + OUTCOME_DAYS))].dropna()
        deftp.append(float(fe.iloc[-1] - fe.iloc[0])
                     if len(fe) >= 2 else np.nan)
    out["d_ctl_28"] = dctl
    out["d_eftp_28"] = deftp
    return out.reset_index(drop=True)


def current_state(df_all: pd.DataFrame) -> dict:
    """Now, in one dict: CTL/ATL/TSB, 28-day TSS, quality share, main types."""
    df = df_all.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    last = df.iloc[-1]
    win = df[df["date"] > df["date"].max() - pd.Timedelta(days=28)]
    tss = pd.to_numeric(win.get("tss"), errors="coerce").fillna(0.0)
    lab = win["training_type"].astype(str).str.strip()
    q = lab.isin(KEY_FAMS + ["VO2MAX"])
    types = lab[q].value_counts()
    tsb = float(last["tsb"]) if pd.notna(last.get("tsb")) else np.nan
    form = ("—"
            if np.isnan(tsb) else
            "fresh" if tsb >= TSB_FRESH else
            "productive" if tsb >= TSB_OK else
            "fatigued" if tsb >= TSB_DEEP else "overreached")
    return {
        "date": pd.Timestamp(last["date"]).date(),
        "ctl": (float(last["ctl"]) if pd.notna(last.get("ctl")) else np.nan),
        "atl": (float(last["atl"]) if pd.notna(last.get("atl")) else np.nan),
        "tsb": tsb,
        "form": form,
        "tss_28": float(tss.sum()),
        "sessions_28": int(len(win)),
        "quality_share": (float(q.mean()) if len(win) else np.nan),
        "main_types": types.head(3).to_dict(),
        "month": int(pd.Timestamp(last["date"]).month),
    }


def similar_weeks(W: pd.DataFrame, month: int, ctl: float, tsb: float,
                  exclude_weeks: int = 8, season_m: int = SEASON_WINDOW_M,
                  ctl_band: float = CTL_BAND,
                  tsb_band: float = TSB_BAND) -> pd.DataFrame:
    """Past weeks under similar conditions: same season, CTL and TSB bands.

    The most recent weeks are excluded — their 28-day outcome does not exist
    yet, and ranking them would be ranking nothing.
    """
    if W is None or not len(W):
        return pd.DataFrame()
    w = W.copy()
    dm = np.abs(w["month"] - int(month))
    dm = np.minimum(dm, 12 - dm)
    m = ((dm <= int(season_m))
         & (np.abs(w["ctl_start"] - ctl) <= float(ctl_band))
         & (np.abs(w["tsb_start"] - tsb) <= float(tsb_band))
         & w["d_ctl_28"].notna())
    if int(exclude_weeks) > 0 and len(w):
        cutoff = w["week"].max() - pd.Timedelta(weeks=int(exclude_weeks))
        m = m & (w["week"] <= cutoff)
    return w[m].reset_index(drop=True)


def base_week_tss(df_all: pd.DataFrame, year: int) -> tuple:
    """Weekly TSS median and IQR of one year — the honest fallback range.

    Used only where no similar week exists to learn a range from: the
    year's own weeks, not a multiplier anyone made up.
    """
    df = df_all.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["tss"] = pd.to_numeric(df.get("tss"), errors="coerce").fillna(0.0)
    w = (df[df["date"].dt.year == int(year)].set_index("date")
         .resample("W-MON")["tss"].sum())
    w = w[w > 0]
    if not len(w):
        return (float("nan"), float("nan"))
    return (float(w.median()), float(w.quantile(0.25)),
            float(w.quantile(0.75)))


def find_similar(W: pd.DataFrame, month: int, ctl: float,
                 tsb: float) -> tuple:
    """Similar weeks, widening the bands until there is something to show.

    Tries strict, then wider, then wide — and says which one it used, because
    a "similar week" under ±20 CTL is a weaker statement than under ±10 and
    the reader must know which one they are looking at.
    """
    for sm, cb, tb, tag in [(1, CTL_BAND, TSB_BAND, "strict"),
                            (2, 15.0, 10.0, "widened"),
                            (3, 20.0, 15.0, "wide")]:
        s = similar_weeks(W, month, ctl, tsb, season_m=sm, ctl_band=cb,
                          tsb_band=tb)
        if len(s) >= 6:
            return (s, tag,
                    {"season_months": sm, "ctl_band": cb, "tsb_band": tb})
    s = similar_weeks(W, month, ctl, tsb, season_m=3, ctl_band=20.0,
                      tsb_band=15.0)
    return s, "wide", {"season_months": 3, "ctl_band": 20.0, "tsb_band": 15.0}


def rank_patterns(S: pd.DataFrame) -> pd.DataFrame:
    """Similar weeks grouped by pattern, ranked by median CTL outcome.

    A pattern is the week's shape (quality sessions, per-family counts, TSS
    third, rest days). Ranked by median next-28-day CTL change with n and the
    spread attached; patterns with fewer than MIN_WEEKS_RANK weeks are listed
    unranked instead of competing. eFTP medians ride along where the file
    carries eFTP for those weeks.
    """
    if S is None or not len(S):
        return pd.DataFrame()
    s = S.copy()
    try:
        s["tss_3"] = pd.qcut(s["tss"], 3, labels=["low", "mid", "high"],
                             duplicates="drop").astype(str)
    except (ValueError, TypeError):                          # noqa: BLE001
        s["tss_3"] = "mid"
    key = ["quality", "ftp_n", "vo2_n", "billat_n", "sst_n", "tempo_n",
           "tss_3", "rest_days"]
    rows = []
    for _, g in s.groupby(key, dropna=False):
        eftp = pd.to_numeric(g["d_eftp_28"], errors="coerce").dropna()
        rows.append({
            "quality": int(g["quality"].iloc[0]),
            "ftp_n": int(g["ftp_n"].iloc[0]),
            "vo2_n": int(g["vo2_n"].iloc[0]),
            "billat_n": int(g["billat_n"].iloc[0]),
            "sst_n": int(g["sst_n"].iloc[0]),
            "tempo_n": int(g["tempo_n"].iloc[0]),
            "tss_3": str(g["tss_3"].iloc[0]),
            "rest_days": int(g["rest_days"].iloc[0]),
            "n": int(len(g)),
            "tss_med": float(g["tss"].median()),
            "tss_lo": float(g["tss"].quantile(0.25)),
            "tss_hi": float(g["tss"].quantile(0.75)),
            "d_ctl_med": float(g["d_ctl_28"].median()),
            "d_ctl_lo": float(g["d_ctl_28"].quantile(0.25)),
            "d_ctl_hi": float(g["d_ctl_28"].quantile(0.75)),
            "d_eftp_med": (float(eftp.median()) if len(eftp) else np.nan),
            "d_eftp_n": int(len(eftp)),
        })
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    out["ranked"] = out["n"] >= MIN_WEEKS_RANK
    return out.sort_values(["ranked", "d_ctl_med"],
                           ascending=[False, False],
                           kind="stable").reset_index(drop=True)
