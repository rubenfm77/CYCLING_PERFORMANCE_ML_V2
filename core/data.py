# core/data.py — data layer for the modern dashboard (app_modern.py).
# NEW FILE: mirrors app.py's load_data()/load_wellness() logic without touching
# app.py. Same sources (local CSV/Excel base + Intervals.icu API overlay),
# same derived metrics (PMC, efficiency, quality score, Norwegian flags), but
# constants come from core.theme so FTP/weight/thresholds have ONE value each.

import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

from core.theme import (
    C, HOT_TEMP_C, IF_Z2_MAX, IF_THRESHOLD, WEIGHT_KG,
)

# ── Small helpers (same semantics as app.py) ──────────────────────────────────
def safe_sum(s) -> float:
    return pd.to_numeric(s, errors="coerce").sum()


def safe_mean(s) -> float:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return v.mean() if len(v) > 0 else 0.0


# ── Intervals.icu credentials ─────────────────────────────────────────────────
def _get_api_credentials():
    """Return (athlete_id, api_key) from st.secrets (Cloud) or .env (local)."""
    try:
        athlete_id = st.secrets["INTERVALS_ATHLETE_ID"]
        api_key = st.secrets["INTERVALS_API_KEY"].replace("API_KEY:", "").strip()
        if athlete_id and api_key:
            return athlete_id, api_key
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    athlete_id = os.environ.get("INTERVALS_ATHLETE_ID")
    api_key = os.environ.get("INTERVALS_API_KEY", "").replace("API_KEY:", "").strip()
    if athlete_id and api_key:
        return athlete_id, api_key
    return None, None


# ── API fetch (activities) ────────────────────────────────────────────────────
# Status of the last live sync — surfaced in the sidebar so the dashboard
# *shows* it is connected to intervals.icu (app.py only warned on failure).
_API_STATUS: dict = {"ok": False, "rows": 0, "newest": None, "error": None,
                     "athlete": None, "days": 60, "attempted": False}

_KNOWN_TYPES = [
    "AEROBIC BASE", "VO2MAX", "VO2 MAX", "Q-I INTERVALS", "FATMAX",
    "PIRAMIDAL", "BILLAT", "TORQUE", "TEMPO", "END", "FTP", "SST",
]
_CANDIDATE_COLS = [
    "icu_workout_name", "workout_name", "workout_doc_name",
    "pairedWorkoutName", "paired_workout_name", "icu_workout", "workout",
]


def _match_type(row) -> str:
    # Only trust real workout-label fields — the activity "name" is the creative
    # Strava title and would false-match ("around the bEND" -> END).
    for col in _CANDIDATE_COLS:
        if col in row and isinstance(row[col], str) and row[col].strip():
            hay = " " + re.sub(r"[^A-Z0-9 ]", " ", row[col].upper())
            hay = re.sub(r"\s+", " ", hay) + " "
            for t in sorted(_KNOWN_TYPES, key=len, reverse=True):
                tok = re.sub(r"[^A-Z0-9 ]", " ", t)
                tok = re.sub(r"\s+", " ", tok).strip()
                if f" {tok} " in hay:
                    return "VO2MAX" if t == "VO2 MAX" else t
    return "—"


def _fetch_from_api(days_back: int = 60) -> pd.DataFrame:
    athlete_id, api_key = _get_api_credentials()
    _API_STATUS.update({"attempted": True, "days": days_back,
                        "athlete": athlete_id, "ok": False,
                        "rows": 0, "error": None, "newest": None})
    if not athlete_id or not api_key:
        _API_STATUS["error"] = "no credentials (INTERVALS_ATHLETE_ID / INTERVALS_API_KEY)"
        return pd.DataFrame()
    date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00")
    date_to = datetime.now().strftime("%Y-%m-%dT23:59:59")
    try:
        r = requests.get(
            f"https://intervals.icu/api/v1/athlete/{athlete_id}/activities",
            auth=("API_KEY", api_key),
            params={"oldest": date_from, "newest": date_to},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        _API_STATUS["error"] = f"{type(e).__name__}: {e}"
        return pd.DataFrame()
    if not data:
        _API_STATUS["ok"] = True
        return pd.DataFrame()
    df = pd.json_normalize(data)
    rename_map = {
        "start_date_local": "date", "moving_time": "duration_secs",
        "distance": "distance_m", "total_elevation_gain": "elevation",
        "average_watts": "power_avg", "weighted_average_watts": "power_np",
        "max_watts": "power_max", "average_heartrate": "hr_avg",
        "average_cadence": "cadence", "average_temp": "temp_avg",
        "icu_training_load": "tss", "icu_intensity": "if_score",
        "icu_eftp": "eftp", "icu_fitness": "ctl", "icu_fatigue": "atl",
        "icu_average_watts": "power_avg_icu", "icu_normalized_watts": "power_np_icu",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "duration_secs" in df.columns:
        df["duration_h"] = pd.to_numeric(df["duration_secs"], errors="coerce") / 3600
    if "power_avg" not in df.columns and "power_avg_icu" in df.columns:
        df["power_avg"] = df["power_avg_icu"]
    if "power_np" not in df.columns and "power_np_icu" in df.columns:
        df["power_np"] = df["power_np_icu"]
    if "power_np" not in df.columns:
        df["power_np"] = df.get("power_avg", pd.Series([np.nan] * len(df)))
    df["weight"] = WEIGHT_KG
    df["training_type"] = df.apply(_match_type, axis=1)
    _API_STATUS.update({
        "ok": True,
        "rows": int(len(df)),
        "newest": str(df["date"].max()) if "date" in df.columns else None,
    })
    return df


# ── Main loader ───────────────────────────────────────────────────────────────
_EXCEL_RENAME = {
    "Activity Date": "date", "TRAINING_TYPE": "training_type",
    "TSS": "tss", "IF": "if_score", "PowerAverage": "power_avg",
    "Weighted Average Power": "power_np", "PowerMax": "power_max",
    "TimeTotalInHours": "duration_h", "HeartRateAverage": "hr_avg",
    "WEIGHT_KG": "weight", "Elevation Gain": "elevation",
    "DistanceInMeters": "distance_m", "Average Cadence": "cadence",
    "Average Temperature": "temp_avg", "Variability": "variability",
    "icu_pm_cp": "vo2max_power",
}


@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    base_df = None
    for p in ["data/combined_training_data.csv", "data/JOIN_STRAVA_TP.xlsx"]:
        if Path(p).exists():
            base_df = pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p).rename(columns=_EXCEL_RENAME)
            break

    recent = _fetch_from_api(days_back=60)

    if base_df is not None and len(base_df) > 0 and len(recent) > 0:
        base_df["date"] = pd.to_datetime(base_df["date"], errors="coerce")
        recent["date"] = pd.to_datetime(recent["date"], errors="coerce")
        api_dates = set(recent["date"].dt.date.astype(str))
        base_df = base_df[~base_df["date"].dt.date.astype(str).isin(api_dates)]
        df = pd.concat([base_df, recent], ignore_index=True)
    elif base_df is not None and len(base_df) > 0:
        df = base_df
    elif len(recent) > 0:
        df = recent
    else:
        df = pd.DataFrame()

    if df is None or len(df) == 0:
        st.error(
            "No data available. Add INTERVALS_ATHLETE_ID and INTERVALS_API_KEY "
            "to Streamlit Cloud secrets (app Settings → Secrets)."
        )
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ["tss", "power_avg", "hr_avg", "duration_h", "elevation",
                "if_score", "power_np", "power_max", "cadence"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df["tss"] = df["tss"].fillna(0)

    # ── Canonical duration: MOVING seconds, one definition for every consumer ──
    # `moving_time` (intervals.icu) is null on 1159/1159 rows of this athlete's
    # history, so it cannot be the source. Measured agreement across the file:
    #   duration_h*3600 == duration_secs   (exact on 192/257, median diff 0 s)
    #   duration_h*3600 ~= Moving Time     (Garmin; median +9 s, p90 66 s)
    #   duration_h*3600 <  Elapsed Time    (median -693 s — that one INCLUDES stops)
    # So duration_h is moving time and is the only field populated on every row
    # (1159/1159). `Moving Time` is the fallback for any row where it is missing.
    _dur = pd.Series(np.nan, index=df.index, dtype="float64")
    for _c in ("duration_h", "Moving Time"):
        if _c in df.columns:
            _v = pd.to_numeric(df[_c], errors="coerce")
            _dur = _dur.fillna(_v * 3600.0 if _c == "duration_h" else _v)
    df["duration_s"] = _dur

    # Some API rows return icu_intensity as a percentage (77.0) not a decimal.
    pct_mask = df["if_score"].notna() & (df["if_score"] > 2.0)
    df.loc[pct_mask, "if_score"] = df.loc[pct_mask, "if_score"] / 100

    df["temp_avg"] = pd.to_numeric(df["temp_avg"], errors="coerce") if "temp_avg" in df.columns else np.nan
    if "weight" in df.columns:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(WEIGHT_KG)
    else:
        df["weight"] = pd.Series([WEIGHT_KG] * len(df))

    # eFTP: icu_eftp is null for this athlete; prefer icu_rolling_ftp when denser.
    if "icu_rolling_ftp" in df.columns:
        _eftp = pd.to_numeric(df["icu_rolling_ftp"], errors="coerce")
        _cur = pd.to_numeric(df.get("eftp"), errors="coerce") if "eftp" in df.columns else pd.Series([np.nan] * len(df))
        if _eftp.notna().sum() > _cur.notna().sum():
            df["eftp"] = _eftp

    # VO2max estimate (ACSM leg-cycling: 10.8 × W/kg + 7) from Critical Power.
    if "vo2max_power" not in df.columns and "icu_pm_cp" in df.columns:
        df["vo2max_power"] = df["icu_pm_cp"]
    if "vo2max_power" not in df.columns:
        df["vo2max_power"] = np.nan
    df["vo2max_power"] = pd.to_numeric(df["vo2max_power"], errors="coerce")
    df["vo2max_est"] = np.where(
        df["vo2max_power"].notna(),
        10.8 * (df["vo2max_power"] / df["weight"]) + 7,
        np.nan,
    )

    df["ctl"] = df["tss"].ewm(span=42, adjust=False).mean()
    df["atl"] = df["tss"].ewm(span=7, adjust=False).mean()
    df["tsb"] = df["ctl"] - df["atl"]

    # NP (not average power) for W/kg and efficiency; fall back to power_avg
    # for API rows without NP recorded.
    _pwr = df["power_np"].fillna(df["power_avg"])
    df["w_per_kg"] = _pwr / df["weight"]
    df["efficiency"] = np.where(df["hr_avg"] > 0, _pwr / df["hr_avg"], np.nan)
    df["ftp_stimulus"] = (df["if_score"] ** 2) * df["duration_h"] * 100

    df["eff_28d"] = df["efficiency"].rolling(28, min_periods=5).mean()
    df["eff_pct_vs_28d"] = (df["efficiency"] - df["eff_28d"]) / df["eff_28d"] * 100

    # ── Session quality score (0-100): efficiency 0.5 + TSS 0.3 + IF 0.2 ──────
    df["quality_score"] = np.nan
    valid = df["efficiency"].notna() & df["eff_28d"].notna()
    if valid.sum() > 10:
        eff_std = max(float(df.loc[valid, "eff_28d"].std()), 0.01)
        tss_std = max(float(df.loc[valid, "tss"].std()), 0.01)
        if_std = max(float(df.loc[valid, "if_score"].std()), 0.01)
        eff_z = (df.loc[valid, "efficiency"] - df.loc[valid, "eff_28d"]) / eff_std
        tss_z = (df.loc[valid, "tss"] - df.loc[valid, "tss"].mean()) / tss_std
        if_z = (df.loc[valid, "if_score"] - df.loc[valid, "if_score"].mean()) / if_std
        raw = eff_z * 0.5 + tss_z * 0.3 + if_z * 0.2
        lo, hi = float(raw.min()), float(raw.max())
        if hi > lo:
            df.loc[valid, "quality_score"] = ((raw - lo) / (hi - lo) * 100).clip(0, 100)

    # ── Norwegian compliance flags ───────────────────────────────────────────
    df["is_quality"] = df["if_score"] >= IF_THRESHOLD          # 0.85
    df["is_z3_drift"] = (df["if_score"] >= IF_Z2_MAX) & (df["if_score"] < IF_THRESHOLD)
    df["is_true_z2"] = df["if_score"] < IF_Z2_MAX
    df["is_hot_session"] = df["temp_avg"].fillna(0) > HOT_TEMP_C

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M")
    df["week"] = df["date"].dt.to_period("W")

    conditions = [
        df["ctl"] < 30, df["tsb"] < -30, df["tsb"] < -10,
        df["tsb"] < 0, df["tsb"] < 10, df["tsb"] < 25,
    ]
    choices = ["Undertrained", "Overreached", "Deep Block",
               "Build Phase", "Neutral", "Fresh"]
    df["fatigue_state"] = np.select(conditions, choices, default="Peak/Detrain Risk")

    # Attach source metadata so the UI can show live-vs-stale honestly.
    df.attrs["api_status"] = dict(_API_STATUS)
    df.attrs["base_rows"] = int(len(base_df)) if base_df is not None else 0
    return df


@st.cache_data(ttl=60)
def get_sync_status() -> dict:
    """Last live-sync result (fresh copy, safe to render)."""
    return dict(_API_STATUS)


# ── Wellness loader (HRV / resting HR) ────────────────────────────────────────
_WELLNESS_RENAME = {
    "restingHR": "resting_hr", "hrv": "hrv", "hrvSDNN": "hrv_sdnn",
    "weight": "weight", "sleepSecs": "sleep_secs",
    "sleepScore": "sleep_score", "fatigue": "wellness_fatigue",
    "mood": "mood", "motivation": "motivation", "kcalConsumed": "kcal",
}


def _wellness_normalize(df: pd.DataFrame) -> pd.DataFrame:
    if "id" in df.columns:
        df["date"] = pd.to_datetime(df["id"])
    df = df.rename(columns={k: v for k, v in _WELLNESS_RENAME.items() if k in df.columns})
    if "sleep_secs" in df.columns:
        df["sleep_h"] = pd.to_numeric(df["sleep_secs"], errors="coerce") / 3600
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def load_wellness() -> pd.DataFrame:
    athlete_id, api_key = _get_api_credentials()
    p = "data/wellness_data.csv"
    if Path(p).exists():
        df = pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"])
        if not athlete_id:
            return df
        try:  # overlay last 60 days so new entries appear every TTL cycle
            date_from = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
            date_to = datetime.now().strftime("%Y-%m-%d")
            r = requests.get(
                f"https://intervals.icu/api/v1/athlete/{athlete_id}/wellness",
                auth=("API_KEY", api_key),
                params={"oldest": date_from, "newest": date_to},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            if data:
                recent = _wellness_normalize(pd.json_normalize(data))
                api_dates = set(recent["date"].dt.date.astype(str))
                df = df[~df["date"].dt.date.astype(str).isin(api_dates)]
                df = pd.concat([df, recent], ignore_index=True).sort_values("date").reset_index(drop=True)
        except Exception:
            pass
        return df
    if not athlete_id:
        return pd.DataFrame()
    try:  # no local CSV — last 365 days from API
        date_from = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        date_to = datetime.now().strftime("%Y-%m-%d")
        r = requests.get(
            f"https://intervals.icu/api/v1/athlete/{athlete_id}/wellness",
            auth=("API_KEY", api_key),
            params={"oldest": date_from, "newest": date_to},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()
        return _wellness_normalize(pd.json_normalize(data))
    except Exception:
        return pd.DataFrame()
