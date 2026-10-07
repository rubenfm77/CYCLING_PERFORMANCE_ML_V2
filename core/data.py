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
    BLANK_TYPE_TOKENS, C, HOT_TEMP_C, IF_Z2_MAX, IF_THRESHOLD, TYPE_ALIASES,
    WEIGHT_KG,
)

# ── Threshold-effort definition — ONE definition, imported everywhere ─────────
# The single source of truth for "what counts as a threshold effort". Used by the
# auto-label rule below and by the Training page's measured-threshold chart, so the
# label a session gets and the number the chart plots always come from the same
# window. Changing these changes both, deliberately.
FTP_AUTO_MIN_S = 18 * 60     # 18 min
FTP_AUTO_MAX_S = 25 * 60     # 25 min
FTP_AUTO_FRAC = 0.95         # >= 95 % of that session's own eFTP
FTP_AUTO_SHARE = 0.15        # the window must be >= 15 % of the ride itself
#
# The share test is what separates a threshold SESSION from a threshold PUSH.
# An unlabelled 3 h ride can contain one 21-min surge at 106 % of eFTP without
# being an FTP session: 4 Oct 2026 held 21 of 189 min (11 %) and is not one,
# while 30 Sep 2026 held 22 of 114 min (19 %) and is. A window that short a
# slice of the ride says nothing about what the ride was FOR, so no type is
# invented for it.


def auto_ftp_mask(df: pd.DataFrame) -> pd.Series:
    """Rows the transparent FTP auto-label rule catches — the whole rule.

    Four conditions must hold together, and the last one is the one that keeps
    the rule honest about INTENT rather than only about power:

      1. the session carries no workout label (an athlete's own label always
         wins and is never touched here);
      2. its peak-meter window is 18–25 min long;
      3. the watts held over that window are >= 95 % of the eFTP recorded on
         that same ride (not today's — the value rolls);
      4. the window spans >= 15 % of the ride, so the ride was built around
         the effort instead of merely containing it.

    Everything a condition cannot decide is False: a missing duration, window
    or eFTP means no label is invented. Called by `load_data` before
    `training_type` is written to, so it always reads the source's own label.
    """
    _tt = (df["training_type"] if "training_type" in df.columns
           else pd.Series(np.nan, index=df.index, dtype="object"))
    _has_label = _tt.notna() & ~_tt.astype(str).str.strip().isin(
        BLANK_TYPE_TOKENS)

    def _n(col):
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype="float64")
        return pd.to_numeric(df[col], errors="coerce")

    _pm_w, _pm_s = _n("icu_pm_ftp_watts"), _n("icu_pm_ftp_secs")
    _ftp, _dur = _n("eftp"), _n("duration_s")
    return ((~_has_label)
            & _pm_s.between(FTP_AUTO_MIN_S, FTP_AUTO_MAX_S)
            & (_pm_w >= FTP_AUTO_FRAC * _ftp)
            & (_pm_s >= FTP_AUTO_SHARE * _dur))

# ── Duplicate-ride tolerances ────────────────────────────────────────────────
# Two rows are the SAME physical ride when they fall on the same calendar day, their
# durations are within DUPE_DUR_TOL_S, and EITHER their average powers are within
# DUPE_PWR_TOL_W OR their distances are within DUPE_DIST_TOL_M. Distance has to be
# an alternative rather than an extra requirement because 10 of the duplicate pairs
# carry no average power on either copy.
#
# The tolerances are what make this safe: 90 s / 3 W / 1 m are loose enough to
# survive the two encodings of one ride and tight enough that two genuinely different
# rides on one day cannot collide. Verified against the data: all 63 matched pairs
# agree to the second, and every real two-ride day (there were 68 before, 5 after)
# falls outside at least one of the windows.
DUPE_DUR_TOL_S = 90.0
DUPE_PWR_TOL_W = 3.0
DUPE_DIST_TOL_M = 1.0
DIST_COL = "distance_m"

# Every spelling the sources use for "no label": `_match_type` returns an em dash
# for an API row it could not match, and the Garmin CSV arrives with an empty cell.
# Defined ONCE, in core/theme.py, and imported above — so the label-carry step
# below, the blank test further down, and the ml/ pages can never disagree about
# what counts as unlabelled. Do not re-declare it here; a second copy here is what
# let ml/year_over_year.py keep 1043 sessions where this module keeps 1039.
# Re-exported under the same name so `from core.data import BLANK_TYPE_TOKENS`
# keeps working for existing callers.


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


def _dur_seconds(frame: pd.DataFrame) -> pd.Series | None:
    """Duration in seconds for either frame, whichever column it carries.

    `duration_s` is the canonical column but it is only built further down, after
    the CSV/API merge, so it does not exist yet when this is called. `duration_h` is
    the one both frames have at this point, and `duration_h * 3600` is exactly the
    value `duration_s` is later set to.
    """
    if "duration_s" in frame.columns:
        return pd.to_numeric(frame["duration_s"], errors="coerce")
    if "duration_h" in frame.columns:
        return pd.to_numeric(frame["duration_h"], errors="coerce") * 3600.0
    if "duration_secs" in frame.columns:
        return pd.to_numeric(frame["duration_secs"], errors="coerce")
    return None


def _carry_csv_labels(base: pd.DataFrame, recent: pd.DataFrame) -> int:
    """Move the Excel's `training_type` from the CSV row onto the API row for the
    same ride, BEFORE the CSV rows are discarded.

    Why this has to exist: the trailing 60 days are rebuilt from intervals.icu, and
    that API can never report a label — the account holds no workout documents, so
    `_match_type` returns the blank token for every one of its rows. The merge
    replaced each CSV row on a shared date with its API twin, so any session the
    athlete had labelled inside the last two months silently lost its type and
    vanished from every type-based view. That is the two months that matter most,
    because it is the present.

    Matching uses the same calendar day and the same 90 s duration window as the
    duplicate-ride rule below, so two genuinely different rides on one day never
    swap labels. Row counts are untouched: this only copies a string onto a row that
    is about to replace the CSV one anyway.
    """
    if "training_type" not in base.columns or "training_type" not in recent.columns:
        return 0

    _b_dur = _dur_seconds(base)
    _r_dur = _dur_seconds(recent)
    if _b_dur is None or _r_dur is None:
        return 0
    _b_lab = base["training_type"].astype(object)
    _b_ok = _b_lab.notna() & ~_b_lab.astype(str).str.strip().isin(BLANK_TYPE_TOKENS)
    if not _b_ok.any():
        return 0

    _b_day = base["date"].dt.normalize()
    _r_day = recent["date"].dt.normalize()
    _out = recent["training_type"].astype(object)
    carried = 0

    for _day, _grp in base[_b_ok].groupby(_b_day[_b_ok]):
        _cands = list(recent.index[_r_day == _day])
        if not _cands:
            continue
        for _i, _row in _grp.iterrows():
            _di = _b_dur.get(_i)
            for _j in _cands:
                _dj = _r_dur.get(_j)
                if pd.isna(_di) or pd.isna(_dj) or abs(_di - _dj) > DUPE_DUR_TOL_S:
                    continue
                # Only fill a genuinely unlabelled API row — never overwrite.
                if str(_out.get(_j)).strip() in BLANK_TYPE_TOKENS:
                    _out[_j] = _row["training_type"]
                    carried += 1
                _cands.remove(_j)
                break

    if carried:
        # Assign in place: rebinding a local copy here would silently leave the
        # caller's frame untouched and every label would still be lost.
        recent["training_type"] = _out
    return carried


def _apply_type_aliases(df: pd.DataFrame) -> int:
    """Rewrite alias spellings of a training type onto the agreed spelling.

    "Ronnestad" and "Billat" are the same session type in Spanish and Catalan.
    Left as separate strings they become two training types, and every type-based
    chart then shows them as series that can never be compared across the years.

    Exact string match only, and BEFORE dedup: the dedup rule below compares two
    labels to decide whether they are the same session, and comparing "RONNESTAD"
    against "BILLAT" would have it treat one ride filed in two languages as two
    different sessions. The count is recorded so the app can disclose it rather
    than quietly changing what the athlete wrote.
    """
    if "training_type" not in df.columns or not len(df):
        return 0
    lab = df["training_type"].astype(object)
    stripped = lab.astype(str).str.strip()
    hit = stripped.isin(TYPE_ALIASES)
    if not hit.any():
        return 0
    n = int(hit.sum())
    df.loc[hit, "training_type"] = stripped[hit].map(TYPE_ALIASES)
    return n


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
        # The API cannot carry a label, so hand the CSV's over before the CSV rows
        # on shared dates are dropped — otherwise every label inside the 60-day
        # window is thrown away on every reload.
        _carried_labels = _carry_csv_labels(base_df, recent)
        api_dates = set(recent["date"].dt.date.astype(str))
        base_df = base_df[~base_df["date"].dt.date.astype(str).isin(api_dates)]
        df = pd.concat([base_df, recent], ignore_index=True)
    elif base_df is not None and len(base_df) > 0:
        _carried_labels = 0
        df = base_df
    elif len(recent) > 0:
        _carried_labels = 0
        df = recent
    else:
        _carried_labels = 0
        df = pd.DataFrame()

    if df is None or len(df) == 0:
        st.error(
            "No data available. Add INTERVALS_ATHLETE_ID and INTERVALS_API_KEY "
            "to Streamlit Cloud secrets (app Settings → Secrets)."
        )
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df.attrs["labels_carried"] = _carried_labels

    # Alias spellings collapse onto the agreed label BEFORE the duplicate-ride
    # rule below compares labels, so one ride filed as "RONNESTAD" and another as
    # "BILLAT" is recognised as the same session rather than kept as two.
    df.attrs["labels_aliased"] = _aliased = _apply_type_aliases(df)
    if _aliased:
        st.caption(
            f"{_aliased:,} session label(s) written as an alias were read as "
            f"{', '.join(sorted(set(TYPE_ALIASES.values())))} "
            f"({', '.join(f'{k} -> {v}' for k, v in sorted(TYPE_ALIASES.items()))}). "
            f"They are the same session type, so they are counted together and "
            f"can be compared across years."
        )

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

    # ── Transparent FTP auto-label ─────────────────────────────────────────
    # intervals.icu only reports a training type when the workout itself carries
    # a name. Unlabelled sessions (131 of 1,164 here) therefore disappear from every
    # type-based view — including genuine threshold work. `_match_type` returns "—"
    # for those; the Garmin-sourced rows arrive as NaN.
    #
    # ONE rule, applied only where there is no label, and shown on the page so it
    # can be audited:
    #     the session contains an 18–25 min peak-meter effort at >= 95 % of that
    #     session's OWN eFTP (not today's eFTP — the value is a rolling one and
    #     using the current number on an old ride would be anachronistic) AND that
    #     window is >= 15 % of the ride itself. The first three conditions say the
    #     effort was threshold work; the last one says the RIDE was built around it.
    #     A 3-hour endurance ride with one 21-min surge inside it fails the share
    #     test and stays unlabelled instead of being typed FTP.
    #
    # Nothing is overwritten silently: `training_type_raw` keeps exactly what the
    # source said and `label_source` records how each row was decided.
    _BLANK = BLANK_TYPE_TOKENS
    _tt = df["training_type"] if "training_type" in df.columns else pd.Series(
        np.nan, index=df.index, dtype="object")
    df["training_type_raw"] = _tt
    _has_label = _tt.notna() & ~_tt.astype(str).str.strip().isin(_BLANK)
    df["label_source"] = np.where(_has_label, "workout", "unlabelled")

    def _n(col):
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype="float64")
        return pd.to_numeric(df[col], errors="coerce")

    _pm_w, _pm_s, _ftp = _n("icu_pm_ftp_watts"), _n("icu_pm_ftp_secs"), _n("eftp")
    _auto_ftp = auto_ftp_mask(df)
    df.loc[_auto_ftp, "training_type"] = "FTP"
    df.loc[_auto_ftp, "label_source"] = "auto-ftp"
    df["auto_ftp_effort_w"] = np.where(_auto_ftp, _pm_w, np.nan)
    df["auto_ftp_effort_s"] = np.where(_auto_ftp, _pm_s, np.nan)
    df["auto_ftp_threshold_w"] = np.where(_auto_ftp, FTP_AUTO_FRAC * _ftp, np.nan)

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

    # ── Drop duplicate rows: one physical ride must count once ───────────────
    # Two separate kinds of duplicate live in the merged frame, and both inflate
    # the session count and every TSS total:
    #
    #   1. BYTE-IDENTICAL ROWS — 8 of them, in 2022/2023/2025. Same date, duration,
    #      power, TSS, IF and type; no way to tell them apart.
    #   2. DOUBLE-IMPORTED RIDES — 63 of them, Feb–Jun 2026. Every one is a pair
    #      where duration matches to the second AND the ride is physically
    #      identical, but the two rows have different intervals.icu ids and
    #      different `source` (WAHOO = head unit, UPLOAD = manual upload of the
    #      same ride). TSS differs by up to 50 because only one copy carries heart
    #      rate, so the two copies are NOT byte-identical and an exact-match dedup
    #      cannot see them.
    #
    # Rule 2 is matched on physical identity rather than on ids, because the ids
    # differ: same calendar day, duration within 90 s, and EITHER avg power within
    # 3 W OR distance within 1 m. Distance is needed because 10 of the 63 pairs have
    # no avg power recorded at all on either copy — those match to the centimetre on
    # distance instead, which is an even tighter identity test than power.
    #
    # Those tolerances are what make this safe: two genuinely different rides on one
    # day (there are 68 such days and they must NOT collapse) never land inside the
    # duration window together with a power or distance match. The more complete
    # record wins, which keeps the head-unit copy and its heart rate.
    #
    # Net effect on the current data: 71 rows removed, total TSS 151,539 -> 143,334.
    # A 5.7 % overstatement that was inflating volume, hours, distance, and every
    # count on the page. The measured threshold series is unchanged — the best
    # ~20 min number in all 10 months is identical either way — but the effort COUNT
    # beside it was double (Mar 14 -> 7, May 14 -> 7, Apr 8 -> 4).
    _dropped_exact, _dropped_phys = 0, 0
    _dupe_dates: list = []
    _DEDUP_KEY = [c for c in ("date", "duration_s", "power_avg", "tss", "if_score",
                              "training_type") if c in df.columns]
    if len(_DEDUP_KEY) > 1:
        _before = len(df)
        df = df.drop_duplicates(subset=_DEDUP_KEY, keep="first")
        _dropped_exact = _before - len(df)

    _dur_n = pd.to_numeric(df.get("duration_s"), errors="coerce")
    _pwr_n = pd.to_numeric(df.get("power_avg"), errors="coerce")
    _dist_n = pd.to_numeric(df.get(DIST_COL), errors="coerce")
    _victims: list = []
    _label_donations: dict = {}
    # True where the row carries a label the athlete actually set, judged on the
    # pre-auto-label value so an inferred FTP never outranks a real one.
    _raw_lab = (df["training_type_raw"].astype(str).str.strip()
                if "training_type_raw" in df.columns
                else pd.Series("", index=df.index, dtype="object"))
    # `.notna()` is required, not optional: in this pandas `astype(str)` leaves a
    # missing cell as NaN instead of turning it into the string "nan", so
    # `.isin({"nan", ...})` alone reports a missing label as a real one — which
    # silently gave every NaN row priority in the duplicate rule below.
    _real_label = _raw_lab.notna() & ~_raw_lab.isin(BLANK_TYPE_TOKENS)
    if _dur_n is not None:
        for _day, _grp in df.groupby(df["date"].dt.normalize()):
            if len(_grp) < 2:
                continue
            _seen: set = set()
            for _i in _grp.index:
                if _i in _seen:
                    continue
                for _j in _grp.index:
                    if _j <= _i or _j in _seen:
                        continue
                    _di, _dj = _dur_n.get(_i), _dur_n.get(_j)
                    if pd.isna(_di) or pd.isna(_dj) or abs(_di - _dj) > DUPE_DUR_TOL_S:
                        continue
                    # Physical identity: same ride if the power matches OR the
                    # distance matches. Either alone is decisive; requiring both
                    # would miss the 10 pairs that carry no power at all.
                    _same_power = False
                    _pi, _pj = _pwr_n.get(_i), _pwr_n.get(_j)
                    if _pwr_n is not None and not (pd.isna(_pi) or pd.isna(_pj)):
                        _same_power = abs(_pi - _pj) <= DUPE_PWR_TOL_W
                    _same_dist = False
                    _xi, _xj = _dist_n.get(_i), _dist_n.get(_j)
                    if _dist_n is not None and not (pd.isna(_xi) or pd.isna(_xj)):
                        _same_dist = abs(_xi - _xj) <= DUPE_DIST_TOL_M
                    if not (_same_power or _same_dist):
                        continue
                    _ni = int(df.loc[_i].notna().sum())
                    _nj = int(df.loc[_j].notna().sum())
                    # Which copy survives is decided on completeness alone, exactly
                    # as before, so no TSS, heart-rate or distance number moves.
                    _survivor, _victim = (_j, _i) if _ni <= _nj else (_i, _j)
                    # The one thing completeness must not decide is the LABEL: the
                    # API can never supply one, so a pair where exactly one copy
                    # was categorised donates its label to the survivor. Pairs in
                    # Feb-May 2026 were losing their type this way — the head-unit
                    # twin carried heart rate, won on notna(), and took the label
                    # down with it. The reverse direction was just as common, so the
                    # donation has to work both ways round.
                    if _real_label[_i] != _real_label[_j]:
                        _donor = _i if _real_label[_i] else _j
                        _label_donations[_survivor] = df.at[_donor, "training_type"]
                    _victims.append(_victim)
                    _dupe_dates.append(str(pd.Timestamp(_day).date()))
                    _seen.update((_i, _j))
                    break
    if _victims:
        _dropped_phys = len(_victims)
        df = df.drop(index=_victims)
    for _row_i, _label in _label_donations.items():
        if _row_i in df.index:
            df.at[_row_i, "training_type"] = _label
            df.at[_row_i, "training_type_raw"] = _label
            df.at[_row_i, "label_source"] = "workout"
    df.attrs["labels_recovered"] = len(_label_donations)
    df = df.reset_index(drop=True)

    df.attrs["dupe_rows_dropped"] = _dropped_exact + _dropped_phys
    df.attrs["dupe_exact"] = _dropped_exact
    df.attrs["dupe_physical"] = _dropped_phys
    df.attrs["dupe_physical_dates"] = sorted(set(_dupe_dates))

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
