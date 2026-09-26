# core/interval_data.py — interval-level data layer. (NEW FILE)
#
# Fetches per-activity interval rows from intervals.icu's interval-level API
# and caches them incrementally on disk, so the Intervals page renders in
# milliseconds after the first sync.
#
# Endpoints (read-only GETs, Basic API-key auth — same scheme as
# src/intervals_api.py, credentials via core.data._get_api_credentials):
#
#   GET /api/v1/athlete/{id}/activities?oldest=&newest=&limit=
#       session rows; `interval_summary` non-empty marks activities that
#       have detected intervals (and carries strain/trimp/RPE fields)
#   GET /api/v1/activity/{activityId}/intervals
#       {"icu_intervals": [...]} — the real per-interval rows
#       (type, duration, intensity, watts, NP, HR, zone, load, ...)
#
# Disk cache: data/interval_cache.csv — append-only by activity id.
# Already-synced activities are never re-fetched; a sync only pulls NEW
# activities (and a force sync re-pulls the current window).

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from requests.auth import HTTPBasicAuth

from core.data import _get_api_credentials

_BASE = "https://intervals.icu/api/v1"
_CACHE = Path(__file__).resolve().parent.parent / "data" / "interval_cache.csv"

# Interval rows as stored on disk / returned to views.
# group_id (v2 schema): intervals.icu's repeat-group key ("29s@323w94rpm") —
# reps sharing one belong to the same SET, which powers set-level matching.
# NEW: ss_cp_w (watts), ss_w_prime_kj (kJ), w5s_cv (fraction) —
#   ss_cp_w / ss_w_prime_kj: intervals.icu's steady-state CP model fit
#   to the interval's own power curve. Only reliable for long (>20 min),
#   variable intervals. For steady threshold reps, treat as unavailable.
#   w5s_cv: CV of 5-second rolling power within the interval. Fraction.
#   0.03 = 3% = very steady. Directly measures set reproducibility.
IV_COLUMNS = [
    "activity_id", "date", "seq", "iv_type", "label", "secs", "elapsed",
    "avg_w", "np_w", "max_w", "hr_avg", "hr_max", "cad_avg", "intensity",
    "load", "zone", "wkg", "joules", "decoupling", "distance", "group_id",
    "ss_cp_w", "ss_w_prime_kj", "w5s_cv",
]

# Session rows returned by fetch_acts().
ACT_COLUMNS = [
    "id", "date", "name", "type", "description", "tss", "if_score",
    "moving_time", "hr_avg", "strain_score", "trimp", "icu_rpe",
    "session_rpe", "feel", "race", "has_iv", "interval_summary", "temp",
]

_MISSING = "MISSING"   # cache sentinel: activity fetched, no interval rows


# ── Session-level list (one API call, cached in memory) ──────────────────────
def _auth() -> HTTPBasicAuth:
    athlete_id, key = _get_api_credentials()
    if not athlete_id:
        raise RuntimeError("intervals.icu credentials not found "
                           "(INTERVALS_ATHLETE_ID / INTERVALS_API_KEY)")
    return HTTPBasicAuth("API_KEY", key)


def _athlete_id() -> str:
    athlete_id, _ = _get_api_credentials()
    if not athlete_id:
        raise RuntimeError("INTERVALS_ATHLETE_ID not found")
    return athlete_id


def _norm_act(a: dict) -> dict:
    try:
        date = pd.Timestamp(a.get("start_date_local") or a.get("start_date"))
    except Exception:
        return {}
    return {
        "id": a.get("id"),
        "date": date,
        "name": a.get("name") or "",
        "type": a.get("type") or "",
        "description": a.get("description"),
        "tss": a.get("icu_training_load"),
        "if_score": a.get("icu_intensity"),
        "moving_time": a.get("moving_time"),
        "hr_avg": a.get("average_heartrate"),
        "strain_score": a.get("strain_score"),
        "trimp": a.get("trimp"),
        "icu_rpe": a.get("icu_rpe"),
        "session_rpe": a.get("session_rpe"),
        "feel": a.get("feel"),
        "race": bool(a.get("race")),
        "has_iv": bool(a.get("interval_summary")),
        "interval_summary": a.get("interval_summary"),
        "temp": a.get("average_temp"),
    }


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_profile() -> dict:
    """Athlete profile (DOB, sex, profile weight) — one call, cached 24 h."""
    r = requests.get(f"{_BASE}/athlete/{_athlete_id()}", auth=_auth(),
                     timeout=30)
    r.raise_for_status()
    return r.json() or {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_acts(cutoff_iso: str) -> pd.DataFrame:
    """All sessions since cutoff_iso with exertion + interval flags (1 call)."""
    cutoff = pd.Timestamp(cutoff_iso)
    today = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    r = requests.get(
        f"{_BASE}/athlete/{_athlete_id()}/activities",
        auth=_auth(),
        params={"oldest": cutoff.strftime("%Y-%m-%d"),
                "newest": today.strftime("%Y-%m-%d"),
                "limit": 5000},
        timeout=60,
    )
    r.raise_for_status()
    rows = [_norm_act(a) for a in (r.json() or [])]
    rows = [x for x in rows if x]
    df = pd.DataFrame(rows, columns=ACT_COLUMNS)
    return df.sort_values("date").reset_index(drop=True)


# ── Interval rows (per-activity calls, append-only disk cache) ───────────────
def _read_cache() -> pd.DataFrame:
    if not _CACHE.exists():
        return pd.DataFrame(columns=IV_COLUMNS)
    try:
        df = pd.read_csv(_CACHE)
    except Exception:
        return pd.DataFrame(columns=IV_COLUMNS)
    # Schema v1 caches (no group_id column) can't build set-level views —
    # drop them so the next sync re-pulls every window activity once.
    if "group_id" not in df.columns:
        return pd.DataFrame(columns=IV_COLUMNS)
    for col in IV_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df[IV_COLUMNS]


def _write_cache(df: pd.DataFrame) -> None:
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_CACHE, index=False)


def _norm_iv(activity_id: str, date, seq: int, iv: dict) -> dict:
    return {
        "activity_id": activity_id, "date": date, "seq": seq,
        "iv_type": iv.get("type"), "label": iv.get("label"),
        "secs": iv.get("moving_time"), "elapsed": iv.get("elapsed_time"),
        "avg_w": iv.get("average_watts"), "np_w": iv.get("weighted_average_watts"),
        "max_w": iv.get("max_watts"),
        "hr_avg": iv.get("average_heartrate"), "hr_max": iv.get("max_heartrate"),
        "cad_avg": iv.get("average_cadence"),
        "intensity": iv.get("intensity"), "load": iv.get("training_load"),
        "zone": iv.get("zone"),
        "wkg": iv.get("average_watts_kg"),
        "joules": iv.get("joules"), "decoupling": iv.get("decoupling"),
        "distance": iv.get("distance"), "group_id": iv.get("group_id"),
        "ss_cp_w": iv.get("ss_cp"),
        "ss_w_prime_kj": iv.get("ss_w_prime"),
        "w5s_cv": iv.get("w5s_variability"),
    }


def _fetch_iv(sess: requests.Session, activity_id: str, date):
    """Interval rows for one activity; [] if none; None if not found."""
    r = sess.get(f"{_BASE}/activity/{activity_id}/intervals", auth=_auth(),
                 timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json() or {}
    ivs = data.get("icu_intervals") or []
    return [_norm_iv(activity_id, date, i, iv) for i, iv in enumerate(ivs)]


def sync_intervals(cutoff_iso: str, force: bool = False,
                   on_progress=None) -> dict:
    """Fetch interval rows for any window activity missing from the cache.

    Append-only: already-seen activity ids are skipped. `force=True` drops
    cached rows inside the window first (re-pull). Returns sync metadata.
    """
    acts = fetch_acts(cutoff_iso)          # memory-cached by Streamlit
    targets = acts[acts["has_iv"]]
    cutoff = pd.Timestamp(cutoff_iso)

    cache = _read_cache()
    if force and len(cache):
        cache = cache[cache["date"].fillna(pd.Timestamp("1970-01-01")) < cutoff]
    seen = set(cache["activity_id"].dropna().astype(str)) if len(cache) else set()

    todo = [a for a in targets.itertuples()
            if str(a.id) not in seen]
    new_rows, missing = [], 0
    if todo and on_progress is not None:
        on_progress(0, len(todo))
    with requests.Session() as sess:
        for n, a in enumerate(todo, start=1):
            try:
                rows = _fetch_iv(sess, str(a.id), a.date)
            except requests.RequestException:
                rows = None          # transient failure: leave unseen, retry next sync
            if rows is None:
                missing += 1
                new_rows.append(_norm_missing(a.id, a.date))
            elif rows:
                new_rows.extend(rows)
            if on_progress is not None:
                on_progress(n, len(todo))

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows, columns=IV_COLUMNS)],
                          ignore_index=True)
        cache = cache.sort_values(["date", "seq"]).reset_index(drop=True)
        _write_cache(cache)

    in_window = cache[cache["date"] >= cutoff] if len(cache) else cache
    return {
        "fetched": len(todo),
        "new_rows": int(sum(1 for r in new_rows
                            if r.get("iv_type") != _MISSING)),
        "missing": missing,
        "cached_rows": int(len(in_window)),
        "window_activities": int(len(targets)),
        "cache_total": int(len(cache)),
    }


def _norm_missing(activity_id, date) -> dict:
    row = {c: None for c in IV_COLUMNS}
    row["activity_id"] = str(activity_id)
    row["date"] = date
    row["iv_type"] = _MISSING
    return row


def read_intervals(cutoff_iso: str) -> pd.DataFrame:
    """Cached interval rows inside the window (MISSING sentinels dropped)."""
    cache = _read_cache()
    if not len(cache):
        return pd.DataFrame(columns=IV_COLUMNS)
    cutoff = pd.Timestamp(cutoff_iso)
    out = cache[(cache["date"] >= cutoff) & (cache["iv_type"] != _MISSING)]
    return out.reset_index(drop=True)
