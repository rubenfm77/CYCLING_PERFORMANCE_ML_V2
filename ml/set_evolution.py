# ml/set_evolution.py — set-level "identical efforts over time". (NEW FILE)
"""
The intervals.icu interval-search API matches single efforts by duration ×
intensity — but workouts are SETS ("29x 29s 323w" = a Ronnestad). This module
rebuilds those sets from the API's group_id field (reps sharing one group_id
belong to one set), matches sets ACROSS sessions by (rep duration × reps)
signature, and answers "do my watts go up or down on identical work — and
what co-moves with them?"

Honesty rules kept from ml/ftp_forecast.py:
  * everything here is DESCRIPTIVE (trend + association), no causal claims;
  * context correlations are reported with n, after removing the time trend,
    and only when n is large enough to say anything;
  * "fresh vs fatigued" compares observed sets, never claims TSB caused the
    difference (who rides fresh is not randomised).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The rebuild lives with the rep classes and the peak-meter columns because it
# must answer to both: it joins the detector's fragments, and it is judged
# against the meter's window. Imported, not retyped.
from ml.interval_watts import REBUILD_SRC, rebuilt_efforts  # noqa: E402

# Minimum pairwise observations before an association is shown at all.
MIN_ASSOC_N = 8
# Minimum sets per TSB side before the fresh/fatigued split is shown.
MIN_SPLIT_N = 4

SET_COLUMNS = [
    "activity_id", "date", "_key", "reps", "rep_secs", "set_w", "set_np",
    "set_hr", "intensity", "load", "cad", "decoupling", "first_seq",
    "last_seq", "rest", "name", "temp", "ctl", "atl", "tsb", "tss_7",
    "acwr", "moving_time", "act_iv_n", "act_iv_secs", "act_iv_dist",
    "act_start", "act_iv_rows",
    "set_ss_cp_w", "set_ss_w_prime_kj", "set_w5s_cv", "db_type",
]

# Duration classes for the evolution-by-duration view (same edges as the
# power curve on the Intervals page). Labels are human durations — minutes,
# with seconds only below 1 min (a 30 s rep IS a 30 s protocol).
BUCKET_EDGES = [0, 30, 60, 120, 180, 300, 480, 720, 1200, 1800, 3600, 1e9]
BUCKET_LABELS = ["0–30 s", "30–60 s", "1–2 min", "2–3 min", "3–5 min",
                 "5–8 min", "8–12 min", "12–20 min", "20–30 min",
                 "30–60 min", "60+ min"]


# ── Set construction ─────────────────────────────────────────────────────────
def _with_context(sets: pd.DataFrame, df_all=None) -> pd.DataFrame:
    """Training-state context: TSB/CTL/ATL + prior-week load from df_all.

    Merged asof-backward on the set date. Shared by `build_sets` and
    `fill_peak_efforts` so a set added from a second source carries the same
    context columns — a row that silently lacked them would fall out of every
    TSB view without anyone saying so.
    """
    if df_all is not None and len(df_all):
        g = df_all.sort_values("date").set_index("date")
        try:
            ctl = g["ctl"].resample("1D").last().ffill()
            atl = g["atl"].resample("1D").last().ffill()
            day_tss = g["tss"].resample("1D").sum()
            ctx = pd.DataFrame({"ctl": ctl, "atl": atl})
            ctx["tsb"] = ctx["ctl"] - ctx["atl"]
            ctx["tss_7"] = day_tss.rolling(7, min_periods=1).sum().shift(1)
            ctx["acwr"] = ctx["atl"] / ctx["ctl"].replace(0.0, np.nan)
            ctx = ctx.reset_index().rename(columns={"index": "date"})
            if "date" not in ctx.columns and g.index.name:
                ctx = ctx.rename(columns={g.index.name: "date"})
            sets = sets.sort_values("date")
            sets = pd.merge_asof(sets, ctx, on="date", direction="backward")
        except Exception:
            for c in ("ctl", "atl", "tsb", "tss_7", "acwr"):
                if c not in sets.columns:
                    sets[c] = np.nan
    for c in ("ctl", "atl", "tsb", "tss_7", "acwr", "temp"):
        if c not in sets.columns:
            sets[c] = np.nan
    return sets.reset_index(drop=True)


def build_sets(iv: pd.DataFrame, acts: pd.DataFrame | None = None,
               df_all=None) -> pd.DataFrame:
    """Group WORK reps into sets (group_id), with measured rest + context.

    Rest = mean seconds of RECOVERY rows between consecutive reps of the set
    — the number that separates a Ronnestad (15 s off) from a Billat (30 s).
    Context: session temperature from the activity row; TSB/CTL/ATL and
    prior-7-day load from df_all, merged asof-backward on the set date.
    """
    if iv is None or not len(iv):
        return pd.DataFrame(columns=SET_COLUMNS)
    w = iv[iv["iv_type"] == "WORK"].copy()
    for c in ("secs", "avg_w", "np_w", "hr_avg", "intensity", "load",
              "cad_avg", "decoupling"):
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w.dropna(subset=["date", "secs", "avg_w"])
    w = w[w["avg_w"] > 0]
    if not len(w):
        return pd.DataFrame(columns=SET_COLUMNS)
    w["date"] = pd.to_datetime(w["date"]).dt.normalize()

    # Sets = activity × group_id; grouped rows without a group_id fall back
    # to a solo set (still comparable via signature).
    if "group_id" not in w.columns:
        w["group_id"] = np.nan
    key = w["group_id"].astype("string")
    solo = "solo:" + w["activity_id"].astype(str) + ":" + w["seq"].astype(str)
    w["_key"] = key.where(key.notna(), solo)

    sets = (w.groupby(["activity_id", "_key"], sort=False)
            .agg(reps=("secs", "size"),
                 rep_secs=("secs", "median"),
                 set_w=("avg_w", "mean"),
                 set_np=("np_w", "mean"),
                 set_hr=("hr_avg", "mean"),
                 intensity=("intensity", "mean"),
                 load=("load", "sum"),
                 cad=("cad_avg", "mean"),
                 decoupling=("decoupling", "mean"),
                 first_seq=("seq", "min"),
                 last_seq=("seq", "max"),
                 date=("date", "first"),
                 set_ss_cp_w=("ss_cp_w", "mean"),
                 set_ss_w_prime_kj=("ss_w_prime_kj", "mean"),
                 set_w5s_cv=("w5s_cv", "mean"))
            .reset_index())

    # Measured rest between reps: median per-gap recovery time, counting
    # only RECOVERY rows whose previous AND next WORK rows belong to the
    # SAME set. That excludes pauses between different sets and lead-in
    # false reps (a group spanning a warm-up gap would otherwise report the
    # whole gap as "rest"), and the median resists mid-set stops.
    tmp = iv[["activity_id", "seq", "iv_type", "secs"]].copy()
    tmp["secs"] = pd.to_numeric(tmp["secs"], errors="coerce")
    tmp = tmp.merge(w[["activity_id", "seq", "_key"]],
                    on=["activity_id", "seq"], how="left")
    tmp = tmp.sort_values(["activity_id", "seq"], kind="stable")
    g = tmp.groupby("activity_id", sort=False)
    tmp["_prev"] = g["_key"].ffill()          # key of previous WORK row
    tmp["_next"] = g["_key"].bfill()           # key of next WORK row
    tmp["_pseq"] = tmp["seq"].where(tmp["_key"].notna()).groupby(
        tmp["activity_id"]).ffill()             # seq of previous WORK row
    rec = tmp[(tmp["iv_type"] == "RECOVERY") & tmp["_prev"].notna() &
              (tmp["_prev"] == tmp["_next"])].copy()
    if len(rec):
        per_gap = (rec.groupby(["activity_id", "_prev", "_pseq"], sort=False)
                   ["secs"].sum().reset_index())
        med = (per_gap.groupby(["activity_id", "_prev"], sort=False)["secs"]
               .median().rename("rest").reset_index()
               .rename(columns={"_prev": "_key"}))
        sets = sets.merge(med, on=["activity_id", "_key"], how="left")
    else:
        sets["rest"] = np.nan
    if "rest" not in sets.columns:
        sets["rest"] = np.nan

    # Session context (temperature, ride name) + the activity FINGERPRINT the
    # duplicate screen needs: moving_time and the raw interval totals. Two
    # activity rows on the same day with the same moving time AND the same
    # interval rows are the same ride synced twice; a per-set structural match
    # alone is NOT enough (two identical blocks in one ride, or two different
    # rides of the same workout, look identical set by set).
    sets["activity_id"] = sets["activity_id"].astype(str)
    if acts is not None and len(acts):
        a = acts.copy()
        a["activity_id"] = a["id"].astype(str)
        cols = ["activity_id"] + [c for c in ("name", "temp", "moving_time")
                                  if c in a.columns]
        sets = sets.merge(a[cols], on="activity_id", how="left")
    else:
        sets["name"] = ""
        sets["moving_time"] = np.nan
    # The athlete's OWN training type for the session each set came from.
    #
    # Merged on the activity id, never asof/backward on the date. A backward fill
    # would hand a set the label of whichever ride happened to precede it, and a
    # label that names the wrong workout is the one thing a label may not do.
    if df_all is not None and len(df_all) and "id" in df_all.columns:
        _lab = (df_all.assign(_id=df_all["id"].astype(str),
                              _lt=df_all["training_type"].astype(str).str.strip())
                .drop_duplicates("_id")[["_id", "_lt"]]
                .rename(columns={"_id": "activity_id", "_lt": "db_type"}))
        sets = sets.merge(_lab, on="activity_id", how="left")
    if "db_type" not in sets.columns:
        sets["db_type"] = ""
    else:
        sets["db_type"] = sets["db_type"].fillna("")

    # Activity START SECOND and the raw row fingerprint, taken from the cache
    # rows BEFORE the calendar normalization above. The duplicate screen needs
    # the exact second a ride began: two activity ids that start at the same
    # second and share an identical interval row are one ride synced twice —
    # this cache holds 47 such pairs (identical start, identical rows, one copy
    # carrying the athlete's own training type and the other carrying none).
    # The calendar DATE is deliberately NOT used as this proof: two different
    # rides of a double day share a date, and merging them would hand one ride
    # the other's training type, which is the one thing a label may not do.
    _st = (pd.DataFrame({"activity_id": iv["activity_id"].astype(str),
                         "_start": pd.to_datetime(iv["date"], errors="coerce")})
           .groupby("activity_id", sort=False)["_start"].min()
           .rename("act_start").reset_index())

    def _row_keys(frame: pd.DataFrame) -> pd.Series:
        """One hashable key per interval row: seconds, avg watts, NP."""
        parts = []
        for c in ("secs", "avg_w", "np_w"):
            v = pd.to_numeric(frame[c], errors="coerce").round(0)
            parts.append(v.fillna(-1).astype("int64").astype(str))
        return parts[0] + "@" + parts[1] + "@" + parts[2]

    _rw = (iv.assign(_k=_row_keys(iv))
           .groupby("activity_id", sort=False)["_k"]
           .apply(lambda s: tuple(sorted(set(s))))
           .rename("act_iv_rows").reset_index())

    w["distance"] = pd.to_numeric(w.get("distance"), errors="coerce")
    fp = (w.groupby("activity_id")
          .agg(act_iv_n=("seq", "size"),
               act_iv_secs=("secs", "sum"),
               act_iv_dist=("distance", "sum")).reset_index())
    fp["act_iv_secs"] = fp["act_iv_secs"].round(0)
    fp["act_iv_dist"] = fp["act_iv_dist"].round(0)
    sets = sets.merge(fp, on="activity_id", how="left")
    sets = sets.merge(_st, on="activity_id", how="left")
    sets = sets.merge(_rw, on="activity_id", how="left")

    # Training-state context: TSB/CTL/ATL + prior-week load from df_all.
    return _with_context(sets, df_all)


# ── The second source: Intervals.icu's peak-power reading ────────────────────
# The interval detector segments a ride by power/cadence stability, so a
# sustained block comes back in pieces: on 15 Sep 2026 the detector's longest
# row for that FTP session was 485 s, while Intervals.icu's peak-power model
# reports 231 W held over 1200 s. The Evolution page plots the peak reading,
# which is why 20-minute intervals are obvious there and invisible here.
# The rule below adds a reading ONLY where the detector did not find that
# effort, so one effort is never counted twice — and `src` on every row says
# which of the two sources reported it.
PEAK_W_COL = "icu_pm_ftp_watts"
PEAK_S_COL = "icu_pm_ftp_secs"
PEAK_MIN_S = 600.0     # 10 min — the sustained domain the detector cuts up
PEAK_NEAR_S = 120.0    # a detected set within 2 min IS this effort
PEAK_SRC = "peak meter (intervals.icu)"
DETECTED_SRC = "detected interval"


def fill_peak_efforts(sets: pd.DataFrame, df_all=None,
                      min_s: float = PEAK_MIN_S,
                      near_s: float = PEAK_NEAR_S,
                      iv: pd.DataFrame | None = None):
    """Add an effort of at least `min_s` that the detector did not report whole.

    Why this exists: the detector found no row of 10 min or longer on the
    20-minute FTP sessions of Sep 2026, so the "20 min" duration class on the
    Intervals page carried a single session while Evolution showed every one
    of them. The peak meter holds the same session's best sustained window.

    TWO sources feed what is added, and which one speaks depends on how much
    of the story the detector already told:

      * pass `iv` and the detector's own fragments are put back together first
        (see `ml.interval_watts.rebuilt_efforts`): one long block cut into
        pieces by the segmenter comes back as its whole runs, one row each —
        which is how a session that rode TWO 20-minute intervals shows two
        rows instead of the meter's single window. A rebuild only ever fires
        where the meter's window vouches for the effort;
      * where nothing could be rebuilt, the meter's own window is added
        exactly as before.

    What it refuses to do, from either source:

      * only efforts of at least `min_s` seconds — the long-effort domain
        where the detector is known to cut sustained work; short intervals
        already come through the detector and are untouched;
      * only when THAT activity has no detected set within `near_s` seconds
        of the reading (the detector found the same effort, under its own
        segmentation) and no detected set in the same whole-minute duration
        class (the two would then be the same effort filed twice);
      * never two rows for one activity from the same source — the meter
        reports one window per ride, and a rebuild that fires replaces it
        rather than sitting beside it;
      * every added row carries `Source` (`rebuilt from detector fragments`
        or `peak meter (intervals.icu)`), every existing row `Source =
        detected interval`, and the count is returned so the page can state
        it instead of hiding it.

    Returns `(sets, n_added)`.
    """
    src = (sets.copy() if sets is not None else pd.DataFrame())
    if not len(src):
        src = src if len(src.columns) else pd.DataFrame(columns=SET_COLUMNS)
    if "Source" not in src.columns:      # never overwrite an earlier tag
        src["Source"] = DETECTED_SRC

    need = {PEAK_W_COL, PEAK_S_COL, "date"}
    if df_all is None or not len(df_all) or not need.issubset(df_all.columns):
        return src, 0

    # The rebuild, keyed on the activity it came from. Without `iv` this is
    # empty and the function behaves exactly as it did before it existed.
    reb_by: dict = {}
    if iv is not None and len(iv):
        try:
            _reb = rebuilt_efforts(iv, df_all)
        except Exception:                  # a cache in the wrong shape
            _reb = None
        if _reb is not None and len(_reb):
            reb_by = {str(a): g for a, g in _reb.groupby("activity_id",
                                                         sort=False)}

    # One candidate per activity id: the peak meter reports ONE window per ride.
    cols = list(need) + [c for c in ("id", "training_type")
                         if c in df_all.columns]
    cand = df_all[cols].copy()
    cand = cand.assign(pk_w=pd.to_numeric(cand[PEAK_W_COL], errors="coerce"),
                       pk_s=pd.to_numeric(cand[PEAK_S_COL], errors="coerce"))
    cand = cand[cand["pk_w"].notna() & cand["pk_s"].notna()
                & (cand["pk_s"] >= min_s)]
    cand["date"] = pd.to_datetime(cand["date"], errors="coerce").dt.normalize()
    cand = cand[cand["date"].notna()]
    if "id" in cand.columns:
        cand = cand[cand["id"].notna()]
    if not len(cand):
        return src, 0

    # What the detector already reports, PER ACTIVITY — the checks are never
    # made across two different rides of a day.
    by_act: dict = {}
    if len(src):
        _r = pd.to_numeric(src["rep_secs"], errors="coerce")
        for a, g in zip(src["activity_id"].astype(str), _r):
            if not np.isnan(g):
                by_act.setdefault(a, []).append(float(g))
    filled = set()

    rows = []
    for r in cand.sort_values("date").itertuples(index=False):
        raw = str(getattr(r, "id", "") or "")
        act = raw
        if not act:
            # No activity id in this file: key the reading on the day itself,
            # so two days never collapse into one bucket.
            act = f"peak:{r.date:%Y%m%d}"
        secs, watts = float(r.pk_s), float(r.pk_w)
        if act in filled:
            continue
        have = by_act.get(act, [])
        # First choice: the detector's own fragments, put back together. Every
        # run must clear the SAME two guards a meter reading has to clear, so
        # whichever source speaks, one effort is still never counted twice.
        hits = reb_by.get(raw) if raw else None
        if hits is not None and len(hits):
            kept = [q for q in hits.itertuples(index=False)
                    if not any(abs(x - float(q.secs)) <= near_s for x in have)
                    and not any(_dur_label(x) == _dur_label(q.secs)
                                for x in have)]
            if kept:
                for q in kept:
                    rows.append({
                        "activity_id": act, "date": r.date,
                        "_key": f"rebuilt:{act}:{float(q.secs):.0f}s",
                        "reps": 1, "rep_secs": float(q.secs), "set_w": float(q.w),
                        "set_np": np.nan, "set_hr": np.nan, "intensity": np.nan,
                        "load": np.nan, "cad": np.nan, "decoupling": np.nan,
                        "first_seq": np.nan, "last_seq": np.nan, "rest": np.nan,
                        "name": "", "temp": np.nan,
                        "moving_time": np.nan, "act_iv_n": np.nan,
                        "act_iv_secs": np.nan, "act_iv_dist": np.nan,
                        "act_start": pd.NaT, "act_iv_rows": np.nan,
                        "set_ss_cp_w": np.nan, "set_ss_w_prime_kj": np.nan,
                        "set_w5s_cv": np.nan,
                        "db_type": str(getattr(r, "training_type",
                                               "") or "").strip(),
                        "Source": REBUILD_SRC,
                    })
                filled.add(act)
                continue                # the meter's window is now explained
        if any(abs(x - secs) <= near_s for x in have):
            continue                       # the detector found this effort
        cls = _dur_label(secs)
        if any(_dur_label(x) == cls for x in have):
            continue                       # same class, same ride: already in
        filled.add(act)
        db = str(getattr(r, "training_type", "") or "").strip()
        rows.append({
            "activity_id": act, "date": r.date,
            "_key": f"peak:{act}:{secs:.0f}s",
            "reps": 1, "rep_secs": secs, "set_w": watts,
            "set_np": np.nan, "set_hr": np.nan, "intensity": np.nan,
            "load": np.nan, "cad": np.nan, "decoupling": np.nan,
            "first_seq": np.nan, "last_seq": np.nan, "rest": np.nan,
            "name": "", "temp": np.nan,
            "moving_time": np.nan, "act_iv_n": np.nan,
            "act_iv_secs": np.nan, "act_iv_dist": np.nan,
            "act_start": pd.NaT, "act_iv_rows": np.nan,
            "set_ss_cp_w": np.nan, "set_ss_w_prime_kj": np.nan,
            "set_w5s_cv": np.nan, "db_type": db,
            "Source": PEAK_SRC,
        })
    if not rows:
        return src, 0

    add = pd.DataFrame(rows).reindex(columns=list(src.columns))
    out = pd.concat([src, add], ignore_index=True, sort=False)
    return _with_context(out, df_all), len(rows)


# ── Data-quality gates ───────────────────────────────────────────────────────
# intervals.icu's detector is good, not perfect. This cache contained exactly
# two kinds of problem worth acting on, and one field that is simply unreliable:
#   1. the same ride present twice — a named activity AND an auto-named
#      "Cycling" copy. Two forms, both decided per ACTIVITY, never per set:
#      (a) same start second + a shared identical interval row (the re-synced
#          copy: 47 pairs in this cache, segmented differently so row counts
#          and interval totals differ), (b) same day, same moving time, same
#      raw interval rows (the older screen, 4 pairs). Counting both would
#      double-count the workout, so the duplicate activity is removed — and
#      the copy that survives is the one the athlete's file labels;
#   2. efforts HOURS apart sharing one group_id — a "2 × 3 min set" with a
#      49-minute gap is two separate efforts, so the set average and the rest
#      figure are both invalid: excluded;
#   3. sub-15 s reps and IF above 150 % are a broken FIELD (intervals.icu
#      extrapolates a 20-min equivalent from very short efforts, so a 30-s rep
#      can read 120-150 % IF). The watts are still real, so these sets are
#      KEPT and FLAGGED, and their intensity band is reported as unusable
#      instead of being silently averaged into a conclusion.
# Nothing is dropped silently: every excluded set and every flag is counted,
# labelled with its reason and listed in the UI.
MAX_PROTOCOL_REST_S = 90.0      # multi-rep sets need plausible recovery …
MAX_PROTOCOL_REST_FACTOR = 1.5  # … either ≤ 90 s or ≤ 1.5 × the rep length
MIN_PROTOCOL_REP_S = 15.0       # shorter spikes: IF unreliable, watts fine
MAX_PLAUSIBLE_IF = 150.0        # > 150 % IF comes from very short rows
DEDUPE_MT_TOL_S = 60.0          # duplicate ride: same moving time ± 60 s …
DEDUPE_IV_TOL_S = 30.0          # … and same total interval seconds ± 30 s
DEDUPE_DIST_TOL_M = 300.0       # … and same interval distance ± 300 m

DUP_REASON = "duplicate ride (synced twice)"
GAP_REASON = "gaps too long — not one protocol"
SHORT_FLAG = "sub-15 s reps — IF unusable"
IF_FLAG = "IF > 150 % — model artefact on short efforts"
SHORT_BAND = "n/a — IF unreliable below 45 s"
BROKEN_IF_BAND = "n/a — IF model artefact"

QUALITY_REASONS = {
    DUP_REASON: "the same ride is in the cache twice under two activity ids: "
                "it starts at the same second and shares an identical interval "
                "row (or, when no start is available, it matches another row "
                "set on the same day, moving time and interval totals) — "
                "counting both would double-count the workout, so the copy the "
                "athlete's own file does not label is the one removed",
    GAP_REASON: "reps sit minutes-to-hours apart; one block, not an "
                "interval set — the average and the rest figure are invalid",
    SHORT_FLAG: "detector rows under 15 s: watts kept, intensity discarded",
    IF_FLAG: "IF above 150 % comes from the duration model on a very short "
             "effort: watts kept, intensity discarded",
}
REPORT_COLUMNS = ["Reason", "Action", "Sets", "Sessions", "What it means"]


def quality_gates(sets: pd.DataFrame):
    """Split sets into (clean, report, excluded) — a reason per excluded row
    and a flag per unreliable-intensity row. `report` is the small DataFrame
    (reason → action, sets, sessions) the UI prints; `excluded` lists every
    removed set; the kept sets carry `q_flag` ("" when clean)."""
    if sets is None or not len(sets):
        empty = pd.DataFrame(columns=REPORT_COLUMNS)
        return sets, empty, sets
    s = sets.copy()
    s["date"] = pd.to_datetime(s["date"]).dt.normalize()
    reps = pd.to_numeric(s["reps"], errors="coerce")
    rep_secs = pd.to_numeric(s["rep_secs"], errors="coerce")
    rest = pd.to_numeric(s.get("rest"), errors="coerce")
    inten = pd.to_numeric(s.get("intensity"), errors="coerce")
    multi = reps >= 2

    # 1 — duplicate rides, decided at ACTIVITY level, not set level
    dup_act = pd.Series(False, index=s.index)
    keep_start = set()          # copies the start-second rule decides to keep
    # 1b — the same ride under a SECOND activity id. The moving-time screen
    # below cannot see these: the re-synced copy is segmented differently, so
    # its row count and its total interval seconds differ. What it cannot
    # change is WHEN the ride began or the watts of a shared row. Rule: two
    # activity ids starting at the same second with at least one byte-identical
    # interval row are one ride synced twice, and the copy that survives is the
    # one the athlete's own file labels — so the workout keeps its training
    # type instead of falling back to a heuristic family, and the ride is
    # counted once instead of twice.
    if "act_start" in s.columns and "act_iv_rows" in s.columns:
        _lab = (s["db_type"].astype(str).str.strip() if "db_type" in s.columns
                else pd.Series("", index=s.index))
        _per = (pd.DataFrame({"activity_id": s["activity_id"].astype(str),
                              "st": pd.to_datetime(s["act_start"],
                                                   errors="coerce"),
                              "rows": s["act_iv_rows"],
                              "lab": _lab.ne("").astype(int)})
                .drop_duplicates("activity_id"))
        _dup_start = set()
        for _stv, _g in _per[_per["st"].notna()].groupby("st", sort=False):
            if len(_g) < 2:
                continue
            _g = _g.sort_values("lab", ascending=False, kind="stable")
            _anchor = _g.iloc[0]                     # the labelled copy first
            keep_start.add(_anchor["activity_id"])
            for _ in range(1, len(_g)):
                _r = _g.iloc[_]
                _rows_a, _rows_b = _anchor["rows"], _r["rows"]
                _shared = bool(_rows_a) and bool(_rows_b) and \
                    bool(set(_rows_a) & set(_rows_b))
                if _shared:
                    _dup_start.add(_r["activity_id"])
        if _dup_start:
            dup_act = dup_act | s["activity_id"].astype(str).isin(_dup_start)
    have_fp = {"moving_time", "act_iv_n", "act_iv_secs"}.issubset(s.columns)
    if have_fp:
        mt = pd.to_numeric(s["moving_time"], errors="coerce")
        n_iv = pd.to_numeric(s["act_iv_n"], errors="coerce")
        iv_s = pd.to_numeric(s["act_iv_secs"], errors="coerce")
        per_act = (pd.DataFrame({"activity_id": s["activity_id"],
                                 "day": s["date"].dt.strftime("%Y-%m-%d"),
                                 "mt": mt, "n_iv": n_iv, "iv_s": iv_s})
                   .drop_duplicates("activity_id"))
        # same day · same moving time · same interval row count · same total
        # interval seconds  →  one ride, two activity rows
        fp = (per_act["day"] + "|" +
              (per_act["mt"] / DEDUPE_MT_TOL_S).round().fillna(-1)
              .astype("int64").astype(str) + "|" +
              per_act["n_iv"].fillna(-1).astype("int64").astype(str) + "|" +
              (per_act["iv_s"] / DEDUPE_IV_TOL_S).round().fillna(-1)
              .astype("int64").astype(str))
        per_act = per_act.assign(fp=fp.values)
        per_act = per_act[per_act["mt"].notna() & per_act["n_iv"].notna()
                          & (per_act["n_iv"] > 0)]
        drop_acts = (per_act[per_act.duplicated("fp", keep="first")]
                     ["activity_id"].unique())
        # never drop a copy the start-second rule above decided to KEEP: the
        # two rules must not between them delete both copies of one ride
        if keep_start:
            drop_acts = [a for a in drop_acts if a not in keep_start]
        dup_act = dup_act | s["activity_id"].isin(drop_acts)

    # 2 — multi-rep sets whose reps are minutes-to-hours apart
    too_gappy = multi & (rest > np.maximum(MAX_PROTOCOL_REST_S,
                                           MAX_PROTOCOL_REST_FACTOR * rep_secs))

    why = pd.Series("", index=s.index, dtype="object")
    why = why.mask(dup_act, DUP_REASON)
    why = why.mask(why.eq("") & too_gappy, GAP_REASON)

    # 3 — unreliable FIELDS: flagged, never deleted
    flag = pd.Series("", index=s.index, dtype="object")
    spike = multi & (rep_secs < MIN_PROTOCOL_REP_S)
    artefact = inten > MAX_PLAUSIBLE_IF
    flag = flag.mask(spike, SHORT_FLAG)
    flag = flag.mask(flag.eq("") & artefact, IF_FLAG)

    s["q_why"] = why
    s["q_flag"] = flag
    clean = (s[s["q_why"] == ""]
             .drop(columns=["q_why"]).reset_index(drop=True))
    dropped = s[s["q_why"] != ""].reset_index(drop=True)
    kept_flagged = clean[clean["q_flag"] != ""]

    rows = []
    for reason in dropped["q_why"].value_counts().index:
        g = dropped[dropped["q_why"] == reason]
        rows.append({"Reason": reason, "Action": "excluded",
                     "Sets": len(g), "Sessions": int(g["date"].nunique()),
                     "What it means": QUALITY_REASONS.get(reason, "")})
    for reason in kept_flagged["q_flag"].value_counts().index:
        g = kept_flagged[kept_flagged["q_flag"] == reason]
        rows.append({"Reason": reason, "Action": "flagged (kept)",
                     "Sets": len(g), "Sessions": int(g["date"].nunique()),
                     "What it means": QUALITY_REASONS.get(reason, "")})
    report = (pd.DataFrame(rows, columns=REPORT_COLUMNS)
              .sort_values(["Action", "Sets"], ascending=[True, False])
              if rows else pd.DataFrame(columns=REPORT_COLUMNS))
    return clean, report.reset_index(drop=True), dropped


# ── Signature matching across sessions ───────────────────────────────────────
def _dur_label(secs: float) -> str:
    """Human duration for protocol labels: seconds under a minute, WHOLE
    minutes from a minute up (the user rule — seconds for an interval that
    lasts less than a minute, minutes for the rest, no decimals in time, and
    a 30 s interval is never "0 min"). Halves DOWN, the same class rule as
    dur_bucket, so the label and the group always agree."""
    try:
        s = float(secs)
    except (TypeError, ValueError):
        return "—"
    if s < 60:
        return f"{s:.0f} s"
    return f"{np.ceil(s / 60.0 - 0.5):.0f} min"


def _sig_label(reps_b: int, secs: float) -> str:
    """Grouping label for the identical-sets dropdown: whole minutes, no
    decimals — plus the exact rep length in seconds, because whole minutes
    alone are NOT unique here. Groups are 10-second buckets, so '3 min' is
    shared by six different groups (190 groups collapse to 93 whole-minute
    labels) and two entries in the dropdown would read identically. The
    seconds disambiguate without putting a decimal back in the minutes — and
    they are appended from ONE MINUTE up, which is where `_dur_label` switches
    to whole minutes; below a minute the label already carries the exact
    seconds, so nothing extra is needed."""
    lab = _dur_label(secs)
    if secs >= 60:
        return f"{reps_b} × {lab} · {secs:.0f} s"
    return f"{reps_b} × {lab}"


def _bucket(value: float, step: float) -> int:
    """Nearest-step bucket (half-up, not banker's rounding)."""
    return int(np.floor(float(value) / step + 0.5) * step)


def _style(rep_secs, reps, rest, db_type=None) -> str:
    """Heuristic family name from rep length + measured rest + reps.

    Raw numbers are always shown next to the label — these are protocol
    patterns, not labels typed by intervals.icu.

    The athlete's own label wins where it applies, and it applies to exactly one
    case. `_style` splits 30 s reps on MEASURED REST, calling a 16 s recovery
    "Ronnestad-style" and a 31 s recovery "Billat-style". On this athlete's file
    that produced the opposite of the truth in both directions: all six sessions
    the app called "Ronnestad-style" are sessions the athlete filed under BILLAT,
    and the two it called "Billat-style" are an AEROBIC BASE ride. So the family
    name was contradicting the athlete's own categorisation, and the BILLAT
    sessions could never sit on one BILLAT series to be compared across years.

    So a 30 s set from a session the athlete labelled BILLAT is BILLAT. The
    measured rest is still computed and still shown - it is a real measurement of
    a real difference - but it no longer gets to rename the workout. Sessions the
    athlete did NOT label BILLAT keep the protocol name, because there the
    heuristic is the only description there is.
    """
    try:
        rep_secs = float(rep_secs)
        reps = int(reps)
    except (TypeError, ValueError):
        return ""
    rest = float(rest) if rest is not None and not np.isnan(rest) else np.nan
    if 24 <= rep_secs <= 40 and reps >= 8:
        # Merge Ronnestad-style into BILLAT. Keep Billat-style only when the
        # athlete did not label it as BILLAT and rest is ~30s.
        if db_type is not None and str(db_type).strip() == "BILLAT":
            return "BILLAT"
        if not np.isnan(rest):
            if 10 <= rest <= 22 and reps >= 15:
                return "BILLAT"  # was Ronnestad-style; merge to BILLAT
            if 23 <= rest <= 45:
                return "Billat-style (30 s on / 30 s off)"
        return "micro-reps"
    if 240 <= rep_secs <= 420 and 2 <= reps <= 8:
        return "FTP / threshold sets"
    if 140 <= rep_secs < 240 and 2 <= reps <= 8:
        return "VO₂ max sets"
    if rep_secs < 15 and reps <= 6:
        return "sprints"
    if rep_secs >= 600:
        return "long efforts"
    return ""


def add_signatures(sets: pd.DataFrame):
    """Add sig/style columns; return (sets, cluster table sorted by usage)."""
    s = sets.copy()
    if not len(s):
        return s, pd.DataFrame(columns=["sig", "n_sets", "n_days", "rep_secs",
                                        "reps", "rest", "best", "style",
                                        "first", "last"])
    rep = s["rep_secs"].astype(float)
    reps = s["reps"].astype(int)
    rep_b = [_bucket(r, 5 if r <= 90 else 10) for r in rep]
    reps_b = [r if r <= 10 else _bucket(r, 5) for r in reps]
    s["rep_b"] = rep_b
    s["reps_b"] = reps_b
    s["sig"] = [_sig_label(nb, rb) for rb, nb in zip(rep_b, reps_b)]
    _db = s["db_type"] if "db_type" in s.columns else [""] * len(s)
    s["style"] = [_style(r, n, t, d)
                  for r, n, t, d in zip(s["rep_secs"], s["reps"], s["rest"],
                                        _db)]

    clusters = (s.groupby("sig", sort=False)
                .agg(n_sets=("set_w", "size"),
                     n_days=("date", "nunique"),
                     rep_secs=("rep_secs", "median"),
                     reps=("reps", "median"),
                     rest=("rest", "median"),
                     best=("set_w", "max"),
                     first=("date", "min"),
                     last=("date", "max"))
                .reset_index())
    mode = s.groupby("sig", sort=False)["style"].agg(
        lambda x: x.value_counts().idxmax() if len(x) else "")
    clusters["style"] = clusters["sig"].map(mode)
    # Multi-rep SETS first (the workouts: Ronnestad/Billat/FTP), singles after.
    clusters["_single"] = (clusters["reps"] <= 1).astype(int)
    clusters = clusters.sort_values(["_single", "n_days", "n_sets"],
                                    ascending=[True, False, False])
    clusters = clusters.drop(columns="_single").reset_index(drop=True)
    return s, clusters


# ── Trend + context for one matched signature ────────────────────────────────
def run_set_evolution(s: pd.DataFrame) -> dict:
    """Descriptive trend + context associations for one matched set type."""
    s = s.sort_values("date").reset_index(drop=True)
    n = len(s)
    n_days = int(s["date"].nunique())
    if n < 3:
        return {"ok": False,
                "reason": f"Only {n} sets match this signature — a trend "
                          f"needs at least 3. Pick a more common set type."}

    t = (s["date"] - s["date"].iloc[0]).dt.days.astype(float).to_numpy()
    y = s["set_w"].astype(float).to_numpy()
    span_days = float(t[-1] - t[0])

    if np.ptp(t) > 0 and np.ptp(y) > 0:
        slope_d, icept = float(np.polyfit(t, y, 1)[0]), float(
            np.polyfit(t, y, 1)[1])
        r_time = float(np.corrcoef(t, y)[0, 1])
    else:
        slope_d, icept = 0.0, float(np.mean(y))
        r_time = float("nan")
    slope_m = slope_d * 30.44                      # W per month
    resid = y - (slope_d * t + icept)
    rho = float(pd.DataFrame({"t": t, "y": y}).corr(method="spearman")
                .iloc[0, 1]) if np.ptp(t) > 0 and np.ptp(y) > 0 else float("nan")

    trend = pd.DataFrame({"date": s["date"],
                          "y_hat": slope_d * t + icept})

    # Context associations with the detrended power residual.
    ctx_vars = [
        ("temp", "Session temperature (°C)"),
        ("tsb", "TSB — freshness before the ride"),
        ("ctl", "CTL — chronic load (fitness)"),
        ("acwr", "Acute:chronic load ratio"),
        ("tss_7", "TSS in the prior 7 days"),
        ("set_hr", "Set average HR (bpm)"),
        ("decoupling", "Set HR decoupling (%)"),
        ("set_ss_cp_w", "Set CP estimate (W)"),
        ("set_ss_w_prime_kj", "Set W' estimate (kJ)"),
        ("set_w5s_cv", "Set w5s CV (fraction)"),
    ]
    rows = []
    for col, label in ctx_vars:
        if col not in s.columns:
            continue
        pair = pd.DataFrame({"r": resid,
                             "v": pd.to_numeric(s[col], errors="coerce")}
                            ).dropna()
        if len(pair) < MIN_ASSOC_N or pair["v"].nunique() < 3:
            continue
        r = float(pair.corr().iloc[0, 1])
        if np.isnan(r):
            continue
        rows.append({"Context": label, "r": round(r, 2),
                     "n": int(len(pair)),
                     "Moves with power": ("higher together" if r > 0
                                          else "opposite directions")})
    assoc = (pd.DataFrame(rows).sort_values("r", key=lambda c: c.abs(),
                                            ascending=False).reset_index(
        drop=True)
        if rows else pd.DataFrame(
        columns=["Context", "r", "n", "Moves with power"]))

    # Fresh vs fatigued split (descriptive, observed sets only).
    fatigue = None
    if "tsb" in s.columns:
        tsb = pd.to_numeric(s["tsb"], errors="coerce")
        fresh = y[(tsb >= 0).to_numpy()]
        tired = y[(tsb <= -10).to_numpy()]
        if len(fresh) >= MIN_SPLIT_N and len(tired) >= MIN_SPLIT_N:
            fatigue = {
                "fresh_n": int(len(fresh)), "fresh_med": float(np.median(fresh)),
                "tired_n": int(len(tired)),
                "tired_med": float(np.median(tired)),
                "fresh_vals": [float(v) for v in fresh],
                "tired_vals": [float(v) for v in tired],
            }

    # Recent vs previous best of the SAME signature (last 28 days).
    now = pd.Timestamp.now().normalize()
    recent = s[s["date"] >= now - pd.Timedelta(days=28)]
    before = s[s["date"] < now - pd.Timedelta(days=28)]
    d_best = (float(recent["set_w"].max()) - float(before["set_w"].max())
              ) if len(recent) and len(before) else None

    return {
        "ok": True, "sets": s, "trend": trend,
        "n": n, "n_days": n_days, "span_days": span_days,
        "best": float(s["set_w"].max()),
        "best_row": s.loc[s["set_w"].idxmax()],
        "last": pd.Timestamp(s["date"].max()),
        "slope_m": slope_m, "r_time": r_time, "rho": rho, "resid": resid,
        "assoc": assoc, "fatigue": fatigue, "d_best": d_best,
    }


# ── Evolution by duration class ──────────────────────────────────────────────
def run_duration_evolution(iv: pd.DataFrame) -> dict:
    """Monthly best watts per duration class: heatmap data + per-class trend."""
    if iv is None or not len(iv):
        return {"ok": False, "reason": "No interval rows cached yet."}
    w = iv[iv["iv_type"] == "WORK"].copy()
    w["secs"] = pd.to_numeric(w["secs"], errors="coerce")
    w["avg_w"] = pd.to_numeric(w["avg_w"], errors="coerce")
    w = w.dropna(subset=["date", "secs", "avg_w"])
    if not len(w):
        return {"ok": False, "reason": "No WORK intervals in this range."}
    w["date"] = pd.to_datetime(w["date"])
    w["bucket"] = pd.cut(w["secs"], bins=BUCKET_EDGES, labels=BUCKET_LABELS,
                         right=False)
    w = w.dropna(subset=["bucket"])
    w["month"] = w["date"].dt.to_period("M").astype(str)

    best = (w.groupby(["bucket", "month"], observed=True)["avg_w"].max()
            .unstack("month"))
    cnt = (w.groupby(["bucket", "month"], observed=True)["avg_w"].size()
           .unstack("month").reindex(index=best.index, columns=best.columns))
    if not len(best):
        return {"ok": False, "reason": "No WORK intervals in this range."}

    # Row-normalise (each duration class coloured by its own worst→best) so
    # a 600 W sprint row and a 200 W tempo row are comparable at a glance.
    # Cells with no data stay NaN (blank), only a perfectly flat class gets
    # a uniform mid colour.
    rmin = best.min(axis=1)
    rmax = best.max(axis=1)
    rng = (rmax - rmin).replace(0.0, np.nan)
    z = ((best.sub(rmin, axis=0)).div(rng, axis=0) * 100)
    flat_rows = rng[rng.isna()].index
    if len(flat_rows):
        z.loc[flat_rows, :] = 50.0

    # Descriptive slope of the monthly bests (W/month), ≥ 4 populated months.
    slopes, months_n = {}, {}
    x_all = np.arange(len(best.columns), dtype=float)
    for lab, row in best.iterrows():
        mask = row.notna().to_numpy()
        months_n[lab] = int(mask.sum())
        if mask.sum() >= 4 and np.ptp(x_all[mask]) > 0:
            slopes[lab] = float(np.polyfit(x_all[mask], row[mask], 1)[0])
        else:
            slopes[lab] = np.nan

    tbl = pd.DataFrame({
        "Duration class": list(best.index),
        "W / month": [round(s, 2) if not np.isnan(s) else None
                      for s in (slopes[lab] for lab in best.index)],
        "Months": [months_n[lab] for lab in best.index],
        "Best ever (W)": [round(float(best.loc[lab].max()), 0)
                          for lab in best.index],
        "Trend": [("↑" if (not np.isnan(slopes[lab]) and slopes[lab] >= 1)
                   else "↓" if (not np.isnan(slopes[lab]) and slopes[lab] <= -1)
                   else "→" if not np.isnan(slopes[lab]) else "·")
                  for lab in best.index],
    })

    return {
        "ok": True, "z": z.to_numpy(), "best": best, "count": cnt,
        "months": list(best.columns), "labels": [str(x) for x in best.index],
        "text": [[("" if pd.isna(v) else f"{v:.0f}")
                  for v in row] for row in best.to_numpy()],
        "count_arr": cnt.fillna(0).to_numpy().astype(int),
        "table": tbl, "n_work": int(len(w)),
    }
