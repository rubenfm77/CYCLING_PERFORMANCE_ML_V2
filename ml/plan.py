# ml/plan.py — a training plan built from the athlete's own history.
#
# No coaching model is invented here: the page replays what the data already
# holds. The base year is scored in the open (labelled sessions, prescribed
# structure, measured efforts), the session mix is counted out of that year —
# prescribed comments where they exist, measured sets where they do not (2026
# has no coach comments left) — the watts come from the athlete's own bests
# expressed as %FTP (intervals.icu expands those off the athlete's current
# FTP setting), and the rests are the medians the athlete actually rode.
# The 12-week shape is a plain 4 + 4 + 3 + 1 progression laid over those
# sessions, stated as such — it is not copied from any year, because no year
# in this file holds a full progression to copy.
import numpy as np
import pandas as pd

from core.theme import FTP_CURRENT
from ml.interval_watts import measured, prescribed, rep_class
from ml.set_evolution import build_sets, fill_peak_efforts, add_signatures
from ml.type_comparison import BLANK_TYPE_TOKENS, dur_bucket, prep_types

KEY_FAMS = ["FTP", "VO2MAX", "BILLAT", "SST", "TEMPO"]

# %FTP targets per family, the physiology bands the app already uses
# (IF_THRESHOLD/IF_VO2 live in ml.type_comparison; these are their
# prescription twins). Absolute watts are NEVER prescribed: the block prints
# %FTP and the athlete's own reference best beside it.
TARGETS = {
    "FTP": [(15, "90-95%"), (10, "93-97%"), (0, "93-97%")],
    "SST": [(0, "85-90%")],
    "TEMPO": [(0, "78-84%")],
    "VO2MAX": [(0, "108-118%")],
    "BILLAT": [(0, "120-130%")],
    "Z2": [(0, "65-75%")],
}
WARMUP = [("15m", "ramp 50%-75%", "90rpm")]
COOLDOWN = [("10m", "ramp 60%-40%", "80rpm")]


def _target_for(family: str, nominal_min: float) -> str:
    bands = TARGETS.get(family, TARGETS["Z2"])
    for lo, t in bands:
        if nominal_min >= lo:
            return t
    return bands[-1][1]


def year_table(df_all: pd.DataFrame, P=None,
               M=None) -> pd.DataFrame:
    """One row per year with the open score the base-year picker shows.

    score = FTP-labelled sessions + key-family prescribed sessions +
    measured efforts of 10 minutes and up. Every component is printed, so
    the ranking can be argued with instead of trusted.
    """
    df = df_all.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    if P is None:
        from ml.interval_watts import prescribed as _p
        P = _p(df_all)
    if M is None:
        M = measured(df_all)
    lab = df["training_type"].astype(str).str.strip()
    ftp_set = (df.assign(_l=lab.values).groupby("year")["_l"]
               .apply(lambda s: int((s == "FTP").sum())))
    presc_key = (P[P["tt"].isin(KEY_FAMS)].groupby("year").size()
                 if len(P) else pd.Series(dtype=int))
    myear = pd.to_numeric(M["year"], errors="coerce")
    meas_long = (M[pd.to_numeric(M["secs"], errors="coerce") >= 600]
                 .assign(_y=myear[pd.to_numeric(M["secs"],
                                                errors="coerce") >= 600]
                         .values).groupby("_y").size())
    ftp_col = (pd.to_numeric(df["FTP"], errors="coerce")
               if "FTP" in df.columns else pd.Series(np.nan, index=df.index))
    rows = []
    for y in sorted(df["year"].dropna().unique()):
        y = int(y)
        rows.append({
            "year": y,
            "sessions": int((df["year"] == y).sum()),
            "ftp_labelled": int(ftp_set.get(y, 0)),
            "prescribed_key": int(presc_key.get(y, 0)),
            "measured_long": int(meas_long.get(y, 0)),
            "ftp_setting": (float(ftp_col[df["year"] == y].max())
                            if ftp_col[df["year"] == y].notna().any()
                            else np.nan),
            "score": (int(ftp_set.get(y, 0)) + int(presc_key.get(y, 0))
                      + int(meas_long.get(y, 0))),
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False,
                                          kind="stable").reset_index(drop=True)


def measured_mix(df_all: pd.DataFrame, iv, year: int,
                 min_sessions: int = 2) -> pd.DataFrame:
    """Session templates counted out of MEASURED sets of one year.

    One row per (family × nominal duration × rep count) with how often it
    was ridden, the median rep length, the median rest actually taken and
    the median watts. Only families the athlete labelled (or the honest
    heuristic fallbacks) and combos ridden at least `min_sessions` times.
    """
    sets = build_sets(iv, None, df_all)
    sets, _ = fill_peak_efforts(sets, df_all, iv=iv)
    sets, _ = add_signatures(sets)
    s = prep_types(sets)
    s = s[pd.to_datetime(s["date"]).dt.year == int(year)]
    if not len(s):
        return pd.DataFrame()
    s = s.copy()
    s["reps_i"] = pd.to_numeric(s["reps"], errors="coerce").fillna(1).astype(int)
    g = (s.groupby(["family", "dur_b", "reps_i"], as_index=False)
         .agg(sessions=("date", "nunique"),
              rep_secs_med=("rep_secs", "median"),
              rest_med=("rest", "median"),
              set_w_med=("set_w", "median")))
    g = g[g["sessions"] >= int(min_sessions)].copy()
    g["nominal"] = g["dur_b"].astype(float)
    return g.sort_values(["sessions", "family"], ascending=[False, True],
                         kind="stable").reset_index(drop=True)


def prescribed_mix(P: pd.DataFrame, year: int) -> pd.DataFrame:
    """Session templates counted out of the COACH'S prescribed comments.

    Same grain as measured_mix so the planner downstream cannot tell which
    source a template came from: family, nominal duration, rep count,
    sessions, median rep length. Prescriptions carry no measured rest, so
    rest stays missing and the planner falls back to the measured median.
    """
    p = P[(P["year"] == int(year)) & (P["tt"].isin(KEY_FAMS))].copy()
    if not len(p):
        return pd.DataFrame()
    p["nominal"] = [float(dur_bucket(x)) for x in
                    pd.to_numeric(p["secs"], errors="coerce")]
    p["reps_i"] = pd.to_numeric(p["n"], errors="coerce").fillna(1).astype(int)
    p["rep_secs_med"] = pd.to_numeric(p["secs"], errors="coerce")
    g = (p.groupby(["tt", "nominal", "reps_i"], as_index=False)
         .agg(sessions=("secs", "size"),
              rep_secs_med=("rep_secs_med", "median")))
    g = g.rename(columns={"tt": "family"})
    g["rest_med"] = np.nan
    g["set_w_med"] = np.nan
    return g.sort_values(["sessions", "family"], ascending=[False, True],
                         kind="stable").reset_index(drop=True)


def measured_refs(df_all: pd.DataFrame, iv) -> dict:
    """The athlete's own reference watts: best held watts per nominal.

    From the interval rows (detector cuts, both copies deduplicated by
    activity id where the file double-syncs — the max is unaffected by
    that) plus the FTP setting the file carries. Returned as {nominal:
    (watts, date)} and the FTP constant alongside.
    """
    out = {"ftp": float(FTP_CURRENT), "best": {}}
    if iv is None or not len(iv):
        return out
    keep = set(df_all["id"].astype(str)) if "id" in df_all.columns else set()
    w = iv[(iv["iv_type"] == "WORK")].copy()
    if keep:
        w = w[w["activity_id"].astype(str).isin(keep)]
    w["secs"] = pd.to_numeric(w["secs"], errors="coerce")
    w["avg_w"] = pd.to_numeric(w["avg_w"], errors="coerce")
    w = w.dropna(subset=["secs", "avg_w"])
    if not len(w):
        return out
    w["nominal"] = [float(dur_bucket(x)) for x in w["secs"]]
    for nom, g in w[w["nominal"] >= 5].groupby("nominal"):
        i = g["avg_w"].idxmax()
        out["best"][float(nom)] = (
            float(g.at[i, "avg_w"]),
            str(pd.to_datetime(g.at[i, "date"]).date()))
    return out


def weekday_pattern(df_all: pd.DataFrame, year: int,
                    fams=None) -> list:
    """Weekdays the athlete rides quality sessions, most frequent first.

    From the file's own dates for the base year, labelled sessions of the
    key families. The plan puts its hard sessions on these days because that
    is when the athlete actually rides them — not from a textbook.
    """
    fams = KEY_FAMS if fams is None else list(fams)
    df = df_all.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    lab = df["training_type"].astype(str).str.strip()
    sub = df[(df["year"] == int(year)) & (lab.isin(fams))]
    if not len(sub):
        return ["Tue", "Thu", "Sat"]
    wd = pd.to_datetime(sub["date"]).dt.strftime("%a")
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    cnt = wd.value_counts()
    return sorted(cnt.index, key=lambda d: (-int(cnt[d]),
                                            order.index(d)))[:4]


def fmt_dur(mins: float) -> str:
    if mins < 1:
        return f"{mins * 60:.0f}s"
    return f"{mins:.0f}m"


def icu_text(title: str, nominal_min: float, reps: int, target: str,
             rest_min=None, cadence: str = "90rpm") -> str:
    """One workout in intervals.icu's plain-text workout format.

    Sections separated by blank lines and the repeat count on the header
    line (`Main Set 2x`) — both are required for the builder to parse the
    steps. Targets are %FTP so the workout follows the athlete's current
    FTP setting wherever it is pasted; cadence rides along as a cue.
    """
    dur = fmt_dur(float(nominal_min))
    lines = [title, "", "Warmup"]
    lines += [f"- {d} {t} {c}" for d, t, c in WARMUP]
    lines += [""]
    if reps and int(reps) > 1:
        lines += [f"Main Set {int(reps)}x",
                  f"- {dur} {target} {cadence}"]
        if rest_min:
            lines += [f"- {fmt_dur(float(rest_min))} 50% {cadence}"]
        lines += [""]
    else:
        lines += ["Main Set", f"- {dur} {target} {cadence}"]
        lines += [""]
    lines += ["Cooldown"]
    lines += [f"- {d} {t} {c}" for d, t, c in COOLDOWN]
    return "\n".join(lines).strip() + "\n"


def build_plan(mix: pd.DataFrame, refs: dict, weekdays: list,
               start: pd.Timestamp, weeks: int = 12,
               rest_fallback: float = 10.0) -> list:
    """Twelve (or N) weeks of key sessions from the base year's templates.

    Each template becomes one weekly session on a pattern weekday; weeks
    1-4 ride it as counted, 5-8 add one rep, 9-11 ride the longest variant,
    12 halves the reps. Endurance fillers and rest days are noted, never
    prescribed as intervals — the file says how much the athlete rides, not
    the planner. Returns a list of week dicts with dated session dicts.
    """
    tpls = mix.head(4).to_dict("records") if len(mix) else []
    days = list(weekdays) + ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    seen = []
    for d in days:
        if d not in seen:
            seen.append(d)
        if len(seen) == 4:
            break
    plan = []
    for wk in range(1, int(weeks) + 1):
        if wk <= 4:
            phase, rep_add, note = ("Base", 0, "as ridden in the base year")
        elif wk <= 8:
            phase, rep_add, note = ("Build", 1, "+1 rep on the main sets")
        elif wk <= max(8, int(weeks) - 1):
            phase, rep_add, note = ("Peak", 1, "longest variant, full reps")
        else:
            phase, rep_add, note = ("Easy", 0, "half reps, form week")
        sessions = []
        for i, t in enumerate(tpls):
            fam = str(t["family"])
            nom = float(t["nominal"])
            reps = int(t.get("reps_i", 1) or 1)
            if phase == "Easy":
                reps = max(1, reps // 2)
            else:
                reps = reps + (rep_add if nom >= 10 else 0)
            rest = t.get("rest_med")
            try:
                rest = float(rest)
            except (TypeError, ValueError):
                rest = np.nan
            rest_min = (rest / 60.0 if np.isfinite(rest) and rest > 0
                        else (nom if nom <= 1.0 else
                              (5.0 if nom < 10 else float(rest_fallback))))
            target = _target_for(fam, nom)
            day = seen[i % len(seen)]
            date = (pd.Timestamp(start)
                    + pd.Timedelta(days=(wk - 1) * 7
                                   + (["Mon", "Tue", "Wed", "Thu", "Fri",
                                       "Sat", "Sun"].index(day)
                                      - pd.Timestamp(start).dayofweek) % 7))
            title = (f"{fam} {reps}x{fmt_dur(nom)}" if reps > 1
                     else f"{fam} {fmt_dur(nom)}")
            ref = refs.get("best", {}).get(nom)
            sessions.append({
                "week": wk, "phase": phase, "day": day, "date": date.date(),
                "family": fam, "nominal": nom, "reps": reps,
                "target": target, "rest_min": round(rest_min, 1),
                "title": title,
                "reference": (f"you held {ref[0]:.0f} W for this length on "
                              f"{ref[1]}" if ref else "no measured best yet"),
                "text": icu_text(title, nom, reps, target, rest_min),
            })
        plan.append({"week": wk, "phase": phase, "note": note,
                     "sessions": sessions})
    return plan


def plan_summary(plan: list) -> pd.DataFrame:
    """Every prescribed interval of the plan, one row per session."""
    rows = []
    for wk in plan:
        for s in wk["sessions"]:
            _d = fmt_dur(s["nominal"])
            iv_txt = (f"{s['reps']}x{_d} @ {s['target']}"
                      if s["reps"] > 1
                      else f"{_d} @ {s['target']}")
            rows.append({"Week": s["week"], "Date": str(s["date"]),
                         "Day": s["day"], "Session": s["title"],
                         "Intervals": iv_txt,
                         "Rest": (f"{fmt_dur(s['rest_min'])} easy"
                                  if s["reps"] > 1 else "—"),
                         "Reference": s["reference"]})
    return pd.DataFrame(rows)


def rest_fallback_for(mix: pd.DataFrame) -> float:
    """Median measured rest of the long templates, in minutes."""
    if mix is None or not len(mix) or "rest_med" not in mix.columns:
        return 10.0
    r = pd.to_numeric(mix[mix["nominal"] >= 10]["rest_med"],
                      errors="coerce").dropna() / 60.0
    if not len(r):
        return 10.0
    return round(float(r.median()), 1)
