# ml/plan.py — a training plan built from the athlete's own history.
#
# No coaching model is invented here: the page replays what the data already
# holds. The base year is scored in the open (labelled sessions, prescribed
# structure, measured efforts), the session mix is counted out of that year —
# prescribed comments where they exist, measured sets where they do not (2026
# has no coach comments left) — the watts are absolute, computed at the
# athlete's current FTP so every pasted step reads in watts, and the rests
# are the medians the athlete actually rode.
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

# ── Book session library (cycling-coach skill: workout-library.md) ──────────
# Structures, %FTP bands and rests as the book prescribes them. Watts are
# ALWAYS rendered beside the % (book: "always show both % FTP and absolute
# watts"), computed at the athlete's current FTP so the paste survives FTP
# changes through the % target. Reps-before-duration (book progression rule)
# is how templates grow; durations themselves come from the athlete's mix.
BOOK_TARGETS = {
    # family: [(nominal_min_floor, target)]
    "FTP": [(15, "95-100%"), (10, "93-97%"), (0, "93-97%")],
    "SST": [(0, "88-94%")],
    "TEMPO": [(0, "76-88%")],
    "VO2MAX": [(4, "106-115%"), (0, "110-120%")],
    "BILLAT": [(0, "120%")],
    "TORQUE": [(0, "85-95%")],
    "Z2": [(0, "56-75%")],
    "REC": [(0, "50%")],
}
BOOK_REST_MIN = {
    # family: rest between reps (min); FTP refines by length below
    "FTP": 10.0, "SST": 5.0, "TEMPO": 6.0, "VO2MAX": 5.0, "BILLAT": 0.5,
    "TORQUE": 5.0,
}
BOOK_BILLAT_SET_REST_MIN = 5.0   # easy spin between 10-pair blocks
BOOK_CADENCE = {"TORQUE": "55rpm"}
BOOK_DEFAULT_CADENCE = "90rpm"
# Fallback (nominal min, reps) when a pattern family is absent from the base
# mix: the book's staple for that bin, labelled as book-standard, never
# passed off as the athlete's own riding.
BOOK_DEFAULTS = {
    "FTP": (20.0, 2), "VO2MAX": (5.0, 5), "BILLAT": (0.5, 20),
    "SST": (12.0, 3), "TEMPO": (20.0, 2), "TORQUE": (8.0, 5),
}
BOOK_PURPOSE = {
    "FTP": "Sustained threshold power for climbing",
    "SST": "High effect per fatigue unit",
    "TEMPO": "Aerobic support",
    "VO2MAX": "Maximal aerobic power",
    "BILLAT": "Time-efficient VO2",
    "TORQUE": "Climbing force, seated",
    "Z2": "Aerobic base",
    "REC": "Recovery",
}
# Book-standard warm-up (15 min progressive + 3x1 min openers) and cool-down,
# written as flat `-` steps only: custom repeat headers inside Warmup are a
# parser risk, so the openers are spelled out instead of folded.
BOOK_WARMUP = [("10m", "ramp 50%-65%", "90rpm"),
               ("1m", "75%", "100rpm"), ("1m", "50%", "90rpm"),
               ("1m", "75%", "100rpm"), ("1m", "50%", "90rpm"),
               ("1m", "75%", "100rpm"), ("3m", "50%", "90rpm")]
BOOK_COOLDOWN = [("10m", "ramp 60%-40%", "80rpm")]


def watts(pct: str, ftp: float = 240.0) -> tuple:
    """A %FTP band -> (lo, hi) absolute watts. "95-100%" -> (228, 240)."""
    try:
        parts = str(pct).replace("%", "").split("-")
        lo = float(parts[0]) / 100.0 * float(ftp)
        hi = float(parts[1]) / 100.0 * float(ftp) if len(parts) > 1 else lo
        return (round(lo), round(hi))
    except (TypeError, ValueError, IndexError):
        return (None, None)


def target_for(family: str, nominal_min: float) -> str:
    """%FTP band for one family at one nominal length (book zones)."""
    bands = BOOK_TARGETS.get(family, BOOK_TARGETS["Z2"])
    for lo, t in bands:
        if nominal_min >= lo:
            return t
    return bands[-1][1]


def book_rest_for(family: str, nominal_min: float) -> float:
    """Book rest between reps (min): FTP refines by length, Billat is 30 s."""
    if family == "FTP":
        return 10.0 if nominal_min >= 15 else 8.0
    return float(BOOK_REST_MIN.get(family, 5.0))


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


def _watts_suffix(target: str, ftp: float) -> str:
    """" (228-240W)" for a band, " (288W)" for a single value, "" unparseable."""
    lo, hi = watts(target, ftp)
    if lo is None:
        return ""
    return f" ({lo}-{hi}W)" if hi != lo else f" ({lo}W)"


def _paste_step(dur: str, target: str, cadence: str, ftp: float) -> str:
    """One flat step in absolute watts: `- 10m ramp 120-156w 90rpm`.

    The book warm-up/cool-down tables are written as %FTP bands ("ramp
    50%-65%", "75%"); the paste carries the same band converted at the
    athlete's current FTP, because this page prescribes watts, not
    percentages. An unparseable target passes through untouched rather
    than vanishing.
    """
    t = str(target)
    prefix, band = ("ramp ", t[len("ramp "):]) if t.startswith("ramp ") else ("", t)
    lo, hi = watts(band, ftp)
    if lo is None:
        return f"- {dur} {t} {cadence}"
    wtxt = f"{lo}-{hi}w" if hi != lo else f"{lo}w"
    return f"- {dur} {prefix}{wtxt} {cadence}"


def build_session(family: str, nominal_min: float, reps: int,
                  rest_min=None, ftp: float = 240.0, ref=None) -> dict:
    """One book-structured workout, paste-ready.

    Name line carries %FTP AND watts (it is not a step, so zero parse risk);
    every step carries absolute watts + cadence. Warm-up/cool-down are the
    book standard; Billat 30/30 blocks are flattened (no nested repeats —
    the builder does not support them) with an easy 5 min every 10 pairs.
    Returns title, purpose, total minutes and the paste text. `ref` is
    (watts, date) of the athlete's own best at this length, or None.
    """
    nom = float(nominal_min)
    n = max(1, int(reps or 1))
    target = target_for(family, nom)
    ftp = float(ftp or 240.0)
    if rest_min is None or not np.isfinite(float(rest_min)):
        rest_min = book_rest_for(family, nom)
    else:
        rest_min = float(rest_min)
    cad = BOOK_CADENCE.get(family, BOOK_DEFAULT_CADENCE)

    def _wstep(t: str, c: str = None) -> str:
        """A work step in absolute watts: `- 20m 228-240w 90rpm`."""
        lo, hi = watts(t, ftp)
        wtxt = f"{lo}-{hi}w" if hi != lo else f"{lo}w"
        return wtxt, (c or cad)

    _wt, _wc = _wstep(target)
    _rt, _ = _wstep("50%")
    _bt, _ = _wstep("55%")
    if family == "BILLAT" and nom <= 1.0:
        _ot, _oc = _wstep("120%", "90rpm")
        main = []
        for i in range(n):
            main.append(("30s", _ot, _oc))
            main.append((f"{fmt_dur(rest_min)}", _bt, BOOK_DEFAULT_CADENCE))
            if (i + 1) % 10 == 0 and i + 1 < n:
                main.append(("5m", _rt, BOOK_DEFAULT_CADENCE))
        rep_header = None
        work_desc = f"{n}x30s on/off"
    elif n > 1:
        main = [(fmt_dur(nom), _wt, _wc),
                (fmt_dur(rest_min), _rt, BOOK_DEFAULT_CADENCE)]
        rep_header = f"Main Set {n}x"
        work_desc = f"{n}x{fmt_dur(nom)}"
    else:
        main = [(fmt_dur(nom), _wt, _wc)]
        rep_header = None
        work_desc = fmt_dur(nom)
    base = f"{family} {work_desc}".strip()
    title = f"{base} @ {target}{_watts_suffix(target, ftp)}"
    lines = [title, "", "Warmup"]
    lines += [_paste_step(d, t, c, ftp) for d, t, c in BOOK_WARMUP]
    lines += [""]
    if rep_header:
        lines += [rep_header] + [f"- {d} {t} {c}" for d, t, c in main] + [""]
    else:
        lines += ["Main Set"] + [f"- {d} {t} {c}" for d, t, c in main] + [""]
    lines += ["Cooldown"]
    lines += [_paste_step(d, t, c, ftp) for d, t, c in BOOK_COOLDOWN]
    lines += [""]
    lines += [f"Targets at FTP {ftp:.0f}W. "
              f"{BOOK_PURPOSE.get(family, 'Aerobic work')}."]
    text = "\n".join(lines).strip() + "\n"
    warm = sum(_step_min(d) for d, _, _ in BOOK_WARMUP)
    cool = sum(_step_min(d) for d, _, _ in BOOK_COOLDOWN)
    work = sum(_step_min(d) for d, _, _ in main)
    if rep_header:
        # Nx repeats the work step AND its rest; no rest after the last rep.
        work = n * work - (float(rest_min) if rest_min else 0.0)
    total = warm + max(work, 0.0) + cool
    return {
        "family": family, "nominal": nom, "reps": n, "target": target,
        "rest_min": round(float(rest_min), 1),
        "title": title, "purpose": BOOK_PURPOSE.get(family, "Aerobic work"),
        "total_min": round(total, 0), "text": text,
        "reference": (f"you held {ref[0]:.0f} W for this length on {ref[1]}"
                      if ref else "no measured best yet"),
    }


def _step_min(dur: str) -> float:
    """"20m" -> 20.0, "30s" -> 0.5. Best effort, never a crash."""
    try:
        s = str(dur).strip().lower()
        if s.endswith("ms"):
            return float(s[:-2])
        if s.endswith("m"):
            return float(s[:-1])
        if s.endswith("s"):
            return float(s[:-1]) / 60.0
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def session_table(sessions: list) -> pd.DataFrame:
    """One row per pasteable session: date, work, rest, reference."""
    rows = []
    for s in sessions:
        _d = fmt_dur(s["nominal"])
        iv_txt = (f"{s['reps']}x{_d} @ {s['target']}"
                  if s["reps"] > 1 and not (
                      s["family"] == "BILLAT" and s["nominal"] <= 1.0)
                  else (f"{s['reps']}x30s on/off @ {s['target']}"
                        if s["family"] == "BILLAT"
                        else f"{_d} @ {s['target']}"))
        rows.append({"Date": str(s.get("date", "—")),
                     "Day": s.get("day", "—"),
                     "Session": s["title"],
                     "Intervals": iv_txt,
                     "Rest": (f"{fmt_dur(s['rest_min'])} easy"
                              if s["reps"] > 1 else "—"),
                     "Reference": s.get("reference", "")})
    return pd.DataFrame(rows)


def endurance_med_duration(df_all: pd.DataFrame) -> float:
    """Median hours of the athlete's own endurance rides (END/AEROBIC BASE).

    The Z2 filler blocks use this, not an invented 2 h — the file says how
    long the athlete actually rides easy. Falls back to 2 h with nothing to
    measure.
    """
    try:
        lab = df_all["training_type"].astype(str).str.strip()
        d = pd.to_numeric(df_all.get("duration_h"), errors="coerce")
        m = d[lab.isin(["END", "AEROBIC BASE"])].dropna()
        if len(m) >= 5:
            return round(float(m.median()), 1)
    except (TypeError, ValueError, KeyError):
        pass
    return 2.0


def icu_text(title: str, nominal_min: float, reps: int, target: str,
             rest_min=None, cadence: str = "90rpm",
             ftp: float = 240.0) -> str:
    """One workout in intervals.icu's plain-text workout format.

    Sections separated by blank lines and the repeat count on the header
    line (`Main Set 2x`) — both are required for the builder to parse the
    steps. Every step carries absolute watts + cadence; the title line
    carries %FTP and watts (never a step, so zero parse risk), plus one
    trailing note line. Kept as the generic builder; build_session() is
    the book-structured path.
    """
    dur = fmt_dur(float(nominal_min))
    wline = _watts_suffix(target, ftp)
    _lo, _hi = watts(target, ftp)
    _wtxt = (f"{_lo}-{_hi}w" if _hi != _lo else f"{_lo}w") \
        if _lo is not None else target
    _rlo, _rhi = watts("50%", ftp)
    _rtxt = (f"{_rlo}-{_rhi}w" if _rhi != _rlo else f"{_rlo}w") \
        if _rlo is not None else "50%"
    lines = [f"{title} @ {target}{wline}", "", "Warmup"]
    lines += [_paste_step(d, t, c, ftp) for d, t, c in BOOK_WARMUP]
    lines += [""]
    if reps and int(reps) > 1:
        lines += [f"Main Set {int(reps)}x",
                  f"- {dur} {_wtxt} {cadence}"]
        if rest_min:
            lines += [f"- {fmt_dur(float(rest_min))} {_rtxt} {cadence}"]
        lines += [""]
    else:
        lines += ["Main Set", f"- {dur} {_wtxt} {cadence}"]
        lines += [""]
    lines += ["Cooldown"]
    lines += [_paste_step(d, t, c, ftp) for d, t, c in BOOK_COOLDOWN]
    lines += [""]
    lines += [f"Targets at FTP {float(ftp):.0f}W."]
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
            day = seen[i % len(seen)]
            date = (pd.Timestamp(start)
                    + pd.Timedelta(days=(wk - 1) * 7
                                   + (["Mon", "Tue", "Wed", "Thu", "Fri",
                                       "Sat", "Sun"].index(day)
                                      - pd.Timestamp(start).dayofweek) % 7))
            ref = refs.get("best", {}).get(nom)
            _s = build_session(fam, nom, reps, rest_min,
                               ftp=refs.get("ftp", 240.0),
                               ref=(ref if ref else None))
            sessions.append({
                "week": wk, "phase": phase, "day": day, "date": date.date(),
                "family": fam, "nominal": nom, "reps": _s["reps"],
                "target": _s["target"], "rest_min": _s["rest_min"],
                "title": _s["title"],
                "reference": _s["reference"],
                "text": _s["text"],
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
