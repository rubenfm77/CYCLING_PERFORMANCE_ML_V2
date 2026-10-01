# ml/interval_watts.py — interval watts, not average watts. (NEW FILE)
r"""
The user rejected average power outright: "i dont give a fuck about average
watts". This module answers the request they actually made — the watts of the
INTERVAL — from the two places interval watts genuinely exist, and refuses to
invent a third.

TWO SOURCES, AND THEY DO NOT OVERLAP
------------------------------------
prescribed   the coach's own text in `WorkoutDescription`, in Spanish/Catalan:
             "WU: 30' entre 155-170W  MS: 3x15' PLA entre 225-240w  10' RECUP".
             That is n = 3 reps of 15 minutes with a target RANGE. The user's
             instruction — "grab the lower end" — makes the LOWER bound the
             number. It is what the coach was willing to call a success, and it
             is the only end of the range that is not rounded-up optimism.
             Coverage: 2019-2025. 2026 has NO prescription at all.

measured     `interval_summary`, intervals.icu's auto-detected efforts:
             "['5x 3m 257w', '2x 1m53s 255w']". These are real measured watts.
             Coverage: 2025 (4 records) and 2026 (635 records).

Because the two cover different years, they are two separate functions, two
separate censuses and two separate charts. They are never concatenated, never
interpolated into one line, and never used to fill each other's gaps: "the watts
you were told to do" and "the watts you actually did" are different numbers and a
line joining them would be a fiction.

THE BUCKETING VARIABLE IS REP LENGTH, NOT SESSION LENGTH
---------------------------------------------------------
A 3x1' sprint prescription and a 2x20' time trial are both "interval watts", and
pooling them produces a number that describes nothing. So every cell here is
keyed on the REP LENGTH CLASS, and reps of different lengths are never averaged
together. The user's duration rule is carried over: under 90 s the class is not
subdivided, because at that scale the exact seconds ARE the description, and the
exact seconds travel with every row.

Below 90 s the class is `under 90s` and fmt_rep() prints e.g. `30s`. From 90 s up
the class is a range, but the exact length is still printed next to the median so
a 10:00 and a 10:20 rep are never presented as the same effort.

A session's own length is still reported per cell (median, and the mix of session
duration classes) rather than being the bucket, because for FTP work a 3-hour ride
and a 90-minute ride containing the same 3x12' set are the same INTERVAL done in a
different ride. Pooling them is disclosed, not hidden.

WHY NOT `PowerMax`
------------------
`PowerMax` is populated for 902 sessions across 2019-2025, which makes it look
like the obvious interval-power column. It is not: it is session PEAK, median
553 W, max 999 W. That is a sprint spike, not an interval. It is deliberately
absent from this module.

HONESTY RULES
-------------
* descriptive only — observed history and descriptive association, never causal
  or predictive; nothing here says a prescription caused a result;
* n is on every point and every table row, beside the number it belongs to;
* a cell below MIN_CELL_N is BLANK, never zero, never carried forward, and the
  line BREAKS there rather than bridging it;
* no cell ever mixes two training types;
* the parse rate is reported with its failures. A parser that silently drops
  rows is the failure mode this project exists to avoid, so `audit()` counts
  what was skipped and says why;
* where a watt figure came from a RANGE, both ends are kept. The lower end is the
  headline only because the user asked for it, not because the upper end was
  hidden.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Same floor as ml/year_over_year.py. Below three sessions in a cell, one hard
# ride moves the median and the chart would be reporting a single session as a
# trend. Kept as a separate constant rather than imported so that this module
# cannot be silently loosened by a change over there.
MIN_CELL_N = 3

# Rep-length classes. `under 90s` is deliberately not subdivided: at that scale
# the exact seconds are the only honest description, and the user rule says
# sub-90-second efforts keep their seconds.
REP_CLASSES = [
    ("under 90s", 0, 90),
    ("90s-5min", 90, 300),
    ("5-10 min", 300, 600),
    ("10-20 min", 600, 1200),
    ("20-30 min", 1200, 1800),
    ("30+ min", 1800, np.inf),
]
REP_ORDER = [c[0] for c in REP_CLASSES]

# `'` is MINUTES and `"` is SECONDS in these comments. They are separate
# alternatives and are never merged into one unit group: reading 10" as ten
# minutes instead of ten seconds inflates a sprint session by 60x, and it is a
# mistake this module has already made once.
_MIN_UNIT = r"(?:'|’|´|min)"
_SEC_QUOTE = r"(?:\"|”|″)"
_SEC_WORD = r"(?:s|s'|seg)"

PRESCRIBED_RE = re.compile(
    r"(?P<n>\d{1,2})\s*[x*]\s*"
    r"(?:(?P<mins>\d{1,3})\s*" + _MIN_UNIT + r""
    r"|(?P<qsec>\d{1,3})\s*" + _SEC_QUOTE + r""
    r"|(?P<sec>\d{1,2})\s*" + _SEC_WORD + r")"
    # The gap between the rep length and the wattage may contain words but NOT
    # digits. That is what stops a rep from borrowing a recovery effort's watts
    # and vice versa: "3x15' (10' recup -150w)" cannot match, because crossing
    # the "10" would require a digit in the gap.
    r"[^0-9]{0,40}?"
    r"(?:entre\s+|a\s+|target\s+)?"
    r"(?P<w1>\d{2,4})\s*(?:-\s*(?P<w2>\d{2,4})\s*)?w\b",
    re.IGNORECASE)

MEASURED_RE = re.compile(
    r"(?P<n>\d{1,2})\s*x\s*"
    r"(?:(?P<mi>\d{1,3})\s*m(?P<s>\d{0,2})\s*s?"
    r"|(?P<sec>\d{1,3})\s*s)"
    r"\s*(?P<w>\d{2,4})\s*w",
    re.IGNORECASE)

# A description cell is blank in any of these ways. Deliberately the SAME set the
# loader uses, imported rather than retyped.
from core.theme import BLANK_TYPE_TOKENS  # noqa: E402

_STOP_MARKERS = ("CD:", "MD:", "FINAL:")


def _blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return str(v).strip() in BLANK_TYPE_TOKENS


def _has_label(series: pd.Series) -> pd.Series:
    """Exactly core.data's test, importing its constant so the two cannot drift."""
    return series.notna() & ~series.astype(str).str.strip().isin(BLANK_TYPE_TOKENS)


def rep_class(secs) -> str:
    """Rep length -> class label. Never crosses a boundary silently."""
    if secs is None or (isinstance(secs, float) and np.isnan(secs)):
        return "unknown"
    secs = float(secs)
    for label, lo, hi in REP_CLASSES:
        if lo <= secs < hi:
            return label
    return "unknown"


def fmt_rep(secs) -> str:
    """Exact rep length. Seconds under 90 s, then m:ss — never a bare class."""
    if secs is None or (isinstance(secs, float) and np.isnan(secs)):
        return "—"
    secs = int(round(float(secs)))
    if secs < 90:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    return f"{m}:{s:02d}" if s else f"{m}:00"


def fmt_watts(v, decimals: int = 0) -> str:
    """Thousands separators on big numbers, per the display rule."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v):,.{decimals}f}"


# ── prescribed ────────────────────────────────────────────────────────────────
def main_set(desc) -> str:
    """The MS: portion only. Warm-up and cool-down are not the workout."""
    d = str(desc)
    if "MS:" in d:
        seg = d.split("MS:", 1)[1]
        for stop in _STOP_MARKERS:
            if stop in seg:
                seg = seg.split(stop, 1)[0]
        return seg.strip()
    return d.strip()


def parse_prescribed(desc) -> list:
    """All 'NxD' reps with a goal wattage in one description.

    Returns [(n_reps, seconds, watts_low, watts_high), ...]. An empty list means
    "nothing unambiguous here", which is a normal outcome and is counted by
    audit() rather than hidden.
    """
    if _blank(desc):
        return []
    out = []
    for m in PRESCRIBED_RE.finditer(main_set(desc)):
        if m.group("mins"):
            secs = int(m.group("mins")) * 60
        elif m.group("qsec"):
            secs = int(m.group("qsec"))
        elif m.group("sec"):
            secs = int(m.group("sec"))
        else:
            continue
        lo = float(m.group("w1"))
        hi = float(m.group("w2")) if m.group("w2") else lo
        if hi < lo:
            lo, hi = hi, lo
        if secs >= 30 and lo >= 100:
            out.append((int(m.group("n")), secs, lo, hi))
    return out


def prescribed(df: pd.DataFrame) -> pd.DataFrame:
    """Session grain: one row per session that has a parseable main set.

    The session's MAIN SET is the rep with the highest goal watts. That is the
    effort the coach actually wanted achieved, and it is why the easy
    "WU: 30' entre 155-170W" warm-up can never win over "MS: 3x15' ... 225-240w".
    When several reps tie on goal watts the longest one wins, because a 4x4' and a
    1x16' at the same watts are different efforts and the longer one is the more
    informative of the two.
    """
    if df is None or not len(df):
        return pd.DataFrame(columns=["year", "tt", "dur_class", "n", "secs",
                                     "cls", "w_lo", "w_hi"])
    label = (df["training_type"].astype(object)
             if "training_type" in df.columns else None)
    keep = _has_label(label) if label is not None else pd.Series(
        True, index=df.index)
    desc = (df["WorkoutDescription"]
            if "WorkoutDescription" in df.columns else None)

    rows = []
    for i in df.index[keep]:
        if desc is None:
            break
        reps = parse_prescribed(desc.at[i])
        if not reps:
            continue
        n, secs, lo, hi = max(reps, key=lambda r: (r[2], r[1]))
        rows.append({
            "year": df.at[i, "date"].year if "date" in df.columns else None,
            "tt": str(label.at[i]).strip(),
            "dur_class": _dur_class(df.at[i, "duration_s"])
            if "duration_s" in df.columns else "unknown",
            "n": n, "secs": secs, "cls": rep_class(secs),
            "w_lo": lo, "w_hi": hi,
            "reps_found": len(reps),
            "has_ms": "MS:" in str(desc.at[i]),
        })
    return pd.DataFrame(rows)


def _dur_class(secs) -> str:
    if secs is None or (isinstance(secs, float) and np.isnan(secs)):
        return "unknown"
    mins = float(secs) / 60.0
    for label, lo, hi in (("under 45 min", 0, 45), ("45-90 min", 45, 90),
                          ("90-150 min", 90, 150), ("150-240 min", 150, 240),
                          ("240+ min", 240, np.inf)):
        if lo <= mins < hi:
            return label
    return "unknown"


def prescribed_cells(p: pd.DataFrame, tt: str,
                     min_n: int = MIN_CELL_N) -> pd.DataFrame:
    """One row per (rep class x year) for ONE type. Drawable flag per cell.

    The cell key is rep class, never session duration class: two 3x12' sets in a
    90-minute ride and a 3-hour ride are the same INTERVAL. The session duration
    mix is carried as a column so a shift between years is visible rather than
    averaged away.
    """
    if p is None or not len(p):
        return pd.DataFrame(columns=["tt", "cls", "year", "n", "med_w_lo",
                                     "med_w_hi", "q1_w_lo", "q3_w_lo",
                                     "rep_secs", "dur_mix", "drawable"])
    s = p[p["tt"] == tt]
    if not len(s):
        return pd.DataFrame(columns=["tt", "cls", "year", "n", "med_w_lo",
                                     "med_w_hi", "q1_w_lo", "q3_w_lo",
                                     "rep_secs", "dur_mix", "drawable"])
    g = s.groupby(["cls", "year"])
    out = g.agg(
        n=("w_lo", "size"),
        med_w_lo=("w_lo", "median"),
        med_w_hi=("w_hi", "median"),
        q1_w_lo=("w_lo", lambda x: x.quantile(0.25)),
        q3_w_lo=("w_lo", lambda x: x.quantile(0.75)),
        rep_secs=("secs", "median"),
    ).reset_index()
    out["tt"] = tt
    mix = s.groupby(["cls", "year"])["dur_class"].agg(
        lambda x: " / ".join(f"{k} {v}" for k, v in x.value_counts().items()))
    out["dur_mix"] = [mix.get((c, y), "") for c, y in zip(out["cls"], out["year"])]
    out["drawable"] = out["n"] >= min_n
    return out.sort_values(["cls", "year"]).reset_index(drop=True)


def prescribed_evolution(p: pd.DataFrame, tt: str,
                         min_n: int = MIN_CELL_N) -> pd.DataFrame:
    """One row per DRAWN point, so a thin year simply has no row and the line
    breaks across it instead of being interpolated over it."""
    c = prescribed_cells(p, tt, min_n)
    if not len(c):
        return c
    return c[c["drawable"]].reset_index(drop=True)


def prescribed_yoy(p: pd.DataFrame, tt: str,
                   min_n: int = MIN_CELL_N) -> pd.DataFrame:
    """'Compare with the previous year', per rep class.

    The previous comparable year is the previous year IN THE SAME REP CLASS that
    clears the floor — not simply year minus one. When the gap is more than one
    year it is printed, so "2022 vs 2025" never reads as if it were a single
    season's change. A delta is emitted only when BOTH years clear the floor.
    """
    c = prescribed_cells(p, tt, min_n)
    if not len(c):
        return pd.DataFrame(columns=["tt", "cls", "year", "n", "med_w_lo",
                                     "prev_year", "prev_n", "prev_med", "delta",
                                     "gap_years", "rep_secs"])
    rows = []
    for cls in c["cls"].unique():
        s = c[c["cls"] == cls].sort_values("year")
        prev_ok = None
        for _, row in s.iterrows():
            if not row["drawable"]:
                continue
            rec = {"tt": tt, "cls": cls, "year": int(row["year"]),
                   "n": int(row["n"]), "med_w_lo": float(row["med_w_lo"]),
                   "rep_secs": int(row["rep_secs"])}
            if prev_ok is None:
                rec.update(prev_year=None, prev_n=None, prev_med=None,
                           delta=None, gap_years=None)
            else:
                rec.update(prev_year=int(prev_ok["year"]),
                           prev_n=int(prev_ok["n"]),
                           prev_med=float(prev_ok["med_w_lo"]),
                           delta=float(row["med_w_lo"]) - float(prev_ok["med_w_lo"]),
                           gap_years=int(row["year"]) - int(prev_ok["year"]))
            rows.append(rec)
            prev_ok = row
    return pd.DataFrame(rows)


# ── measured ──────────────────────────────────────────────────────────────────
def parse_measured(raw) -> list:
    """Detected efforts in one interval_summary cell -> [(n, seconds, watts)]."""
    if _blank(raw):
        return []
    out = []
    for chunk in re.findall(r"[^'\"]*?\d{1,2}\s*x\s*[^'\"]+", str(raw)):
        m = MEASURED_RE.search(chunk)
        if not m:
            continue
        secs = ((int(m.group("mi")) * 60 + int(m.group("s") or 0))
                if m.group("mi") else int(m.group("sec")))
        out.append((int(m.group("n")), secs, float(m.group("w"))))
    return out


def measured(df: pd.DataFrame) -> pd.DataFrame:
    """One row per DETECTED EFFORT. Rep grain, not session grain.

    This is deliberately a different grain from prescribed(), which is session
    grain. The two are never joined: a detected effort is a thing the head unit
    saw, and a prescribed rep is a thing a human wrote down.
    """
    if df is None or not len(df):
        return pd.DataFrame(columns=["year", "tt", "n", "secs", "cls", "w"])
    label = (df["training_type"].astype(object)
             if "training_type" in df.columns else None)
    keep = _has_label(label) if label is not None else pd.Series(
        True, index=df.index)
    src = (df["interval_summary"]
           if "interval_summary" in df.columns else None)
    rows = []
    for i in df.index[keep]:
        if src is None:
            break
        for n, secs, w in parse_measured(src.at[i]):
            rows.append({
                "year": df.at[i, "date"].year if "date" in df.columns else None,
                "tt": str(label.at[i]).strip(),
                "n": n, "secs": secs, "cls": rep_class(secs), "w": w,
            })
    return pd.DataFrame(rows)


def measured_cells(m: pd.DataFrame, year=None, min_n: int = MIN_CELL_N
                   ) -> pd.DataFrame:
    """One row per (type x rep class) for the given year, drawable flag set.

    Rep classes are NEVER pooled here: each is its own row with its own n, which
    is why the caller draws one line per class instead of a single "interval
    watts" number that would mean different things for different efforts.
    """
    cols = ["tt", "cls", "year", "n", "med_w", "q1_w", "q3_w", "rep_secs",
            "sessions", "drawable"]
    if m is None or not len(m):
        return pd.DataFrame(columns=cols)
    s = m if year is None else m[m["year"] == year]
    if not len(s):
        return pd.DataFrame(columns=cols)
    out = s.groupby(["tt", "cls", "year"]).agg(
        n=("w", "size"),
        med_w=("w", "median"),
        q1_w=("w", lambda x: x.quantile(0.25)),
        q3_w=("w", lambda x: x.quantile(0.75)),
        rep_secs=("secs", "median"),
    ).reset_index()
    sess = s.groupby(["tt", "cls", "year"]).apply(
        lambda g: g.index.nunique(), include_groups=False)
    out["sessions"] = [int(sess.get((t, c, y), 0))
                       for t, c, y in zip(out["tt"], out["cls"], out["year"])]
    out["drawable"] = out["n"] >= min_n
    return out.sort_values(["tt", "cls"]).reset_index(drop=True)


# ── honesty surfaces ─────────────────────────────────────────────────────────
def audit(df: pd.DataFrame) -> dict:
    """What the prescription parser saw, kept, and dropped. Never a bare rate.

    `skipped_steady` is separated from `skipped_unread` on purpose: a
    description with no repetition in it is a steady endurance ride and has no
    intervals by definition, which is a correct non-match and not a loss.
    Reporting one combined number would make a healthy parser look like it lost
    60% of the file.
    """
    out = {"seen": 0, "parsed": 0, "skipped_steady": 0, "skipped_unread": 0,
           "skipped_unlabelled": 0, "years": {}, "parsed_years": {},
           "examples_unread": []}
    if df is None or not len(df):
        return out
    if "WorkoutDescription" not in df.columns:
        return out
    desc = df["WorkoutDescription"]
    label = (df["training_type"].astype(object)
             if "training_type" in df.columns else None)
    yrs = df["date"].dt.year if "date" in df.columns else None

    for i in df.index:
        d = desc.at[i]
        if _blank(d):
            continue
        out["seen"] += 1
        y = int(yrs.at[i]) if yrs is not None and pd.notna(yrs.at[i]) else None
        if y is not None:
            out["years"][y] = out["years"].get(y, 0) + 1
        labelled = True if label is None else bool(_has_label(label.loc[[i]]).iloc[0])
        reps = parse_prescribed(d)
        if not reps:
            # No repetition syntax at all -> a steady ride, correctly no intervals.
            if re.search(r"\d{1,2}\s*[x*]\s*[\d'\"mins]", main_set(d)):
                out["skipped_unread"] += 1
                if len(out["examples_unread"]) < 8:
                    out["examples_unread"].append(str(d)[:150])
            else:
                out["skipped_steady"] += 1
            continue
        if not labelled:
            out["skipped_unlabelled"] += 1
            continue
        out["parsed"] += 1
        if y is not None:
            out["parsed_years"][y] = out["parsed_years"].get(y, 0) + 1
    return out


def coverage_note(p: pd.DataFrame, m: pd.DataFrame) -> dict:
    """The two walls, stated as counts so the UI can print them.

    `prescribed_last_year` is the headline fact: the coach's comments stop, and
    the prescribed series must therefore END there. Extending it into 2026 by
    carrying the last value forward is the single most tempting dishonesty in
    this module, so the year is returned explicitly and the chart is expected to
    stop at it.
    """
    p_years = sorted(int(y) for y in p["year"].dropna().unique()) if len(p) else []
    m_years = sorted(int(y) for y in m["year"].dropna().unique()) if len(m) else []
    return {
        "prescribed_years": p_years,
        "prescribed_last_year": p_years[-1] if p_years else None,
        "measured_years": m_years,
        "measured_last_year": m_years[-1] if m_years else None,
        "prescribed_n": int(len(p)) if len(p) else 0,
        "measured_n": int(len(m)) if len(m) else 0,
        "measured_sessions": int(m["tt"].count() / max(1, len(m))) if len(m) else 0,
    }
