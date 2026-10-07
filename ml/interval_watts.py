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

Below 90 s the class is `under 90s`. Printing follows the athlete's own rule:
seconds while an interval lasts less than a minute, minutes from a minute up —
so fmt_rep() reads `30s` at 30 s and `1:15` at 75 s (exact to the second, and
in minutes), and from a minute up the exact length still travels next to the
median so a 10:00 and a 10:20 rep are never presented as the same effort.

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
# ── rep-length classes: the athlete's own tolerance rule ──────────────────────
# A class holds the effort it was PLANNED as, not the second the meter stopped
# on. The athlete rides 20-minute and 30-minute efforts; the detector and the
# peak meter read them back as 20:03, 20:07, 21:00, 30:07 … and a boundary at
# exactly 20:00 / 30:00 split one workout across two classes (17 Apr rode four
# ~20-minute efforts that landed as one "10-20 min" plus three "20-30 min",
# so each page showed a single interval where two were ridden).
#
# The rule, stated by the athlete: anything UNDER 22 minutes is a 10-20 min
# effort, and the same +2 minutes of tolerance applies at 30 (under 32 min is
# a 20-30 min effort). The short classes keep exact bounds — a 90-second
# sprint, a 5-minute and a 9:50 effort are distinct protocols, and the 5:00 and
# 90-second boundaries carry hundreds of efforts that must not be shuffled.
# Half-open on the right: `lo <= secs < hi`, so exactly 22:00 (1320 s) opens
# "20-30 min" and exactly 32:00 (1920 s) opens "30+ min". Nothing is lost by
# it — every bar and every table row also carries its exact length (fmt_rep),
# so a 20:07 effort is always visible as "20:07" and never only as a class.
REP_CLASSES = [
    ("under 90s", 0, 90),
    ("90s-5min", 90, 300),
    ("5-10 min", 300, 600),
    ("10-20 min", 600, 1320),
    ("20-30 min", 1320, 1920),
    ("30+ min", 1920, np.inf),
]
REP_ORDER = [c[0] for c in REP_CLASSES]

# Short class names for series display: "FTP (10-20)", "VO2MAX (<8 min)" is
# NOT one of these — the under-8-minute multi-rep band is a FAMILY rule
# (rule A files it as VO2MAX), while these name the rep-length class inside
# any family. Grouping always uses REP_CLASSES above; these strings never
# classify anything.
CLS_SHORT = {
    "under 90s": "<90s",
    "90s-5min": "90s-5",
    "5-10 min": "5-10",
    "10-20 min": "10-20",
    "20-30 min": "20-30",
    "30+ min": ">30",
}


def series_name(family, cls) -> str:
    """Display name of one comparable series: family + length bin.

    "FTP (10-20)" is the 10-to-22-minute threshold work (the tolerance rule),
    "FTP (20-30)" the 22-to-32-minute work, "FTP (>30)" everything longer.
    """
    return f"{family} ({CLS_SHORT.get(cls, cls)})"

# `'` is MINUTES and `"` is SECONDS in these comments. They are separate
# alternatives and are never merged into one unit group: reading 10" as ten
# minutes instead of ten seconds inflates a sprint session by 60x, and it is a
# mistake this module has already made once.
_MIN_UNIT = r"(?:'|’|´|min)"
_SEC_QUOTE = r"(?:\"|”|″)"
_SEC_WORD = r"(?:s|s'|seg)"

# ── reading a rep out of the coach's sentence ─────────────────────────────────
# A rep marker: "2x20'", '8x30"', "4x45s", "4x1'30"". The mixed minute-second
# form comes FIRST in the alternation so 1'30" is read as ninety seconds and
# not as one minute plus a stray 30".
REPMARK_RE = re.compile(
    r"(?P<n>\d{1,2})\s*[x*×]\s*"
    r"(?:(?P<mmin>\d{1,3})\s*" + _MIN_UNIT + r"\s*(?P<msec>\d{1,2})\s*"
    + _SEC_QUOTE + r""
    r"|(?P<mins>\d{1,3})\s*" + _MIN_UNIT + r""
    r"|(?P<qsec>\d{1,3})\s*" + _SEC_QUOTE + r""
    r"|(?P<sec>\d{1,2})\s*" + _SEC_WORD + r")",
    re.IGNORECASE)

# Everything after one of these is recovery, warm-up or the rest of the ride,
# so the window in which a rep's target wattage may be searched ENDS here —
# rather than trying to reject individual watt figures afterwards. "3x30" MAX
# amb 4'30" recup entre intervals. Completar fins 4h entre 155-165w" must never
# let a 30-second rep borrow the endurance pace written a sentence later, and
# "4x15' pujada (10' recup -150w)" must never lend its watts to the rep.
_CUE_RE = re.compile(
    r"recup|recuperaci|repos|descans|\btornada\b|completar|cool\s?down|"
    r"\bcd\b|\bmd\b|\(\s*\d{1,3}\s*['\"]?\s*r\s*\)",
    re.IGNORECASE)

# A wattage in the coach's own notation: "265-280w", "300w", "+700W". Cadence
# ("105-110rpm"), pulse ("155-165pols") and speed ("10k/h") carry no trailing
# w, so none of them can match — that is the whole reason for the suffix.
_WATT_RE = re.compile(r"(?P<w1>\d{2,4})\s*(?:-\s*(?P<w2>\d{2,4})\s*)?w\b",
                      re.IGNORECASE)

# A length written INSIDE the window: 4', 30", 1'30". It says which sub-block
# of a rep a wattage belongs to, which is how "2x20' ... (4'265-280w+1' suau x4
# cops)" is read as a 20-minute target instead of being refused.
_DUR_RE = re.compile(
    r"(?:(?P<mm>\d{1,3})\s*" + _MIN_UNIT + r"\s*(?P<mss>\d{1,2})\s*"
    + _SEC_QUOTE + r""
    r"|(?P<m>\d{1,3})\s*" + _MIN_UNIT + r""
    r"|(?P<s>\d{1,3})\s*" + _SEC_QUOTE + r")",
    re.IGNORECASE)

MEASURED_RE = re.compile(
    r"(?P<n>\d{1,2})\s*x\s*"
    r"(?:(?P<mi>\d{1,3})\s*m(?P<s>\d{0,2})\s*s?"
    r"|(?P<sec>\d{1,3})\s*s)"
    r"\s*(?P<w>\d{2,4})\s*w",
    re.IGNORECASE)

# intervals.icu's peak-meter: the highest average power the rider SUSTAINED in
# that session, reported together with the window it was sustained over.
#
# This is the third source, and for long efforts it is the only good one. The
# auto-detected `interval_summary` reports what is UNUSUAL inside a ride, so on
# an FTP session it fills up with 10-second spikes and 1-5 minute rolling
# sections and almost never with the set that was actually ridden. Measured on
# this file: across 659 detected efforts the most common lengths are 10-14 s and
# 55-84 s, and exactly ONE detected effort in the whole file sits between 19 and
# 21 minutes. The peak-meter, by contrast, has a value on all 27 FTP sessions of
# 2026 and reports its own duration.
#
# It is NOT a prescription and NOT a validated test. It is the best sustained
# effort the head unit saw, which is a real measurement of something real and
# nothing more than that. Named "peak" throughout so it is never mistaken for a
# threshold reading.
BEST_W_COL = "icu_pm_ftp_watts"
BEST_S_COL = "icu_pm_ftp_secs"

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
    """Exact rep length, in the athlete's rule: seconds under a minute, m:ss
    from a minute up — never a bare class, never a rounded minute.

    m:ss keeps both halves of what is asked for at once: it reads in MINUTES
    for anything a minute long or more, and it stays exact to the second, so a
    19:59 effort never prints as "20 min" on a bar label or a hover. Under a
    minute the seconds are already the exact and the natural unit, so they are
    printed bare (`30s`).
    """
    if secs is None or (isinstance(secs, float) and np.isnan(secs)):
        return "—"
    secs = int(round(float(secs)))
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    return f"{m}:{s:02d}" if s else f"{m}:00"


def fmt_axis_dur(secs) -> str:
    """Duration-axis label: seconds under a minute, minutes above it.

    The athlete's rule, asked for directly and twice: a duration reads in
    SECONDS while it lasts less than a minute and in MINUTES from there up —
    `4824 s` on a tick is what was rejected. From ten minutes up the minutes
    are the NEAREST 5 (half up) — the same nominal rule `dur_bucket` files
    classes by — so a 20:07 effort ticks as "20 min" and a 24:47 as "25 min"
    instead of every measured wobble getting its own tick. Below ten minutes
    the minutes stay whole halves-DOWN, so a 30 s rep is never merged and a
    5:10 never reads as anything but "5 min". Exactness is not traded away:
    the exact measured length stays on the hover and in `duration`; only the
    tick is rounded, and `axis_labels()` refuses to let two ticks round to the
    same word.
    """
    if secs is None or (isinstance(secs, float) and np.isnan(secs)):
        return "—"
    secs = float(secs)
    if secs < 60:
        return f"{secs:.0f} s"
    if secs >= 600:
        return f"{int(np.floor(secs / 300.0 + 0.5) * 5):d} min"
    return f"{int(np.ceil(secs / 60.0 - 0.5)):d} min"


def _mss(secs: float) -> str:
    """m:ss — what a colliding minute tick falls back to."""
    m, s = divmod(int(round(float(secs))), 60)
    return f"{m}:{s:02d}" if s else f"{m}:00"


def axis_labels(seqs) -> list:
    """Minute labels for a whole duration axis; m:ss only where minutes clash.

    Two points four seconds apart both become "5 min" and the axis would
    then carry two ticks reading identically — a number nobody can tell
    apart. Those points fall back to m:ss (5:01 / 5:05) so every tick keeps
    its own name. Seconds under a minute are already exact and never need
    it, so those fall back to the exact second count instead.
    """
    vals = [None if (s is None or (isinstance(s, float) and np.isnan(s)))
            else float(s) for s in seqs]
    labs = ["—" if v is None else fmt_axis_dur(v) for v in vals]
    count = {}
    for lab in labs:
        count[lab] = count.get(lab, 0) + 1
    return [(fmt_rep(v) if (v is not None and count[lab] > 1 and v < 60)
             else _mss(v) if (v is not None and count[lab] > 1) else lab)
            for v, lab in zip(vals, labs)]


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


def _dur_secs(m) -> int:
    """Seconds for one length token: 1'30" -> 90, 4' -> 240, 30" -> 30."""
    if m.group("mm"):
        return int(m.group("mm")) * 60 + int(m.group("mss"))
    if m.group("m"):
        return int(m.group("m")) * 60
    if m.group("s"):
        return int(m.group("s"))
    return 0


def _rep_secs(m):
    """Seconds for the rep marker itself, or None if the form is unreadable."""
    if m.group("mmin"):
        return int(m.group("mmin")) * 60 + int(m.group("msec"))
    if m.group("mins"):
        return int(m.group("mins")) * 60
    if m.group("qsec"):
        return int(m.group("qsec"))
    if m.group("sec"):
        return int(m.group("sec"))
    return None


def _window(text: str, m, nxt) -> str:
    """Text this rep may claim: up to the next rep marker, then to the first
    recovery or rest-of-ride cue. Nothing outside the window is ever read."""
    seg = text[m.end(): nxt.start() if nxt is not None else len(text)]
    cue = _CUE_RE.search(seg)
    return seg[:cue.start()] if cue else seg


# A wattage written as a CEILING rather than a target: "per sota 265w",
# "sense passar 200w", "sense superar els 210w" — all Catalan for "below /
# without passing / without exceeding". Those numbers are real and are the
# coach's own mark, but they are a top edge, so they are tagged and reported
# as ceilings instead of being presented as an exact target.
_CEIL_RE = re.compile(
    r"per sota|\bsota\b|sense passar|sense sobrepassar|sense superar|"
    r"menys de|no passar|\bmàxim\b|\bmaxim\b", re.IGNORECASE)


def _is_ceiling(seg: str, pos: int) -> bool:
    return bool(_CEIL_RE.search(seg[max(0, pos - 40):pos]))


def _target_for(seg: str, rep_secs: int):
    """(status, lo, hi, basis, ceiling) — which wattage belongs to this rep.

    direct : the coach wrote the wattage straight after the rep.
    inside : the rep is a block and the wattage sits on one of its sub-lengths,
             so the sub-length closest to (and never longer than) the rep wins.
    Refused, never guessed, when there is no wattage (a cadence, a pulse or an
    all-out set) or when the only candidates are structures this parser does
    not understand.
    """
    atoms = []
    for a in _WATT_RE.finditer(seg):
        lo, hi = float(a.group("w1")), float(a.group("w2") or a.group("w1"))
        if hi < lo:
            lo, hi = hi, lo
        durs = [_dur_secs(d) for d in _DUR_RE.finditer(seg[:a.start()])]
        atoms.append({"lo": lo, "hi": hi,
                      "dur": durs[-1] if durs else None,
                      "pos": a.start()})
    if not atoms:
        return "no_watts", None, None, None, False
    bare = [a for a in atoms if a["dur"] is None]
    if bare:
        a = bare[0]
        return "ok", a["lo"], a["hi"], "direct", _is_ceiling(seg, a["pos"])
    # A block LONGER than the rep cannot be part of this rep, which means the
    # window spans more than the rep does and nothing inside it can be
    # attributed safely. Refuse rather than choose between them.
    if any(a["dur"] and a["dur"] > rep_secs for a in atoms):
        return "ambiguous", None, None, None, False
    inside = [a for a in atoms
              if a["dur"] is not None and 0 < a["dur"] <= rep_secs]
    if not inside:
        return "ambiguous", None, None, None, False
    best = min(inside, key=lambda a: max(a["dur"] / rep_secs,
                                         rep_secs / a["dur"]))
    ratio = max(best["dur"] / rep_secs, rep_secs / best["dur"])
    # One wattage written inside the block IS that rep's target even when its
    # sub-length is short (4' inside a 20-minute rep). Several candidates that
    # are all far from the rep length are a structure we do not understand, and
    # those are refused rather than guessed.
    if ratio <= 2.5 or len(inside) == 1:
        return ("ok", best["lo"], best["hi"], "inside",
                _is_ceiling(seg, best["pos"]))
    return "ambiguous", None, None, None, False


def _outcomes(desc) -> list:
    """One dict per rep marker, so a refusal keeps its reason.

    status: ok | no_watts | ambiguous | too_short | implausible
    """
    if _blank(desc):
        return []
    text = main_set(desc)
    marks = list(REPMARK_RE.finditer(text))
    out = []
    for i, m in enumerate(marks):
        secs = _rep_secs(m)
        if not secs:
            out.append({"status": "ambiguous"})
            continue
        seg = _window(text, m, marks[i + 1] if i + 1 < len(marks) else None)
        status, lo, hi, basis, ceil = _target_for(seg, secs)
        if status != "ok":
            out.append({"status": status})
        elif secs < 30:
            out.append({"status": "too_short", "secs": secs, "lo": lo})
        elif lo < 100:
            out.append({"status": "implausible", "secs": secs, "lo": lo})
        else:
            out.append({"status": "ok", "n": int(m.group("n")), "secs": secs,
                        "lo": lo, "hi": hi, "basis": basis,
                        "ceiling": bool(ceil)})
    return out


def parse_prescribed(desc) -> list:
    """All 'NxD' reps with a goal wattage in one description.

    Returns [(n_reps, seconds, watts_low, watts_high), ...]. An empty list means
    "nothing unambiguous here", which is a normal outcome and is counted by
    audit() rather than hidden.
    """
    return [(o["n"], o["secs"], o["lo"], o["hi"])
            for o in _outcomes(desc) if o["status"] == "ok"]


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


# Physiology of one effort, by the user's own rule:
#   >= 10 min  -> FTP work
#   <  10 min  -> VO2MAX work (1-6 and 6-10 min both count as VO2MAX)
# Sub-minute efforts are short work and fall on the VO2MAX side of the split;
# they are never pooled with the long ones, only labelled beside them.
FTP_MIN_S = 600


def phys_class(secs: float) -> str:
    """FTP for efforts of 10 minutes or more, VO2MAX below that."""
    return "FTP" if float(secs) >= FTP_MIN_S else "VO2MAX"


def _isolated_push_mask(out: pd.DataFrame) -> pd.Series:
    """True for an effort over one minute that is a lone push.

    The user's rule: an effort longer than a minute only counts when it is part
    of several intervals - either several reps inside the one detected effort
    (`n >= 2`, "3x 15m") or several efforts on the same day. A single 15-minute
    effort alone on its day is one isolated push and is discarded.
    """
    big = out["secs"] > 60
    day_n = out.groupby("date")["secs"].transform("size")
    return big & (out["n"] < 2) & (day_n < 2)


def measured(df: pd.DataFrame) -> pd.DataFrame:
    """One row per DETECTED EFFORT. Rep grain, not session grain.

    This is deliberately a different grain from prescribed(), which is session
    grain. The two are never joined: a detected effort is a thing the head unit
    saw, and a prescribed rep is a thing a human wrote down.

    Two of the user's rules are applied here, before any chart sees the data:
    an effort over one minute that is the only effort of its day (and carries
    a single rep) is an isolated push and is discarded; every kept effort gets
    its physiology label - FTP at ten minutes and above, VO2MAX below.
    """
    cols = ["date", "year", "tt", "n", "secs", "cls", "phys", "w"]
    if df is None or not len(df):
        return pd.DataFrame(columns=cols)
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
        d = df.at[i, "date"] if "date" in df.columns else None
        for n, secs, w in parse_measured(src.at[i]):
            rows.append({
                "date": d,
                "year": d.year if hasattr(d, 'year') else None,
                "tt": str(label.at[i]).strip(),
                "n": n, "secs": secs, "cls": rep_class(secs), "w": w,
            })
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    out = out[~_isolated_push_mask(out)].reset_index(drop=True)
    out["phys"] = [phys_class(s) for s in out["secs"]]
    out.attrs["isolated_discarded"] = int(
        _isolated_push_mask(pd.DataFrame(rows)).sum())
    return out


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


# ── ley de potencias ─────────────────────────────────────────────────────────
# Two fits of the same power-duration law, both reported side by side because
# they answer slightly different questions and neither may hide behind the other:
#   CP model     P = CP + W'/t   -> critical power (W) and anaerobic work
#                                   capacity W' (kJ). Linear in 1/t: the fit is
#                                   a least-squares line of P against 1/t, so CP
#                                   is the intercept and W' the slope (W·s).
#   power law    P = a · t^-b    -> the exponent b. Fit in log-log space, which
#                                   is the only way a straight line means
#                                   anything on a power-duration curve.
# The input is the BEST watts observed at each duration (mean-maximal by
# duration), never an average across durations: a power-duration law fitted to
# averages fits nothing. n is the number of distinct durations that fed the
# fit, and R^2 is reported for both models so neither can be oversold.
MIN_FIT_PTS = 4


def power_law(m: pd.DataFrame, tt: str | None = None,
              min_pts: int = MIN_FIT_PTS) -> dict:
    """Fit the ley de potencias to the best watts per duration.

    Returns a dict with both models, each carrying its parameters, R^2 and the
    duration range it was fitted on. `ok` is False when there are fewer than
    `min_pts` distinct durations - a two-point line is not a law, and the
    caller must show that instead of a fitted number.
    """
    empty = {"ok": False, "reason": "not enough distinct durations",
             "n": 0, "cp": None, "w_prime_kj": None, "r2_cp": None,
             "a": None, "b": None, "r2_law": None,
             "t_lo": None, "t_hi": None}
    if m is None or not len(m) or "secs" not in m.columns:
        empty["reason"] = "no measured efforts"
        return empty
    s = m if tt is None else m[m["tt"] == tt]
    if not len(s):
        empty["reason"] = "no efforts for this type"
        return empty
    s = s.copy()
    s["secs"] = pd.to_numeric(s["secs"], errors="coerce")
    s["w"] = pd.to_numeric(s["w"], errors="coerce")
    s = s.dropna(subset=["secs", "w"])
    s = s[(s["secs"] > 0) & (s["w"] > 0)]
    if not len(s):
        empty["reason"] = "no usable efforts"
        return empty
    # best watts per distinct duration - the mean-maximal curve, one point each
    best = (s.groupby("secs", as_index=False)["w"].max()
            .sort_values("secs"))
    t = best["secs"].to_numpy(float)
    p = best["w"].to_numpy(float)
    empty["n"] = int(len(best))
    empty["t_lo"], empty["t_hi"] = int(t.min()), int(t.max())
    if len(best) < min_pts:
        return empty

    def _r2(y, yhat):
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot <= 0:
            return None
        return 1.0 - ss_res / ss_tot

    out = dict(empty)
    # CP model: P = CP + W'/t  (linear in x = 1/t)
    x = 1.0 / t
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        coef, *_ = np.linalg.lstsq(A, p, rcond=None)
        slope, intercept = float(coef[0]), float(coef[1])
        out["cp"] = round(intercept, 1)
        out["w_prime_kj"] = round(slope * 1.0 / 1000.0, 1)   # W·s -> kJ
        out["r2_cp"] = (round(_r2(p, A @ coef), 3)
                        if _r2(p, A @ coef) is not None else None)
    except Exception:
        pass
    # power law: P = a · t^-b  (linear in log-log space)
    try:
        lx, lp = np.log(t), np.log(p)
        B = np.vstack([lx, np.ones_like(lx)]).T
        coef2, *_ = np.linalg.lstsq(B, lp, rcond=None)
        out["b"] = round(-float(coef2[0]), 3)
        out["a"] = round(float(np.exp(coef2[1])), 1)
        out["r2_law"] = (round(_r2(lp, B @ coef2), 3)
                         if _r2(lp, B @ coef2) is not None else None)
    except Exception:
        pass
    out["ok"] = True
    out["reason"] = ""
    return out


# ── putting back together the effort the detector cut into pieces ─────────────
# The detector segments a ride by power and cadence stability, so ONE long
# block comes back as several rows with short recovery rows between them: on
# 30 Sep 2026 the meter held 244 W over 22:00, while the detector's own rows
# for the two intervals behind that reading are 158 + 285 + 384 + 180 s
# (separated by 36, 87 and 27 s) and 882 + 324 s. A meter that reports ONE
# window per ride can therefore never show a rider who did TWO intervals what
# they rode — which is the defect: one bar where two efforts happened.
#
# Every condition below exists to stop the rebuild inventing an effort:
#
#   * the ride's OWN peak window vouches for it: the window is at least
#     REBUILD_MIN_S long and the detector reported nothing in its class —
#     where the detector already told the story whole, nothing is rebuilt;
#   * WORK rows join only across breaks shorter than REBUILD_DIP_S, the WHOLE
#     break counted: a 27 s glance at the computer does not end a 20-minute
#     interval, but 60 s + 60 s of rest does, so consecutive recovery rows
#     accumulate before the decision is made;
#   * the joined run must be within REBUILD_LEN_TOL of the meter's window and
#     within REBUILD_W_TOL of its watts. The meter is what vouches for the
#     rebuild, so a run the meter does not describe is left alone;
#   * no detected effort within REBUILD_NEAR_S of the run, so one effort is
#     never drawn twice — and that already covers the whole-minute class test,
#     because two lengths under the same whole-minute label are at most 60 s
#     apart and 60 < 120.
#
# A rebuilt run is MEASURED from the detector's own rows, so its class is the
# class of its OWN length and never the meter's: 19:17 sits in "10-20 min" and
# 20:07 in "20-30 min" because that is where those lengths sit. The exact
# seconds ride on every bar, so the class is a bucket and never the
# description of the effort.
REBUILD_MIN_S = 600.0        # the sustained floor, same as the peak fill
REBUILD_NEAR_S = 120.0       # a detected effort this close IS this effort
REBUILD_DIP_S = 90.0         # a break this short belongs to the effort
REBUILD_LEN_TOL = 0.15       # run length within 15 % of the meter's window
REBUILD_W_TOL = 0.10         # run watts within 10 % of the meter's watts
REBUILD_SRC = "rebuilt from detector fragments"
PEAK_ROW_SRC = "peak meter (intervals.icu)"


def _iv_runs(rows: pd.DataFrame) -> list:
    """WORK runs of one activity, bridging breaks shorter than REBUILD_DIP_S.

    `rows` is one activity's interval rows in `seq` order. A break is bridged
    only when the WHOLE of it is short, and the break's seconds then sit
    inside the span with their own watts weighted by their own seconds — so a
    run's average is the block's own average, never a mean of means.
    """
    runs, cur = [], None
    dip_s = dip_e = 0.0
    for r in rows.itertuples(index=False):
        try:
            secs = float(getattr(r, "secs", np.nan))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(secs) or secs <= 0:
            continue
        try:
            w = float(getattr(r, "avg_w", np.nan))
        except (TypeError, ValueError):
            w = np.nan
        if not np.isfinite(w):
            w = 0.0
        # The row's own sequence number, when the frame carries one: it lets
        # callers tell a fragment the run CONTAINS from a second effort that
        # merely reads the same length. A bare frame without sequence numbers
        # still joins — the seqs just stay empty.
        try:
            _sq = getattr(r, "oseq", getattr(r, "seq",
                                             getattr(r, "_seq", np.nan)))
            seq = float(_sq)
        except (TypeError, ValueError):
            seq = np.nan
        if not np.isfinite(seq):
            seq = np.nan
        is_work = str(getattr(r, "iv_type", "")).strip().upper() == "WORK"
        if is_work:
            if cur is not None and dip_s > 0:
                if dip_s < REBUILD_DIP_S:          # one effort, continued
                    cur["secs"] += dip_s
                    cur["energy"] += dip_e
                    cur["n_dips"] += 1
                else:                              # two efforts, parted
                    runs.append(cur)
                    cur = None
            dip_s = dip_e = 0.0
            if cur is None:
                cur = {"secs": 0.0, "energy": 0.0, "parts": [],
                       "seqs": [], "n_dips": 0}
            cur["secs"] += secs
            cur["energy"] += secs * w
            cur["parts"].append(secs)
            if np.isfinite(seq):
                cur["seqs"].append(seq)
        elif cur is not None:
            dip_s += secs
            dip_e += secs * w
    if cur is not None:
        runs.append(cur)      # a ride's tail never shortens its last effort
    return [{"secs": r["secs"], "w": r["energy"] / r["secs"],
             "parts": r["parts"], "seqs": r["seqs"], "n_dips": r["n_dips"]}
            for r in runs if r["secs"] > 0]


def rebuilt_efforts(iv: pd.DataFrame, df_all=None) -> pd.DataFrame:
    """Whole efforts the detector cut into pieces — see the note above.

    One row per rebuilt effort with `activity_id, date, day, secs, w, cls,
    parts, pieces, n_dips, peak_w, peak_s`. The meter's own window travels
    beside every rebuild so a reader can check one against the other. Empty
    (with its columns) whenever there is nothing to rebuild, and nothing is
    ever rebuilt without a peak window vouching for it.
    """
    cols = ["activity_id", "date", "day", "secs", "w", "cls", "parts",
            "pieces", "seqs", "n_dips", "peak_w", "peak_s"]
    empty = pd.DataFrame(columns=cols)
    if iv is None or not len(iv) or df_all is None or not len(df_all):
        return empty
    if not {"iv_type", "secs", "avg_w", "activity_id"}.issubset(iv.columns):
        return empty
    if not {"id", BEST_W_COL, BEST_S_COL, "date"}.issubset(df_all.columns):
        return empty

    pk = df_all[["id", BEST_W_COL, BEST_S_COL, "date"]].copy()
    pk["activity_id"] = pk["id"].astype(str)
    pk["pk_w"] = pd.to_numeric(pk[BEST_W_COL], errors="coerce")
    pk["pk_s"] = pd.to_numeric(pk[BEST_S_COL], errors="coerce")
    pk["date"] = pd.to_datetime(pk["date"], errors="coerce")
    pk = pk[pk["pk_w"].notna() & pk["pk_s"].notna() & (pk["pk_w"] > 0)
            & (pk["pk_s"] >= REBUILD_MIN_S) & pk["date"].notna()]
    pk = pk.drop_duplicates("activity_id", keep="first")
    if not len(pk):
        return empty

    frame = pd.DataFrame({
        "_aid": iv["activity_id"].astype(str),
        "iv_type": iv["iv_type"].astype(str),
        "secs": pd.to_numeric(iv["secs"], errors="coerce"),
        "avg_w": pd.to_numeric(iv["avg_w"], errors="coerce"),
        # NOT "_seq": pandas itertuples() renames underscore-led columns
        # ("_seq" arrives as "_4"), so the sequence number travels bare.
        "oseq": (pd.to_numeric(iv["seq"], errors="coerce")
                 if "seq" in iv.columns else np.arange(len(iv))),
    })

    rows = []
    for p in pk.itertuples(index=False):
        g = frame[frame["_aid"] == str(p.activity_id)]
        if not len(g):
            continue
        g = g.sort_values("oseq", kind="stable")
        window = float(p.pk_s)
        work = g.loc[g["iv_type"].str.upper() == "WORK", "secs"].dropna()
        # The anchor: where the detector already reports an effort of about
        # the window's LENGTH (±2 minutes), it told this story whole and
        # nothing is rebuilt. A mere class match is NOT enough: a 683-second
        # fragment shares the 21:00 window's class without being the effort,
        # and anchoring on it both blocked the rebuild and let the meter row
        # draw on top of its own fragment (04 Jul 2026 drew 683 s + 1260 s
        # for one 20-minute effort). Same ±2 minutes the peak fill uses, so
        # the two guards can never disagree about one session.
        if len(work) and bool((np.abs(work.values - window)
                               <= REBUILD_NEAR_S).any()):
            continue
        for run in _iv_runs(g):
            s, w, pk_s, pk_w = run["secs"], run["w"], float(p.pk_s), float(p.pk_w)
            if s < REBUILD_MIN_S:
                continue
            if abs(s - pk_s) > REBUILD_LEN_TOL * pk_s:
                continue
            if abs(w - pk_w) > REBUILD_W_TOL * pk_w:
                continue
            if len(work) and bool((np.abs(work.values - s)
                                   <= REBUILD_NEAR_S).any()):
                continue
            rows.append({
                "activity_id": str(p.activity_id),
                "date": p.date,
                "day": pd.Timestamp(p.date).normalize(),
                "secs": int(round(s)),
                "w": float(w),
                "cls": rep_class(s),
                "parts": len(run["parts"]),
                "pieces": " + ".join(fmt_rep(x) for x in run["parts"]),
                "seqs": tuple(float(x) for x in run["seqs"]),
                "n_dips": int(run["n_dips"]),
                "peak_w": float(pk_w),
                "peak_s": int(round(pk_s)),
            })
    return pd.DataFrame(rows, columns=cols)


# ── per day: the bar-and-line chart ──────────────────────────────────────────
def effort_best(df: pd.DataFrame, iv: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per sustained effort — session grain, plus what was rebuilt.

    The meter's own window is the source: it is the only reading that says
    how long a long effort was actually held, while `measured()` is rep grain
    and mostly reports the 10-second accelerations at the start of each rep.
    Where the detector cut that effort into pieces, `iv` lets the rebuilt runs
    replace the single meter reading — one row per effort actually ridden, so
    a session that held two 20-minute intervals draws two bars instead of the
    meter's one window. `source` says which of the two reported each row.

    The duration travels with the wattage and is NOT assumed. A peak of 258 W
    over 8:00 is a different effort from 188 W over 26:00 and averaging the two
    would be the exact error this module exists to avoid, so `cls` is derived
    per row and callers bucket on it.
    """
    cols = ["date", "day", "src", "tt", "secs", "cls", "w", "n", "source",
            "pieces"]
    if df is None or not len(df):
        return pd.DataFrame(columns=cols)
    if BEST_W_COL not in df.columns or BEST_S_COL not in df.columns:
        return pd.DataFrame(columns=cols)
    label = (df["training_type"].astype(object)
             if "training_type" in df.columns else None)
    keep = _has_label(label) if label is not None else pd.Series(
        True, index=df.index)
    w = pd.to_numeric(df[BEST_W_COL], errors="coerce")
    s = pd.to_numeric(df[BEST_S_COL], errors="coerce")
    ok = keep & w.notna() & s.notna() & (s > 0) & (w > 0)
    # Rebuilt efforts, keyed on the session they came from. A session with any
    # of them draws those instead of its meter window — the runs ARE the
    # window, seen as the efforts that were ridden. `iv=None` (every caller
    # that does not pass the cache) changes nothing at all.
    reb_by = {}
    if iv is not None and len(iv):
        rb = rebuilt_efforts(iv, df)
        if len(rb):
            reb_by = {a: g for a, g in rb.groupby("activity_id", sort=False)}
    rows = []
    for i in df.index[ok]:
        d = df.at[i, "date"] if "date" in df.columns else None
        # Normalised to the calendar day. Two rides on one date can carry
        # different clock times, and leaving them in would put two bars on what
        # the athlete calls one day and make the day-over-day line zigzag for a
        # reason that has nothing to do with training.
        day = pd.to_datetime(d).normalize() if d is not None else None
        base = {
            "date": d,
            "day": day,
            # The index of the session in `df`. Carried explicitly because this
            # frame is rebuilt from a list, so its own index is positional and
            # would silently pick the wrong session's average watts.
            "src": i,
            "tt": str(label.at[i]).strip() if label is not None else "",
            "n": 1,
        }
        act = str(df.at[i, "id"]) if "id" in df.columns else ""
        hits = reb_by.get(act)
        if hits is not None and len(hits):
            for r in hits.itertuples(index=False):
                rows.append({**base, "secs": int(r.secs), "cls": r.cls,
                             "w": float(r.w), "source": REBUILD_SRC,
                             "pieces": r.pieces})
            continue
        rows.append({**base,
                     "secs": int(round(float(s.at[i]))),
                     "cls": rep_class(float(s.at[i])),
                     "w": float(w.at[i]),
                     "source": PEAK_ROW_SRC,
                     "pieces": ""})
    out = pd.DataFrame(rows)
    if len(out):
        # From `day`, not from the raw date string. `day` was parsed one value
        # at a time and is already a datetime, so this does not re-parse a
        # column that may legitimately mix "2026-04-17" and
        # "2026-05-01 08:30:00" — which pd.to_datetime refuses without an
        # explicit format.
        out["year"] = pd.to_datetime(out["day"]).dt.year
    return out


def _effort_by_day(df: pd.DataFrame, with_avg: bool = True,
                   iv: pd.DataFrame | None = None) -> pd.DataFrame:
    """The ONE day-level frame the whole bar chart reads.

    Both `day_series` and `day_options` read this, and that is the point: when
    they were computed separately the picker's count and the number of bars drawn
    disagreed, because one counted sessions and the other drew days. A dropdown
    promising four days and then drawing two bars is exactly the kind of quiet
    mismatch this project refuses to ship.

    One row per BAR, under two rules decided here and nowhere else:

      * several RIDES on one day still give that day ONE bar — its hardest
        effort. The mean of two rides is a third number describing neither,
        and "what did I do on the 14th" is answered by the hardest effort of
        it. `n_sessions` records that the day held more than one ride rather
        than hiding it;
      * ONE ride that made SEVERAL efforts of this class draws every one of
        them. That is 30 Sep 2026: two 20-minute intervals inside one session
        and a meter that reports a single window. Collapsing them to that one
        window is the defect — a day the athlete rode twice is not the same
        thing as an athlete who rode once.

    `day_avg` is the plain mean of that day's bars — the number to follow over
    time — computed strictly within ONE length class, so a 7:30 rep never
    meets a 20:00 one.
    """
    cols = ["day", "date", "src", "tt", "cls", "secs", "w", "n", "n_sessions",
            "day_avg", "avg_w", "dur_s", "year", "source", "pieces"]
    b = effort_best(df, iv)
    if b is None or not len(b):
        return pd.DataFrame(columns=cols)

    avg = dur = None
    if with_avg and "power_avg" in getattr(df, "columns", []):
        avg = pd.to_numeric(df["power_avg"], errors="coerce")
    if "duration_s" in getattr(df, "columns", []):
        dur = pd.to_numeric(df["duration_s"], errors="coerce")

    rows = []
    for (tt, cls, day), grp in b.groupby(["tt", "cls", "day"]):
        # `src` is each session's own index in `df`, so these are the right rows
        # and not merely rows at the same positions. Unique, because one
        # session may now appear twice and its averages must not be counted
        # twice with it.
        idxs = list(dict.fromkeys(grp["src"]))
        n_rides = int(grp["src"].nunique())
        sub = grp if n_rides == 1 else grp.loc[[grp["w"].idxmax()]]
        day_avg = float(grp["w"].mean())
        for top in sub.itertuples(index=False):
            rows.append({
                "day": day,
                "date": top.date,
                "src": int(top.src),
                "tt": tt,
                "cls": cls,
                "secs": int(top.secs),
                "w": float(top.w),
                "n": 1,
                "n_sessions": n_rides,
                # The day's own AVERAGE across every interval of this class it
                # held, kept beside the bars instead of replacing them. It is
                # what a reader tracks over time ("what did 30 Sep average"),
                # while the bars stay the efforts actually ridden — and the two
                # are only equal on a single-effort day.
                "day_avg": day_avg,
                "avg_w": (float(avg.loc[idxs].mean())
                          if avg is not None and idxs else float("nan")),
                "dur_s": (float(dur.loc[idxs].mean())
                          if dur is not None and idxs else float("nan")),
                "year": int(top.year),
                "source": getattr(top, "source", ""),
                "pieces": getattr(top, "pieces", ""),
            })
    return pd.DataFrame(rows).sort_values(["tt", "cls", "day"]).reset_index(
        drop=True)


def day_series(df: pd.DataFrame, tt: str, cls: str,
               with_avg: bool = True,
               iv: pd.DataFrame | None = None) -> pd.DataFrame:
    """Bars for ONE type and ONE rep-length class: one row per effort drawn.

    A day that rode two efforts of this class comes back twice (the two 20-min
    intervals of 30 Sep 2026), each carrying its own exact length, and the
    day's own average watts travels on every one of them so the two can be read
    against each other on the same axis without ever being averaged into one
    number.
    """
    cols = ["day", "date", "src", "tt", "cls", "secs", "w", "n", "n_sessions",
            "day_avg", "avg_w", "dur_s", "year", "source", "pieces"]
    d = _effort_by_day(df, with_avg, iv)
    if d is None or not len(d):
        return pd.DataFrame(columns=cols)
    s = d[(d["tt"] == tt) & (d["cls"] == cls)]
    if not len(s):
        return pd.DataFrame(columns=cols)
    return s.sort_values("day").reset_index(drop=True)


def day_options(df: pd.DataFrame,
                iv: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every (type x rep class) pair the day chart can draw, counted honestly.

    Two counts, because they are no longer the same number and the picker may
    not promise what it does not draw: `days` is distinct calendar days, `bars`
    is what will be drawn (a day that rode two efforts of this class draws
    two). `sessions` counts rides, deduplicated per day so a two-bar day is
    never reported as two sessions.
    """
    cols = ["tt", "cls", "days", "bars", "sessions", "med_w", "med_secs",
            "first", "last", "years"]
    d = _effort_by_day(df, iv=iv)
    if d is None or not len(d):
        return pd.DataFrame(columns=cols)
    bars = d.groupby(["tt", "cls"], as_index=False).size().rename(
        columns={"size": "bars"})
    per_day = d.drop_duplicates(["tt", "cls", "day"])
    day_lvl = per_day.groupby(["tt", "cls"], as_index=False).agg(
        days=("day", "nunique"), sessions=("n_sessions", "sum"))
    g = (d.groupby(["tt", "cls"], as_index=False)
         .agg(med_w=("w", "median"), med_secs=("secs", "median"),
              first=("day", "min"), last=("day", "max"),
              years=("year", "nunique"))
         .merge(day_lvl, on=["tt", "cls"], how="left")
         .merge(bars, on=["tt", "cls"], how="left"))
    g = g[cols]
    return g.sort_values(["years", "bars"], ascending=False).reset_index(
        drop=True)


# ── the power curve, built from the file ─────────────────────────────────────
# Canonical durations of a power-duration curve. A target with no effort inside
# its window is SKIPPED, never interpolated: a gap the athlete never filled is a
# gap on the chart.
PC_TARGETS = (5, 10, 20, 30, 45, 60, 90, 120, 180, 300, 420, 600, 780, 900,
              1200, 1800, 2400, 3000)
PC_WINDOW = 0.12                     # ±12 % around the target duration


def power_curve(df: pd.DataFrame, window: float = PC_WINDOW
                ) -> pd.DataFrame:
    """Best observed watts near each canonical duration — the power curve.

    Built from what is already in the training file, so it works with no API
    call, no credentials and no `data/power_curve.csv`: that file is produced
    by a script which does not exist in this repo, which is why the chart used
    to render as "unavailable" on every deploy.

    One REAL effort stands behind every point: the best watts among the
    detected efforts whose length sits within `window` of the target. The
    exact length, the date, and how many efforts fell inside the window travel
    with the row so the chart can print them. Nothing is interpolated between
    points and nothing is extrapolated past the longest effort ridden — the
    smooth curve across all durations is the power-duration law FIT, and that
    one is labelled as a fit where it is drawn.
    """
    cols = ["secs", "watts", "duration", "target", "n", "date", "year"]
    M = measured(df)
    if M is None or not len(M):
        return pd.DataFrame(columns=cols)
    M = M.dropna(subset=["secs", "w"]).copy()
    M["w"] = pd.to_numeric(M["w"], errors="coerce")
    M = M[M["w"] > 0]
    if not len(M):
        return pd.DataFrame(columns=cols)

    rows = []
    for target in PC_TARGETS:
        lo, hi = target * (1.0 - window), target * (1.0 + window)
        c = M[(M["secs"] >= lo) & (M["secs"] <= hi)]
        if not len(c):
            continue
        best = c.loc[c["w"].idxmax()]
        rows.append({
            "secs": int(round(float(best["secs"]))),
            "watts": float(best["w"]),
            "duration": fmt_rep(float(best["secs"])),
            "target": int(target),
            "n": int(len(c)),
            "date": best["date"],
            "year": int(best["year"]) if pd.notna(best["year"]) else None,
        })
    out = pd.DataFrame(rows, columns=cols)
    if len(out):
        # Two targets can land on the same exact length (±12 % windows touch at
        # the edges). A repeated tick position makes the axis ambiguous, so the
        # point with more efforts behind it wins.
        out = (out.sort_values(["secs", "n"], ascending=[True, False])
               .drop_duplicates("secs").sort_values("secs")
               .reset_index(drop=True))
    return out


# ── honesty surfaces ─────────────────────────────────────────────────────────
def audit(df: pd.DataFrame) -> dict:
    """What the prescription parser saw, kept, and dropped. Never a bare rate.

    `skipped_steady` is separated from `skipped_unread` on purpose: a
    description with no repetition in it is a steady endurance ride and has no
    intervals by definition, which is a correct non-match and not a loss.
    Reporting one combined number would make a healthy parser look like it lost
    60% of the file.

    `skipped_unread` is then SPLIT, because "refused" hides three very
    different facts: no wattage was written (a pulse or an all-out set), the
    wattage is there but the rep is under the 30 s interval floor, or the
    wattage could not be attributed to this rep. The three add up to
    `skipped_unread`, so the balance the tests check still holds.

    The note window (`note_first` / `note_last` / `rides_after_last_note`) is
    reported too: this file's coach notes stop on their own, and a reader must
    be able to see that rather than infer it from an empty chart.
    """
    out = {"seen": 0, "parsed": 0, "skipped_steady": 0, "skipped_unread": 0,
           "skipped_unlabelled": 0, "years": {}, "parsed_years": {},
           "examples_unread": [], "refused_no_watts": 0,
           "refused_too_short": 0, "refused_implausible": 0,
           "refused_ambiguous": 0, "reads_direct": 0, "reads_inside": 0,
           "reads_ceiling": 0,
           "note_first": None, "note_last": None,
           "rides_after_last_note": 0}
    if df is None or not len(df):
        return out
    if "WorkoutDescription" not in df.columns:
        return out
    desc = df["WorkoutDescription"]
    label = (df["training_type"].astype(object)
             if "training_type" in df.columns else None)
    yrs = df["date"].dt.year if "date" in df.columns else None
    note_dates = []

    for i in df.index:
        d = desc.at[i]
        if _blank(d):
            continue
        out["seen"] += 1
        y = int(yrs.at[i]) if yrs is not None and pd.notna(yrs.at[i]) else None
        if y is not None:
            out["years"][y] = out["years"].get(y, 0) + 1
        if "date" in df.columns and pd.notna(df.at[i, "date"]):
            note_dates.append(df.at[i, "date"])
        labelled = True if label is None else bool(_has_label(label.loc[[i]]).iloc[0])
        outs = _outcomes(d)
        if not outs:
            # No repetition syntax at all -> a steady ride, correctly no intervals.
            out["skipped_steady"] += 1
            continue
        oks = [o for o in outs if o["status"] == "ok"]
        if not oks:
            out["skipped_unread"] += 1
            reason = _refusal_reason(outs)
            out[f"refused_{reason}"] += 1
            if len(out["examples_unread"]) < 8:
                out["examples_unread"].append(str(d)[:150])
            continue
        if not labelled:
            out["skipped_unlabelled"] += 1
            continue
        out["parsed"] += 1
        if all(o.get("basis") == "direct" for o in oks):
            out["reads_direct"] += 1
        else:
            out["reads_inside"] += 1
        if any(o.get("ceiling") for o in oks):
            # Overlaps direct/inside on purpose: a ceiling is still read from
            # where it was written, it is only the KIND of number that differs.
            out["reads_ceiling"] += 1
        if y is not None:
            out["parsed_years"][y] = out["parsed_years"].get(y, 0) + 1

    if note_dates:
        out["note_first"] = min(note_dates)
        out["note_last"] = max(note_dates)
        if "date" in df.columns:
            out["rides_after_last_note"] = int(
                (df["date"] > out["note_last"]).sum())
    return out


def _refusal_reason(outs) -> str:
    """The most informative reason among the refused reps, in priority order."""
    have = {o.get("status") for o in outs}
    for key in ("ambiguous", "too_short", "implausible", "no_watts"):
        if key in have:
            return key
    return "no_watts"


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
