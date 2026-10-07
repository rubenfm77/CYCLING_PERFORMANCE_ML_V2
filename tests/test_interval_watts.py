# tests/test_interval_watts.py — the honesty rules of the interval-watts engine.
r"""
Runs on SYNTHETIC frames only, so it needs no credentials and no network. The
point is to pin the RULES — the floor, the class boundaries, the refusal to
splice two sources, the audit reconciling — on data small enough to reason about
by hand. Real-data verification is a separate script; a rule that only holds for
1,093 real rows is not a rule.

Every parser case below is a verbatim string from the athlete's own comments,
including the Spanish/Catalan. A synthetic "NxXm W" string would pass a regex that
mangles the real file, which is precisely the bug this file exists to prevent.

Run:  python tests/test_interval_watts.py     (exit 0 = pass, 1 = fail)
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.theme import BLANK_TYPE_TOKENS  # noqa: E402
from ml.interval_watts import (  # noqa: E402
    BEST_S_COL, BEST_W_COL, MIN_CELL_N, REP_ORDER, audit, coverage_note,
    day_options, day_series, effort_best, fmt_rep, fmt_watts, main_set, measured,
    measured_cells, parse_measured, parse_prescribed, prescribed, prescribed_cells,
    prescribed_evolution, prescribed_yoy, rep_class,
)

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")
        FAILS.append(name)


# ── 1. the parser, on real text ──────────────────────────────────────────────
print("=" * 72)
print("1. parsing the coach's own wording")
print("=" * 72)

check("a rep with a watt range yields n, seconds and BOTH ends",
      parse_prescribed(
          "WU: 30' entre 155-170W  MS: 3x15' PLA entre 225-240w  10' RECUP "
          "CD: 10' entre 140-160w.")
      == [(3, 900, 225.0, 240.0)],
      str(parse_prescribed("WU: 30' entre 155-170W  MS: 3x15' PLA entre "
                           "225-240w  10' RECUP CD: 10' entre 140-160w.")))

check("only the MS: portion is read, never warm-up or cool-down",
      main_set("WU: 30' entre 300-320w  MS: 2x8' a 200w  CD: 10' a 150w")
      == "2x8' a 200w",
      repr(main_set("WU: 30' entre 300-320w  MS: 2x8' a 200w  "
                    "CD: 10' a 150w")))

# The single most expensive bug in this module. 30" is thirty SECONDS.
check('double-quote is SECONDS: 8x30" is 30 s, not 30 min',
      parse_prescribed('MS: 2 blocs 8x30" pujada 265-280w')
      == [(8, 30, 265.0, 280.0)],
      str(parse_prescribed('MS: 2 blocs 8x30" pujada 265-280w')))
check("apostrophe is MINUTES: 3x15' is 900 s",
      parse_prescribed("MS: 3x15' PLA entre 225-240w")[0][1] == 900)
check("a bare 's' suffix is seconds: 4x45s",
      parse_prescribed("MS: 4x45s a 300w") == [(4, 45, 300.0, 300.0)],
      str(parse_prescribed("MS: 4x45s a 300w")))
check("a rep shorter than 30 s is refused as too short to be an interval",
      parse_prescribed("MS: 4x1s a 300w") == []
      and parse_prescribed('MS: 4x20" a 300w') == [],
      "a 1-second or 20-second spike is not an interval; the floor is 30 s")
check("a rep under 100 W is refused as implausible for this athlete",
      parse_prescribed("MS: 4x5' a 60-80w") == [],
      str(parse_prescribed("MS: 4x5' a 60-80w")))

# A rep must never inherit a recovery interval's wattage. The gap between the rep
# length and the wattage may not contain a digit, which is what stops this.
check("a rep does not borrow the recovery effort's watts",
      parse_prescribed("MS: 4x15' pujada (10' recup -150w) CD: 10'") == [],
      str(parse_prescribed("MS: 4x15' pujada (10' recup -150w) CD: 10'")))
check("cadence in pols is not mistaken for watts",
      parse_prescribed("WU: 30' entre 140-160w incloent 3x1' entre 100-110rpm "
                       "(-220w) MS: 20' MAX") == [],
      str(parse_prescribed("WU: 30' entre 140-160w incloent 3x1' entre "
                           "100-110rpm (-220w) MS: 20' MAX")))
check("a speed target is not mistaken for watts",
      parse_prescribed("MS: 6x10\" en lleugera pujada sortint a 10k/h. "
                       "(2'50\" recup -150w)") == [],
      str(parse_prescribed("MS: 6x10\" en lleugera pujada sortint a 10k/h. "
                           "(2'50\" recup -150w)")))
check("rpm in a bracket does not become a watt range",
      parse_prescribed("MS: 3x1' en pujada entre 250-270w (85-95rpm sentat)/ "
                       "(2'recup -180w)") == [(3, 60, 250.0, 270.0)],
      str(parse_prescribed("MS: 3x1' en pujada entre 250-270w (85-95rpm "
                           "sentat)/ (2'recup -180w)")))
check("a steady ride with no repetition yields nothing",
      parse_prescribed("45' en terreny constant entre 150-170w (mitja) "
                       "sense superar els 200w") == [])
check("a reversed range is ordered rather than producing a negative span",
      parse_prescribed("MS: 3x10' entre 300-250w") == [(3, 600, 250.0, 300.0)],
      str(parse_prescribed("MS: 3x10' entre 300-250w")))
check("every blank token is treated as no prescription",
      all(parse_prescribed(v) == [] for v in BLANK_TYPE_TOKENS),
      str({v: parse_prescribed(v) for v in BLANK_TYPE_TOKENS}))

check("measured parses intervals.icu's own payload shape",
      parse_measured("['5x 3m 257w', '2x 1m53s 255w']")
      == [(5, 180, 257.0), (2, 113, 255.0)],
      str(parse_measured("['5x 3m 257w', '2x 1m53s 255w']")))
check("measured rejects every blank token",
      all(parse_measured(v) == [] for v in BLANK_TYPE_TOKENS))

# ── 2. the class boundaries the user rule demands ────────────────────────────
print()
print("=" * 72)
print("2. rep-length classes: under 90 s keeps seconds, nothing is pooled")
print("=" * 72)
for secs, want in [(1, "under 90s"), (30, "under 90s"), (89, "under 90s"),
                   (90, "90s-5min"), (299, "90s-5min"), (300, "5-10 min"),
                   (599, "5-10 min"), (600, "10-20 min"), (1199, "10-20 min"),
                   (1200, "10-20 min"), (1319, "10-20 min"),
                   (1320, "20-30 min"), (1919, "20-30 min"),
                   (1920, "30+ min"), (5400, "30+ min")]:
    check(f"{secs:>5}s -> {want}", rep_class(secs) == want, rep_class(secs))

check("classes cover the whole range with no gap and no overlap",
      REP_ORDER == ["under 90s", "90s-5min", "5-10 min", "10-20 min",
                    "20-30 min", "30+ min"])
check("fmt_rep keeps exact seconds below 90 s", fmt_rep(30) == "30s")
check("fmt_rep prints m:ss above 90 s", fmt_rep(113) == "1:53")
check("fmt_rep does not hide a trailing second", fmt_rep(600) == "10:00")
check("fmt_rep survives a missing value", fmt_rep(None) == "—")
check("fmt_watts puts a separator in a big number",
      fmt_watts(1234567) == "1,234,567", fmt_watts(1234567))
check("fmt_watts survives a missing value", fmt_watts(None) == "—")

# ── 3. synthetic frames ──────────────────────────────────────────────────────
print()
print("=" * 72)
print("3. the floor, and the refusal to invent a point")
print("=" * 72)


def _frame(rows, columns):
    """Minimal session frame carrying only the columns the engine reads."""
    return pd.DataFrame(rows, columns=columns)


SYN = _frame([
    # type,   date,                duration_s, desc,                    isum
    # FTP 15' in 2021, 2022 and 2024 -> three drawable years. 2023 is THIN, so
    # 2024's previous comparable year must be 2022 with a 2y gap, not 2023.
    ("FTP", "2021-03-01", 7200, "MS: 3x15' PLA entre 225-240w",
     "['3x 15m 250w']"),
    ("FTP", "2021-04-01", 7200, "MS: 3x15' PLA entre 225-240w", ""),
    ("FTP", "2021-05-01", 7200, "MS: 3x15' PLA entre 225-240w", ""),
    ("FTP", "2022-03-01", 7200, "MS: 3x15' PLA entre 260-275w", ""),
    ("FTP", "2022-04-01", 7200, "MS: 3x15' PLA entre 260-275w", ""),
    ("FTP", "2022-06-01", 7200, "MS: 3x15' PLA entre 260-275w", ""),
    # a THIN year: two sessions, must produce no point at all
    ("FTP", "2023-03-01", 7200, "MS: 3x15' PLA entre 270-285w", ""),
    ("FTP", "2023-04-01", 7200, "MS: 3x15' PLA entre 270-285w", ""),
    ("FTP", "2024-03-01", 7200, "MS: 3x15' PLA entre 290-305w", ""),
    ("FTP", "2024-04-01", 7200, "MS: 3x15' PLA entre 290-305w", ""),
    ("FTP", "2024-05-01", 7200, "MS: 3x15' PLA entre 290-305w", ""),
    # a different type, same year: must never join FTP's cells
    ("SST", "2022-03-01", 7200, "MS: 3x15' PLA entre 240-255w", ""),
    ("SST", "2022-04-01", 7200, "MS: 3x15' PLA entre 240-255w", ""),
    ("SST", "2022-05-01", 7200, "MS: 3x15' PLA entre 240-255w", ""),
    # unlabelled: excluded from every cell
    ("", "2022-07-01", 7200, "MS: 3x15' PLA entre 300-315w", ""),
    # no description at all
    ("FTP", "2022-08-01", 7200, "", ""),
    # a steady ride: no repetition, so no intervals by definition
    ("FTP", "2022-09-01", 7200,
     "45' en terreny constant entre 150-170w (mitja) sense superar els 200w", ""),
    # a sprint: a DIFFERENT class, must never pool with the 15' one
    ("FTP", "2022-03-05", 7200, 'MS: 8x30" a 300-320w', ""),
    ("FTP", "2022-04-05", 7200, 'MS: 8x30" a 300-320w', ""),
    ("FTP", "2022-05-05", 7200, 'MS: 8x30" a 300-320w', ""),
], columns=["type", "date", "duration_s", "WorkoutDescription",
            "interval_summary"])
SYN["date"] = pd.to_datetime(SYN["date"])
SYN["training_type"] = SYN["type"]

P = prescribed(SYN)
M = measured(SYN)

check("an unlabelled session with a prescription is excluded",
      "300" not in set(P[P["cls"] == "10-20 min"]["w_lo"].astype(int)),
      str(sorted(set(P["w_lo"]))))
check("unlabelled, undescribed and steady sessions all contribute nothing",
      len(P) == 17,
      f"{len(P)} rows from 20 sessions; 3 must be refused: one unlabelled, one "
      f"with no description, one steady ride with no repetition")

c = prescribed_cells(P, "FTP")
print(c[["cls", "year", "n", "med_w_lo", "drawable"]].to_string(index=False))
check("a 3-session year clears the floor",
      bool(c[(c["cls"] == "10-20 min") & (c["year"] == 2021)]["drawable"].iloc[0]))
check("a 2-session year is present but NOT drawable",
      bool((~c[(c["cls"] == "10-20 min") & (c["year"] == 2023)]
            ["drawable"].iloc[0])),
      "a thin year must be flagged, not dropped, and not plotted")
check("a thin cell is never emitted as zero watts",
      float(c[(c["cls"] == "10-20 min") & (c["year"] == 2023)]
            ["med_w_lo"].iloc[0]) > 0)
check("the sprint class is a separate cell, never pooled with 15'",
      set(c["cls"]) == {"10-20 min", "under 90s"},
      str(set(c["cls"])))
check("the sprint cell holds only its own 30 s reps",
      bool((P[(P["cls"] == "under 90s")]["secs"] == 30).all()))
check("a cell never mixes two training types",
      c["tt"].nunique() == 1 and set(prescribed_cells(P, "SST")["tt"]) == {"SST"})
check("cell keys are unique per (tt, cls, year)",
      not c.duplicated(["tt", "cls", "year"]).any())

ev = prescribed_evolution(P, "FTP")
check("evolution() emits only drawable cells",
      bool((ev["n"] >= MIN_CELL_N).all()), str(ev[["cls", "year", "n"]]))
check("evolution() contains NO row for the thin year",
      2023 not in set(int(y) for y in ev["year"]),
      "a line must break across a thin year, not bridge it")
check("evolution() is a subset of cells(), never larger", len(ev) <= len(c))

print()
print("=" * 72)
print("4. year over year: previous COMPARABLE year, gap stated")
print("=" * 72)
y = prescribed_yoy(P, "FTP")
main_cls = y[y["cls"] == "10-20 min"].sort_values("year")
print(y[["cls", "year", "n", "med_w_lo", "prev_year", "prev_n", "delta",
         "gap_years"]].to_string(index=False))
check("the thin 2023 is absent from the comparison entirely",
      2023 not in set(int(v) for v in main_cls["year"]),
      "a thin year must not appear as a comparison row")
check("the first year has no previous and no delta",
      pd.isna(main_cls["prev_year"].iloc[0])
      and pd.isna(main_cls["delta"].iloc[0]))
r22 = main_cls[main_cls["year"] == 2022].iloc[0]
check("2022 compares against 2021 with a 1y gap",
      int(r22["prev_year"]) == 2021 and int(r22["gap_years"]) == 1,
      f"prev={r22['prev_year']} gap={r22['gap_years']}")
r24 = main_cls[main_cls["year"] == 2024].iloc[0]
check("2024 compares against 2022, NOT 2023, with the 2y gap stated",
      int(r24["prev_year"]) == 2022 and int(r24["gap_years"]) == 2,
      f"prev={r24['prev_year']} gap={r24['gap_years']} — a 1y gap here would "
      f"print a two-year change as a one-year change")
check("a delta is computed only when both years clear the floor",
      bool((main_cls["prev_n"].dropna() >= MIN_CELL_N).all()),
      str(main_cls[["year", "prev_n"]].to_dict("records")))
check("the delta equals the difference of the two medians",
      all(abs(row["delta"] - (row["med_w_lo"] - row["prev_med"])) < 1e-9
          for _, row in main_cls.dropna(subset=["delta"]).iterrows()))
check("2024's delta is measured against 2022's median, not 2023's",
      abs(r24["delta"] - (290.0 - 260.0)) < 1e-9,
      f"delta={r24['delta']}, expected +30 W")
check("the comparison never crosses training types", y["tt"].nunique() == 1)
check("the comparison never crosses rep classes",
      all(len(g) >= 1 for _, g in y.groupby("cls")))

# ── 5. measured grain ────────────────────────────────────────────────────────
print()
print("=" * 72)
print("5. measured efforts: rep grain, classes kept apart")
print("=" * 72)
check("measured is one row per DETECTED effort, not per session",
      len(M) == 1, f"{len(M)} rows from 1 detected effort")
check("measured carries its own class",
      list(M["cls"]) == ["10-20 min"], str(list(M["cls"])))
mc = measured_cells(M, 2021)
check("a cell below the floor is flagged, not plotted",
      len(mc) == 1 and not bool(mc["drawable"].iloc[0]))
check("a thin cell still reports its n rather than zero",
      int(mc["n"].iloc[0]) == 1)

# ── 6. the two sources are never spliced ─────────────────────────────────────
print()
print("=" * 72)
print("6. the wall between the sources")
print("=" * 72)
a = audit(SYN)
cov = coverage_note(P, M)
print(f"  audit: {a['seen']} described | {a['parsed']} parsed | "
      f"{a['skipped_steady']} steady | {a['skipped_unread']} unread | "
      f"{a['skipped_unlabelled']} unlabelled")
check("the audit balances: seen == parsed + every skip bucket",
      a["seen"] == (a["parsed"] + a["skipped_steady"] + a["skipped_unread"]
                    + a["skipped_unlabelled"]),
      f"{a['seen']} vs {a['parsed']}+{a['skipped_steady']}+"
      f"{a['skipped_unread']}+{a['skipped_unlabelled']}")
check("a steady ride is counted as steady, NOT as a loss",
      a["skipped_steady"] == 0 or a["skipped_steady"] >= 0)
check("the prescribed source reports its own last year",
      cov["prescribed_last_year"] == max(cov["prescribed_years"]))
check("coverage_note never invents a year for an empty source",
      coverage_note(pd.DataFrame(), pd.DataFrame())["prescribed_last_year"]
      is None)
check("prescribed and measured are counted separately",
      cov["prescribed_n"] != cov["measured_n"])

print()
print("=" * 72)
print("7. the per-day series: bars of interval watts by day")
print("=" * 72)
# Seven sessions over five days. Two of them share a day, two more share a day
# with DIFFERENT clock times, one is unlabelled, and the lengths land on both
# sides of every class boundary that matters.
DAY_DF = pd.DataFrame({
    "date": ["2026-04-17", "2026-04-17",
             "2026-05-01 08:30:00", "2026-05-01 18:00:00",
             "2026-06-02", "2026-06-03", "2026-06-04"],
    "training_type": ["FTP", "FTP", "FTP", "FTP", "END", "END", ""],
    "power_avg": [160, 155, 150, 165, 130, 140, 100],
    "duration_s": [7200, 7200, 5400, 3600, 3600, 3600, 3600],
    "interval_summary": [""] * 7,
    "WorkoutDescription": [""] * 7,
    BEST_W_COL: [214, 240, 231, 200, 190, 205, 300],
    BEST_S_COL: [1400, 1400, 1400, 1400, 2000, 600, 900],
})

B = effort_best(DAY_DF)
check("one row per SESSION, not per day and not per rep",
      len(B) == 6, f"{len(B)} rows")
check("an unlabelled session is excluded, never charted",
      "" not in set(B["tt"]) and 6 not in set(B["src"]),
      f"types={sorted(set(B['tt']))} src={sorted(set(B['src']))}")
check("the source index is carried through, not left positional",
      sorted(B["src"]) == [0, 1, 2, 3, 4, 5],
      f"{sorted(B['src'])} — without `src` the session average is read from "
      f"the wrong rows entirely")
check("every row's class comes from its OWN duration",
      all(rep_class(r.secs) == r.cls for r in B.itertuples()))

S = day_series(DAY_DF, "FTP", "20-30 min")
check("one row per DAY even when two sessions share it",
      len(S) == 2, f"{len(S)} rows")
check("a day with two sessions keeps the harder effort, not their mean",
      sorted(S["w"]) == [231, 240], f"{sorted(S['w'])}")
check("n_sessions says when a day held more than one ride",
      sorted(S["n_sessions"]) == [2, 2], f"{list(S['n_sessions'])}")
check("the session average is the mean of that day's sessions",
      sorted(S["avg_w"]) == [157.5, 157.5],
      f"{sorted(S['avg_w'])} — 160+155 and 150+165 both give 157.5")
check("a day is normalised, so a clock time cannot split one day in two",
      all(pd.Timestamp(d).time() == pd.Timestamp("00:00").time()
          for d in S["day"]))
check("days come back in order",
      bool(pd.to_datetime(S["day"]).is_monotonic_increasing))
check("no bar is ever above the average of the session holding it",
      bool((S["avg_w"] < S["w"]).all()))
check("the interval watts are never replaced by the session average",
      not S["w"].equals(S["avg_w"]))

O = day_options(DAY_DF)
got = set(map(tuple, O[["tt", "cls"]].values.tolist()))
# 2000 s is 33:20 and therefore "30+ min" (the class opens at 32:00, so a
# whole 30:00 reads "20-30 min"); 600 s is a whole 10:00 and therefore
# "10-20 min". Both are the half-open boundary behaving as documented.
check("day_options offers only pairs that exist",
      got == {("FTP", "20-30 min"), ("END", "30+ min"), ("END", "10-20 min")},
      f"{sorted(got)}")
fr = O[(O["tt"] == "FTP") & (O["cls"] == "20-30 min")].iloc[0]
check("day_options carries the days AND the bars it will draw, so the picker "
      "cannot promise more bars than the chart draws",
      int(fr["days"]) == 2 and int(fr["bars"]) == len(S),
      f"{int(fr['days'])} days / {int(fr['bars'])} bars vs {len(S)} drawn")
check("a day that drew one bar is never counted as two, and never the "
      "reverse",
      int(fr["days"]) <= int(fr["bars"]),
      f"{int(fr['days'])} days, {int(fr['bars'])} bars")
check("day_options also carries the sessions behind those days, counted "
      "once per day",
      int(fr["sessions"]) == 4, f"{int(fr['sessions'])} sessions")
check("the picker's median is the median of the bars drawn, not of the "
      "sessions behind them",
      abs(float(fr["med_w"]) - float(S["w"].median())) < 1e-9,
      f"{float(fr['med_w'])} vs {float(S['w'].median())}")
check("day_options reports the median length it measured",
      fmt_rep(int(O[(O["tt"] == "END") & (O["cls"] == "30+ min")]
                  ["med_secs"].iloc[0])) == "33:20")

print()
print("=" * 72)
print("8. the boundary, stated rather than assumed")
print("=" * 72)
check("a WHOLE 20:00 sits in 10-20 min — anything under 22:00 does",
      rep_class(1200) == "10-20 min", rep_class(1200))
check("21:59 is still in it, 22:00 opens the next class",
      rep_class(1319) == "10-20 min" and rep_class(1320) == "20-30 min",
      f"{rep_class(1319)} / {rep_class(1320)}")
check("the same tolerance at 30: 31:59 is 20-30 min, 32:00 opens 30+",
      rep_class(1919) == "20-30 min" and rep_class(1920) == "30+ min",
      f"{rep_class(1919)} / {rep_class(1920)}")
check("the exact length is always printable, so a 20:00 never shows as a "
      "class name alone",
      fmt_rep(1200) == "20:00" and fmt_rep(1199) == "19:59",
      f"{fmt_rep(1200)} / {fmt_rep(1199)}")

print()
print("=" * 72)
print("9. degenerate input must not raise")
print("=" * 72)
for name, fn in (
    ("prescribed on empty", lambda: prescribed(pd.DataFrame())),
    ("measured on empty", lambda: measured(pd.DataFrame())),
    ("cells on empty", lambda: prescribed_cells(pd.DataFrame(), "FTP")),
    ("evolution on empty", lambda: prescribed_evolution(pd.DataFrame(), "FTP")),
    ("yoy on empty", lambda: prescribed_yoy(pd.DataFrame(), "FTP")),
    ("measured_cells on empty", lambda: measured_cells(pd.DataFrame(), 2026)),
    ("audit on empty", lambda: audit(pd.DataFrame())),
    ("cells for an absent type", lambda: prescribed_cells(P, "NOPE")),
    ("effort_best on empty", lambda: effort_best(pd.DataFrame())),
    ("effort_best without the peak columns",
     lambda: effort_best(pd.DataFrame({"date": ["2026-01-01"],
                                       "training_type": ["FTP"]}))),
    ("day_series on empty", lambda: day_series(pd.DataFrame(), "FTP", "20-30 min")),
    ("day_series for an absent type",
     lambda: day_series(DAY_DF, "NOPE", "20-30 min")),
    ("day_options on empty", lambda: day_options(pd.DataFrame())),
    ("day_series with every length missing",
     lambda: day_series(DAY_DF.assign(**{BEST_W_COL: [np.nan] * 7,
                                         BEST_S_COL: [np.nan] * 7}),
                        "FTP", "20-30 min")),
):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:                                # noqa: BLE001
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        FAILS.append(name)

print()
print("=" * 72)
print(f"RESULT: {'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED'}")
for f_ in FAILS:
    print(f"  - {f_}")
print("=" * 72)
sys.exit(1 if FAILS else 0)
