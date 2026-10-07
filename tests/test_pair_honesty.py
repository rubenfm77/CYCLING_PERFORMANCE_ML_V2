"""One bar per effort ridden — never two, never zero.

The reported case: the athlete rides 2x20 min, but the Intervals detail and
the Evolution chart show ONE interval per session. Three mechanisms caused
it, each pinned here on synthetic frames (no files, no credentials):

  1. the CLASS SPLIT: a 19:17 and a 20:07 of one session landed in
     "10-20 min" and "20-30 min" — each chart drew one. The tolerance rule
     (anything under 22:00 is 10-20 min) keeps them together;
  2. the DUPLICATE COPY: the same ride sits in the cache twice under two
     activity ids with different segmentations, and the kept copy had merged
     work into recovery (a 24:47 block at 119 W next to 12-minute blocks at
     245 W). The coherent copy wins even unlabelled, inherits the ride's
     label, and the swap is reported;
  3. the DOUBLE COUNT: a rebuilt run (or a meter window) drawn next to the
     very fragments it was joined from — 683 s + 1260 s for one 20-minute
     effort. Consumed fragments are not drawn again, matched on
     (activity, seq) so a genuine second effort is never mistaken for one.

Run:  python tests/test_pair_honesty.py     (exit 0 = pass, 1 = fail)
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd

from ml.interval_watts import (rep_class, rebuilt_efforts,
                               REBUILD_SRC, PEAK_ROW_SRC)
from ml.set_evolution import (build_sets, fill_peak_efforts, quality_gates,
                              SWAP_REASON, DUP_REASON)
from ml.type_comparison import (dur_bucket, dur_class_label, fmt_min,
                                series_sel)

FAILED = []


def check(name, cond, detail=""):
    print(f"   {'PASS' if cond else 'FAIL'}  {name}"
          + (f" — {detail}" if (not cond and detail) else ""))
    if not cond:
        FAILED.append(name)


print("=" * 74)
print("1. the tolerance rule keeps one workout in one class")
print("=" * 74)
for secs, want in [(1157, "10-20 min"), (1200, "10-20 min"),
                   (1207, "10-20 min"), (1260, "10-20 min"),
                   (1319, "10-20 min"), (1320, "20-30 min"),
                   (1440, "20-30 min"), (1860, "20-30 min"),
                   (1920, "30+ min")]:
    check(f"{secs}s -> {want}", rep_class(secs) == want, rep_class(secs))

print()
print("=" * 74)
print("2. durations round to the closest nominal, never to seconds noise")
print("=" * 74)
for secs, want_b, want_lab in [(1207, 20.0, "20 min"), (1260, 20.0, "20 min"),
                               (1349, 20.0, "20 min"), (1487, 25.0, "25 min"),
                               (1702, 30.0, "30 min"), (1860, 30.0, "30 min"),
                               (966, 15.0, "15 min"), (750, 15.0, "15 min"),
                               (734, 10.0, "10 min"),
                               (302, 5.0, "5 min"), (181, 3.0, "3 min"),
                               (30, 0.5, "30 s")]:
    check(f"dur_bucket({secs}) == {want_b}",
          dur_bucket(secs) == want_b, dur_bucket(secs))
    check(f"label({secs}) == {want_lab!r}",
          dur_class_label(dur_bucket(secs)) == want_lab
          and fmt_min(secs) == want_lab,
          f"{dur_class_label(dur_bucket(secs))} / {fmt_min(secs)}")

print()
print("=" * 74)
print("3. the coherent duplicate copy wins, labelled or not")
print("=" * 74)


def _sets_df(rows):
    f = pd.DataFrame(rows)
    f["date"] = pd.to_datetime("2026-04-22")
    return f


# Same ride, same start second, one shared row — two segmentations. The
# labelled copy merged the work into a 24:47 block at 119 W; the other copy
# holds three 12-minute blocks at ~245 W.
dup = _sets_df([
    {"activity_id": "iA", "date": pd.Timestamp("2026-04-22"),
     "_key": "k1", "reps": 1, "rep_secs": 1487.0, "set_w": 119.0,
     "set_np": np.nan, "set_hr": np.nan, "intensity": np.nan, "load": np.nan,
     "cad": np.nan, "decoupling": np.nan, "first_seq": np.nan,
     "last_seq": np.nan, "rest": np.nan, "name": "", "temp": np.nan,
     "moving_time": 9606.0, "act_iv_n": 6.0, "act_iv_secs": 2620.0,
     "act_iv_dist": np.nan, "act_start": pd.Timestamp("2026-04-22 15:24:28"),
     "act_iv_rows": ("254@183@180", "734@244@240"), "db_type": "FTP",
     "set_ss_cp_w": np.nan, "set_ss_w_prime_kj": np.nan,
     "set_w5s_cv": np.nan, "Source": "detected interval"},
    {"activity_id": "iA", "date": pd.Timestamp("2026-04-22"),
     "_key": "k2", "reps": 1, "rep_secs": 734.0, "set_w": 244.0,
     "set_np": np.nan, "set_hr": np.nan, "intensity": np.nan, "load": np.nan,
     "cad": np.nan, "decoupling": np.nan, "first_seq": np.nan,
     "last_seq": np.nan, "rest": np.nan, "name": "", "temp": np.nan,
     "moving_time": 9606.0, "act_iv_n": 6.0, "act_iv_secs": 2620.0,
     "act_iv_dist": np.nan, "act_start": pd.Timestamp("2026-04-22 15:24:28"),
     "act_iv_rows": ("254@183@180", "734@244@240"), "db_type": "FTP",
     "set_ss_cp_w": np.nan, "set_ss_w_prime_kj": np.nan,
     "set_w5s_cv": np.nan, "Source": "detected interval"},
    {"activity_id": "iB", "date": pd.Timestamp("2026-04-22"),
     "_key": "k3", "reps": 3, "rep_secs": 734.0, "set_w": 245.7,
     "set_np": np.nan, "set_hr": np.nan, "intensity": np.nan, "load": np.nan,
     "cad": np.nan, "decoupling": np.nan, "first_seq": np.nan,
     "last_seq": np.nan, "rest": 300.0, "name": "", "temp": np.nan,
     "moving_time": 9606.0, "act_iv_n": 6.0, "act_iv_secs": 2476.0,
     "act_iv_dist": np.nan, "act_start": pd.Timestamp("2026-04-22 15:24:28"),
     "act_iv_rows": ("254@183@180", "734@244@240"), "db_type": "",
     "set_ss_cp_w": np.nan, "set_ss_w_prime_kj": np.nan,
     "set_w5s_cv": np.nan, "Source": "detected interval"},
    {"activity_id": "iB", "date": pd.Timestamp("2026-04-22"),
     "_key": "k4", "reps": 1, "rep_secs": 13.0, "set_w": 410.0,
     "set_np": np.nan, "set_hr": np.nan, "intensity": np.nan, "load": np.nan,
     "cad": np.nan, "decoupling": np.nan, "first_seq": np.nan,
     "last_seq": np.nan, "rest": np.nan, "name": "", "temp": np.nan,
     "moving_time": 9606.0, "act_iv_n": 6.0, "act_iv_secs": 2476.0,
     "act_iv_dist": np.nan, "act_start": pd.Timestamp("2026-04-22 15:24:28"),
     "act_iv_rows": ("254@183@180", "734@244@240"), "db_type": "",
     "set_ss_cp_w": np.nan, "set_ss_w_prime_kj": np.nan,
     "set_w5s_cv": np.nan, "Source": "detected interval"},
])
clean, report, dropped = quality_gates(dup)
kept_ids = set(clean["activity_id"].astype(str))
check("the coherent copy survives, the merged one does not",
      kept_ids == {"iB"}, f"kept {sorted(kept_ids)}")
check("the swap is reported as a swap, not a plain duplicate",
      set(dropped["q_why"]) == {SWAP_REASON}, set(dropped["q_why"]))
check("the kept copy inherits the ride's label, not a heuristic",
      set(clean["db_type"].astype(str)) == {"FTP"},
      set(clean["db_type"].astype(str)))

# Identical rows, both coherent: the label still decides, nothing swaps.
same = dup.copy()
same.loc[same["activity_id"] == "iB", "rep_secs"] = [1487.0, 734.0]
same.loc[same["activity_id"] == "iB", "set_w"] = [119.0, 244.0]
same.loc[same["activity_id"] == "iB", "db_type"] = ""
clean2, _, dropped2 = quality_gates(same)
check("identical copies keep the old label-first order",
      set(clean2["activity_id"].astype(str)) == {"iA"}
      and set(dropped2["q_why"]) == {DUP_REASON},
      f"{sorted(set(clean2['activity_id']))} / {set(dropped2['q_why'])}")

print()
print("=" * 74)
print("4. a run and its fragments are never drawn twice")
print("=" * 74)


def _iv_df(rows, act="i9", day="2026-07-04"):
    out = pd.DataFrame([{"activity_id": act, "date": day, "iv_type": t,
                         "secs": float(s), "avg_w": float(w), "np_w": np.nan,
                         "hr_avg": np.nan, "intensity": np.nan, "load": np.nan,
                         "cad_avg": np.nan, "decoupling": np.nan,
                         "distance": np.nan, "ss_cp_w": np.nan,
                         "ss_w_prime_kj": np.nan, "w5s_cv": np.nan,
                         "group_id": None, "seq": float(i + 1)}
                        for i, (t, s, w) in enumerate(rows)])
    out["date"] = pd.to_datetime(out["date"])
    return out


frag = _iv_df([("WORK", 683, 242), ("RECOVERY", 18, 145),
               ("WORK", 511, 245)])
frag.loc[frag["seq"] == 1.0, "group_id"] = "683s@242w92rpm"
frag.loc[frag["seq"] == 3.0, "group_id"] = "511s@245w93rpm"
rdf = pd.DataFrame([{"id": "i9", "date": pd.Timestamp("2026-07-04"),
                     "training_type": "FTP", "icu_pm_ftp_watts": 240.0,
                     "icu_pm_ftp_secs": 1260.0}])
reb = rebuilt_efforts(frag, rdf)
check("the 683 + 511 join into one 1212 run",
      len(reb) == 1 and int(reb["secs"].iloc[0]) == 1212,
      reb["secs"].tolist() if len(reb) else "nothing rebuilt")
check("the run remembers which rows it was joined from",
      tuple(reb["seqs"].iloc[0]) == (1.0, 3.0),
      reb["seqs"].tolist() if len(reb) else "no seqs")

from ml.protocol_reps import _set_members, _uncovered_sets  # noqa: E402

sets9 = build_sets(frag, None, rdf)
sets9, _ = fill_peak_efforts(sets9, rdf, iv=frag)
sel = sets9.copy()
sel["family"] = "FTP"
sel["dur_cls"] = sel["rep_secs"].map(rep_class)
sel = series_sel(sel, "FTP", dur_cls="10-20 min")
check("the 683 fragment and the run sit in the same selection",
      set(sel["_key"].astype(str)) == {"683s@242w92rpm",
                                       "rebuilt:i9:1212s"},
      sel["_key"].tolist())
reps, consumed = _set_members(frag, sel)
check("the run's fragments fall out of the member rows",
      not len(reps), f"{len(reps)} member rows left")
solo = _uncovered_sets(sel, reps, consumed)
check("the swallowed set is not solo-drawn beside the run",
      set(solo["_key"].astype(str)) == {"rebuilt:i9:1212s"},
      solo["_key"].tolist() if len(solo) else "no solo bars")

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    raise SystemExit(1)
print("ALL CHECKS PASS")
