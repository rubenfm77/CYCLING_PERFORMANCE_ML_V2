# tests/test_label_authority.py — the athlete's own label outranks the heuristic.
r"""
Runs on SYNTHETIC frames with a stubbed streamlit: no credentials, no network.

Two rules are pinned here, and they are the same rule seen from two directions.

1. `set_evolution._style` names a 30 s rep "Ronnestad-style" on a measured 16 s
   recovery and "Billat-style" on 31 s. That measurement is real, but on this
   athlete's file it renamed the workout: all six sets it called "Ronnestad-style"
   are sessions filed under BILLAT, and the two it called "Billat-style" are an
   AEROBIC BASE ride. So when the session carries BILLAT, the family is BILLAT.

2. `core.data._apply_type_aliases` reads "RONNESTAD" — the same workout type
   written in Spanish — as BILLAT, so it can never become a twelfth training type
   and split a series the athlete wants compared across years.

The point of both is the same: a categorisation the athlete did not make must
not sit beside theirs in a chart, and a name they wrote must not fork into two.

3. `core.data.auto_ftp_mask` types an unlabelled session FTP only when its
   18–25 min peak window is >= 95 % of that ride's own eFTP **and** >= 15 % of
   the ride itself. The share test is what stopped 4 Oct 2026 — a 3 h 09 ride
   at IF 0.77 whose best 21 min happened to hit 106 % of eFTP — from being
   charted as an FTP session, while 30 Sep 2026 (2 × 20 min inside a 1 h 54
   ride, 19 %) keeps the label. No invention in either direction.

Run:  python tests/test_label_authority.py     (exit 0 = pass, 1 = fail)
"""
import os
import sys
import types
import warnings

warnings.filterwarnings("ignore")


def _cm(*_a, **_k):
    class _C:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *exc):
            return False
    return _C()


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "cache_data":
            def _cache(f=None, **k):
                return f if f is not None else (lambda g: g)
            return _cache
        if name == "session_state":
            return {}
        if name in ("columns", "tabs"):
            return lambda spec, *a, **k: [_cm() for _ in
                                         range(spec if isinstance(spec, int)
                                              else len(list(spec)))]
        return lambda *a, **k: None


_st = _Stub("streamlit")
_st.session_state = {"_range": "All", "_types": []}
sys.modules.setdefault("streamlit", _st)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

from core.data import (FTP_AUTO_SHARE, _apply_type_aliases,
                       auto_ftp_mask)                        # noqa: E402
from core.theme import BLANK_TYPE_TOKENS, MAIN_TYPES, TYPE_ALIASES  # noqa: E402
from ml.set_evolution import _style                       # noqa: E402
from ml.type_comparison import FAMILY_ORDER, FAMILY_SHORT  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        if detail:
            print(f"        {detail}")
        FAILS.append(name)


# ── 1. the style heuristic, and the label that outranks it ────────────────────
print("=" * 72)
print("1. a 30 s set is named after the athlete's label when there is one")
print("=" * 72)

check("a BILLAT session's 30 s set is BILLAT, at a 16 s recovery",
      _style(29, 29, 16, "BILLAT") == "BILLAT",
      _style(29, 29, 16, "BILLAT"))
check("the same set at a 31 s recovery is still BILLAT",
      _style(30, 20, 31, "BILLAT") == "BILLAT",
      _style(30, 20, 31, "BILLAT"))
check("Ronnestad-style is merged to BILLAT (no label)",
      _style(29, 29, 16, "") == "BILLAT",
      _style(29, 29, 16, ""))
check("and the other way too",
      _style(30, 20, 31, "") == "Billat-style (30 s on / 30 s off)",
      _style(30, 20, 31, ""))
check("a missing label is treated the same as empty",
      _style(29, 29, 16, None) == "BILLAT")
check("the label does not overreach: an AEROBIC BASE 30 s set is not BILLAT",
      _style(30, 20, 31, "AEROBIC BASE") == "Billat-style (30 s on / 30 s off)",
      _style(30, 20, 31, "AEROBIC BASE"))
check("the label only ever applies to the 30 s branch",
      _style(300, 4, 60, "BILLAT") == "FTP / threshold sets",
      _style(300, 4, 60, "BILLAT"))
check("a long rep from a BILLAT ride is not called BILLAT either",
      _style(1200, 1, 0, "BILLAT") == "long efforts",
      _style(1200, 1, 0, "BILLAT"))
check("an 8-rep 30 s set below the 15-rep floor is not swallowed",
      _style(32, 8, 16, "BILLAT") == "BILLAT"
      and _style(32, 8, 16, "") == "micro-reps",
      f"labelled={_style(32, 8, 16, 'BILLAT')} unlabelled={_style(32, 8, 16, '')}")
check("the call still works with no label argument at all",
      _style(29, 29, 16) == "BILLAT")
check("BILLAT is registered as a family, first, and maps to itself",
      FAMILY_ORDER[0] == "BILLAT" and FAMILY_SHORT["BILLAT"] == "BILLAT",
      f"{FAMILY_ORDER[:3]}")
check("BILLAT is one of the agreed types, so no label was invented",
      "BILLAT" in MAIN_TYPES)

# ── 2. the alias map ─────────────────────────────────────────────────────────
print()
print("=" * 72)
print("2. RONNESTAD is read as BILLAT, and only exactly")
print("=" * 72)
print(f"  TYPE_ALIASES = {TYPE_ALIASES}")

check("every alias target is an agreed type",
      set(TYPE_ALIASES.values()) <= set(MAIN_TYPES),
      f"{sorted(set(TYPE_ALIASES.values()) - set(MAIN_TYPES))}")

A = pd.DataFrame({"training_type": [
    "RONNESTAD", "ronnestad", "Ronnestad", " RONNESTAD ",
    "BILLAT", "FTP", "END", "—", "nan", None, "RONNESTADA", "X-RONNESTAD",
]})
n = _apply_type_aliases(A)
check("the exact spelling is rewritten", A.at[0, "training_type"] == "BILLAT",
      A.at[0, "training_type"])
check("and only the exact spelling — case is NOT folded",
      A.at[1, "training_type"] == "ronnestad" and
      A.at[2, "training_type"] == "Ronnestad",
      f"{A.at[1, 'training_type']!r}, {A.at[2, 'training_type']!r}")
check("surrounding whitespace is stripped before matching",
      A.at[3, "training_type"] == "BILLAT", A.at[3, "training_type"])
check("a word merely CONTAINING the alias is not rewritten",
      A.at[10, "training_type"] == "RONNESTADA" and
      A.at[11, "training_type"] == "X-RONNESTAD",
      f"{A.at[10, 'training_type']!r}, {A.at[11, 'training_type']!r}")
check("every other label is left exactly as written",
      [A.at[i, "training_type"] for i in (4, 5, 6, 7, 8)] ==
      ["BILLAT", "FTP", "END", "—", "nan"],
      str([A.at[i, "training_type"] for i in (4, 5, 6, 7, 8)]))
check("a missing label stays missing, never becomes the string 'None'",
      pd.isna(A.at[9, "training_type"]),
      repr(A.at[9, "training_type"]))
check("the count reports what changed", n == 2, f"{n} reported, 2 expected")
check("a blank token is not an alias",
      not (set(BLANK_TYPE_TOKENS) & set(TYPE_ALIASES)))

B = pd.DataFrame({"training_type": ["BILLAT", "FTP"]})
check("a frame with no alias changes nothing and reports zero",
      _apply_type_aliases(B) == 0 and B["training_type"].tolist() ==
      ["BILLAT", "FTP"])
check("an empty frame is safe",
      _apply_type_aliases(pd.DataFrame({"training_type": []})) == 0)
check("a frame with no training_type column is safe",
      _apply_type_aliases(pd.DataFrame({"date": ["2026-01-01"]})) == 0)

# ── 3. the two together ──────────────────────────────────────────────────────
print()
print("=" * 72)
print("3. both spellings end up on ONE series")
print("=" * 72)
C = pd.DataFrame({"training_type": ["RONNESTAD", "BILLAT", "RONNESTAD"]})
_apply_type_aliases(C)
check("a session written RONNESTAD is indistinguishable from one written "
      "BILLAT, so both count in the same comparison",
      set(C["training_type"]) == {"BILLAT"},
      str(C["training_type"].tolist()))
check("and its 30 s set is the same family either way",
      _style(29, 29, 16, C.at[0, "training_type"]) ==
      _style(29, 29, 16, C.at[1, "training_type"]) == "BILLAT")

print()
# ── 4. the auto-label only calls it FTP when the ride was built around it ────
print()
print("=" * 72)
print("4. the FTP auto-label needs the window to define the ride")
print("=" * 72)
print(f"  18–25 min · ≥ 95 % of the ride's eFTP · SHARE={FTP_AUTO_SHARE:.0%} "
      f"of the ride")

# Shapes taken from the two sessions the rule has ever caught on the live file.
# eFTP is back-solved from the % the audit table prints (106.2 % / 109.3 %).
AUTO = pd.DataFrame([
    # label,              W,   window,   eFTP,  ride secs   →  expected
    dict(training_type=None,           icu_pm_ftp_watts=236.0,
     icu_pm_ftp_secs=1260.0, eftp=222.2, duration_s=11340.0),   # 4 Oct: 11 %
    dict(training_type=None,           icu_pm_ftp_watts=244.0,
     icu_pm_ftp_secs=1320.0, eftp=223.2, duration_s=6840.0),    # 30 Sep: 19 %
    dict(training_type="AEROBIC BASE", icu_pm_ftp_watts=244.0,
     icu_pm_ftp_secs=1320.0, eftp=223.2, duration_s=6840.0),    # label wins
    dict(training_type=None,           icu_pm_ftp_watts=244.0,
     icu_pm_ftp_secs=900.0, eftp=223.2, duration_s=6840.0),     # 15 min window
    dict(training_type=None,           icu_pm_ftp_watts=244.0,
     icu_pm_ftp_secs=1320.0, eftp=223.2, duration_s=float("nan")),  # no ride
    dict(training_type=None,           icu_pm_ftp_watts=210.0,
     icu_pm_ftp_secs=1320.0, eftp=223.2, duration_s=6840.0),    # 94 % of eFTP
    dict(training_type=None,           icu_pm_ftp_watts=244.0,
     icu_pm_ftp_secs=1080.0, eftp=223.2, duration_s=7200.0),    # exactly 15 %
    dict(training_type=None,           icu_pm_ftp_watts=244.0,
     icu_pm_ftp_secs=1560.0, eftp=223.2, duration_s=6840.0),    # 26 min window
    dict(training_type="—",            icu_pm_ftp_watts=244.0,
     icu_pm_ftp_secs=1320.0, eftp=223.2, duration_s=6840.0),    # blank token
])
_m = auto_ftp_mask(AUTO)
print(f"  matched {int(_m.sum())} of {len(AUTO)} synthetic sessions")

check("4 Oct 2026 — 21 of 189 min inside a 3 h ride — is NOT typed FTP",
      not _m.iloc[0],
      f"window {AUTO.icu_pm_ftp_secs.iloc[0]:.0f}s of "
      f"{AUTO.duration_s.iloc[0]:.0f}s")
check("30 Sep 2026 — 22 of 114 min, the ride he showed up for — IS typed FTP",
      bool(_m.iloc[1]))
check("an athlete's own label is never touched by the rule",
      not _m.iloc[2])
check("a 15-min window is outside the threshold-effort definition",
      not _m.iloc[3])
check("a missing ride duration invents nothing (no share, no label)",
      not _m.iloc[4])
check("watts below 95 % of that ride's eFTP do not match",
      not _m.iloc[5])
check("exactly 15 % of the ride still counts",
      bool(_m.iloc[6]),
      f"{AUTO.icu_pm_ftp_secs.iloc[6]:.0f}/{AUTO.duration_s.iloc[6]:.0f}")
check("a 26-min window is outside the definition too",
      not _m.iloc[7])
check("a blank type token still means unlabelled, so the rule may apply",
      bool(_m.iloc[8]))
check("the share threshold is the one the pages print",
      FTP_AUTO_SHARE == 0.15, str(FTP_AUTO_SHARE))
check("the mask never returns rows the frame does not have",
      len(_m) == len(AUTO) and _m.dtype == bool)
check("a frame with none of the columns is safe and matches nothing",
      not bool(auto_ftp_mask(pd.DataFrame({"date": ["2026-10-04"]})).any()))
check("an empty frame is safe",
      len(auto_ftp_mask(pd.DataFrame({"training_type": [],
                                      "icu_pm_ftp_secs": []}))) == 0)

print()
print("=" * 72)
print(f"RESULT: {'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED'}")
for f_ in FAILS:
    print(f"  - {f_}")
print("=" * 72)
sys.exit(1 if FAILS else 0)