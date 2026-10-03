"""The coach-note parser, on the athlete's own wording.

These are real sentences from this file, including the ones the athlete
spelled out himself: a 2 x 20-minute set built out of 4-minute blocks, a
cadence bracket between the rep and its wattage, a 15-minute rep holding a
30-second sprint and a 14-minute block, and a pulse-only note that must stay
refused.

It also pins the direction of travel: whatever the old rule could read, the
new one must still read. Nothing may be lost.

Run: python -c "from tests import test_coach_note"
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: F401

from core.data import load_data
from ml import interval_watts as iw
from ml.interval_watts import _outcomes, audit, main_set, parse_prescribed

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")
        FAILS.append(name)


print("=" * 72)
print("1. the sentences the athlete reads as obvious")
print("=" * 72)

flat20 = ("WU: 20' entre 160-170w MS: 2x20' terreny constant (4'265-280w+1' "
          "suau x4 cops) (5'recup) CD: COMPLETAR FINS 1'30H entre 160-170W")
check("2x20' built from 4-minute blocks reads as 2 x 20:00 @ 265-280 W",
      parse_prescribed(flat20) == [(2, 1200, 265.0, 280.0)],
      str(parse_prescribed(flat20)))
check("that target is reported as read FROM INSIDE the rep",
      _outcomes(flat20)[0].get("basis") == "inside",
      str(_outcomes(flat20)))

rpm_gap = ("2h terreny constant entre 140-160w incloent 3x1' entre 100-110rpm "
           "(-220w) per després realitzar un esforç de 5' al MÀXIM (pujada). "
           "La tornada entre 140-160w (mitja).")
check("a cadence bracket between the rep and its watts does not block the read",
      parse_prescribed(rpm_gap) == [(3, 60, 220.0, 220.0)],
      str(parse_prescribed(rpm_gap)))
check("the endurance pace AFTER the set is not taken for the rep",
      all(r[2] >= 200 for r in parse_prescribed(rpm_gap)),
      str(parse_prescribed(rpm_gap)))

block15 = ("WU: 45' escalfar entre 150-165w. MS: 3x15' pujada (30\" +330w "
           "+14' 250w +30\" +330w) (10' recup -150w ) CD: COMPLETAR FINS 3H.")
check("a 15-minute rep takes the 14-minute block's 250 W, not the 30 s sprint",
      parse_prescribed(block15) == [(3, 900, 250.0, 250.0)],
      str(parse_prescribed(block15)))

pulse = ("WU: 60' entre 120-135pols. MS:2 blocs de 15' (5x30\" +165pols + "
         "2'30\" 155-165pols) (15'recup -120 pols) CD: COMPLETAR FINS 3h.")
check("a pulse-only note stays refused: pols is not watts",
      parse_prescribed(pulse) == [], str(parse_prescribed(pulse)))
check("its refusal reason is 'no wattage written', not a guess",
      {o["status"] for o in _outcomes(pulse)} == {"no_watts"},
      str(_outcomes(pulse)))

check("a mixed minute-second rep is ninety seconds, not one minute",
      parse_prescribed("MS: 4x1'30\" en pujada entre 345-355w (3'30\" recup)")
      == [(4, 90, 345.0, 355.0)],
      str(parse_prescribed("MS: 4x1'30\" en pujada entre 345-355w "
                           "(3'30\" recup)")))

print()
print("=" * 72)
print("2. the walls that must not fall")
print("=" * 72)

check("a rep does not borrow the recovery effort's watts",
      parse_prescribed("MS: 4x15' pujada (10' recup -150w) CD: 10'") == [],
      str(parse_prescribed("MS: 4x15' pujada (10' recup -150w) CD: 10'")))

after_set = ("MS: 3x30\" MÀX amb 4'30\" recup entre intervals. Completar fins "
             "4h per entre 155-165w")
check("the endurance pace written after 'Completar' is never the rep's target",
      parse_prescribed(after_set) == [], str(parse_prescribed(after_set)))

check("a sub-block LONGER than the rep is refused, not adopted",
      parse_prescribed("MS: 3x15' pujada (20'265-280w) (5'recup)") == [],
      str(parse_prescribed("MS: 3x15' pujada (20'265-280w) (5'recup)")))

check("two candidates far from the rep length are refused as unattributable",
      parse_prescribed("MS: 3x15' pujada (2'265-280w + 40'300w) (5'recup)")
      == [],
      str(parse_prescribed("MS: 3x15' pujada (2'265-280w + 40'300w) "
                           "(5'recup)")))

check("a rep shorter than 30 s keeps its readable watts but is still refused",
      [{o["status"] for o in _outcomes('MS: 6x10" a 700w')} == {"too_short"},
       parse_prescribed('MS: 6x10" a 700w') == []] == [True, True],
      str(_outcomes('MS: 6x10" a 700w')))

check("a wattage under 100 W is refused as implausible",
      {o["status"] for o in _outcomes("MS: 4x5' a 60-80w")} == {"implausible"},
      str(_outcomes("MS: 4x5' a 60-80w")))

print()
print("=" * 72)
print("3. a ceiling is tagged as a ceiling")
print("=" * 72)

ceiling = ("WU: 15' entre 155-170w 10' entre 185-195w MS: 6X4' pujada 10% "
           "entre 40-50rpm (sentat, per sota 265w) (4'recup -155w)")
o = _outcomes(ceiling)
check("'per sota 265w' is read as a target AND flagged as a ceiling",
      o and o[0]["status"] == "ok" and o[0]["lo"] == 265.0
      and o[0]["ceiling"] is True, str(o))

cap = 'MS: 5x1\' terreny plà entre 105-110rpm (cadència) sense passar 200w (2\'recup)'
o = _outcomes(cap)
check("'sense passar 200w' is read and flagged as a ceiling",
      o and o[0]["status"] == "ok" and o[0]["lo"] == 200.0
      and o[0]["ceiling"] is True, str(o))

plain = "MS: 3x15' PLA entre 225-240w 10' RECUP"
o = _outcomes(plain)
check("an ordinary target is not flagged as a ceiling",
      o and o[0]["status"] == "ok" and o[0]["ceiling"] is False, str(o))

print()
print("=" * 72)
print("4. on the real file: nothing lost, reasons balance")
print("=" * 72)

# The previous rule, verbatim, so "nothing was lost" is checkable forever.
OLD = __import__("re").compile(
    r"(?P<n>\d{1,2})\s*[x*]\s*"
    r"(?:(?P<mins>\d{1,3})\s*(?:'|’|´|min)"
    r"|(?P<qsec>\d{1,3})\s*(?:\"|”|″)"
    r"|(?P<sec>\d{1,2})\s*(?:s|s'|seg))"
    r"[^0-9]{0,40}?(?:entre\s+|a\s+|target\s+)?"
    r"(?P<w1>\d{2,4})\s*(?:-\s*(?P<w2>\d{2,4})\s*)?w\b",
    __import__("re").IGNORECASE)


def legacy(desc):
    if iw._blank(desc):
        return []
    out = []
    for m in OLD.finditer(main_set(desc)):
        if m.group("mins"):
            secs = int(m.group("mins")) * 60
        elif m.group("qsec"):
            secs = int(m.group("qsec"))
        elif m.group("sec"):
            secs = int(m.group("sec"))
        else:
            continue
        lo, hi = float(m.group("w1")), float(m.group("w2") or m.group("w1"))
        if hi < lo:
            lo, hi = hi, lo
        if secs >= 30 and lo >= 100:
            out.append((int(m.group("n")), secs, lo, hi))
    return out


df = load_data()
a = audit(df)
print(f"  audit: {a['seen']} seen | {a['parsed']} parsed | "
      f"{a['skipped_steady']} steady | {a['skipped_unread']} refused | "
      f"{a['skipped_unlabelled']} unlabelled")
print(f"  refusals: {a['refused_no_watts']} no watts, "
      f"{a['refused_too_short']} under the 30 s floor, "
      f"{a['refused_implausible']} implausible, "
      f"{a['refused_ambiguous']} unattributable")
print(f"  reads: {a['reads_direct']} direct, {a['reads_inside']} inside a "
      f"block, {a['reads_ceiling']} ceilings")

check("the audit balances: seen == parsed + every skip bucket",
      a["seen"] == (a["parsed"] + a["skipped_steady"] + a["skipped_unread"]
                    + a["skipped_unlabelled"]),
      str(a["seen"]), )
check("the refusal split adds up to skipped_unread",
      a["skipped_unread"] == (a["refused_no_watts"] + a["refused_too_short"]
                              + a["refused_implausible"]
                              + a["refused_ambiguous"]),
      str([a["skipped_unread"], a["refused_no_watts"],
           a["refused_too_short"], a["refused_implausible"],
           a["refused_ambiguous"]]))
check("direct + inside reads is exactly the parsed count",
      a["reads_direct"] + a["reads_inside"] == a["parsed"],
      str([a["reads_direct"], a["reads_inside"], a["parsed"]]))
check("the note window is reported, with the rides that carry none",
      a["note_last"] is not None and a["note_first"] is not None
      and a["rides_after_last_note"] >= 0,
      str([a["note_first"], a["note_last"], a["rides_after_last_note"]]))

desc = df["WorkoutDescription"]
lost, gained = [], 0
for i in df.index:
    d = desc.at[i]
    if iw._blank(d):
        continue
    old, new = legacy(d), parse_prescribed(d)
    if any(r not in new for r in old):
        lost.append((d, old, new))
    if new and not old:
        gained += 1
check("every rep the old rule could read is still read (nothing lost)",
      not lost, str(lost[:3]))
print(f"  gained {gained} description(s) the old rule refused entirely")
check("the gain is real: the parser reads more than it did", gained >= 50,
      str(gained))

print()
print("=" * 72)
print(f"RESULT: {'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED'}")
for f_ in FAILS:
    print(f"  - {f_}")
print("=" * 72)
sys.exit(1 if FAILS else 0)
