# Resume notes — Thu 01 Oct 2026 (V2 repo bring-up + Training composition rebuild)

Supersedes `RESUME_NOTES_2026-09-23.md` for the V2 repo. That file remains the record
for the original `cycling-performance-ml` repo.

---

## 1. Where things live

| Thing | Value |
|---|---|
| New repo | `rubenfm77/CYCLING_PERFORMANCE_ML_V2` (branch `main`) |
| Cloud app main file | `app_modern.py` |
| Cloud secrets | `INTERVALS_ATHLETE_ID`, `INTERVALS_API_KEY` (Settings → Secrets) |
| Local clone | `C:\Users\levod\Downloads\CYCLING_DATA\PY\V_CYCLING` |
| Original, untouched | `C:\Users\levod\Downloads\CYCLING_DATA\PY\cycling-performance-ml` |
| 43 tracked files, 7 pages | Overview · Fitness · Forecast · Intervals · Training · Trends · Sessions |

Old Cloud app `cycling-performance-ml-gwl7kzbkctmdgnatg2jvnt` returns 403 — superseded.
V2 was created as a clean copy rather than a repair of the old app; `app.py`, `src/` and
`main.py` in the ORIGINAL folder were never modified.

**Status at end of session: Training page green, 7/7 pages rendering, no errors.**

---

## 2. THE `.gitignore` TRAP — read this first

`.gitignore` shipped with `data/*.csv`. That silently excluded
`data/combined_training_data.csv` from the repo, so Cloud loaded **37 rows covering
2026-07-28 → 2026-09-26** instead of **1,162 rows covering 2019-11-16 → 2026-09-26**.

Every "the charts only show last months / the April and May FTP sessions are missing"
symptom traced back to this one line. It is now `data/interval_cache.csv` only, and the
CSV is committed (1,161 lines).

**Check `git ls-files data/` after ANY data change.** A missing data file produces no
error, no warning, no empty state — just a confident, wrong chart. This is the failure
mode that cost the most time in this session.

---

## 3. Diagnosing a stale Streamlit Cloud deploy

The nastiest failure in this session: four consecutive "fixes" pushed, four identical
`KeyError`s. The cause was **Cloud serving a build three commits old** — Streamlit Cloud
does not hot-reload, and a redeploy that silently fails keeps serving the previous bundle.

**The method that settles it in one step:** read the offending line out of the traceback,
then diff it against each commit.

```powershell
foreach ($c in @('45fc7de','6a4e4bf','a0ed513','7aa3031')) {
  (git show "${c}:views/training.py" | Select-Object -Skip 189 -First 1)
}
```

The traceback said line 190 was the `...endswith("_drop")]]` list-comprehension. That text
matched `6a4e4bf` character-for-character and no other commit. Cloud was on `6a4e4bf`
while HEAD was three commits ahead. Confirmed stale build, not a code bug.

Corollary: if a traceback points at a line whose text does not match your current file,
**stop reading the stack and go check the deploy.** Do not patch again.

To recover: Cloud → app → **Deployments** → **Rerun** (~2–3 min, reinstalls deps), then
`Ctrl+Shift+R`. Rerun keeps secrets and settings; Delete does not, so it is the fallback
only.

**Was it a Python 3.14 / Streamlit incompatibility? No.** The `KeyError` came from pandas
`DataFrame.__getitem__` with a *list* key (`df[[cols]]` → `columns.get_loc(list)`, which
cannot hash a list). That code was my own bad patch and was broken on every Python
version. A genuine version incompatibility would fail on the *current* line, not a deleted
one. Do not re-open this.

---

## 4. Why the error "looked like it was on another page"

`app_modern.py:87` is `nav.run()`, the single line that dispatches to the selected page.
A crash inside `views/<page>.py` aborts the **entire** run. Everything Streamlit already
rendered survives — header, nav pills, the top half of the page — and only the unrendered
remainder is replaced by the red error box.

So a crash at the bottom of Training looks like "the page works fine, something else is
broken". It is not. The file named in the traceback is the page.

---

## 5. `views/training.py` — composition section rebuilt

`render()` had three real bugs in a row, not one:

1. `src.config` was missing → `core/theme.py:6` import killed the whole app. Fixed by
   adding `src/config.py` (a copy supplying `ATHLETE`, `MAIN_TYPES`, `ZONES`,
   `TYPE_COLOURS`).
2. Merge on `period` (str) against a Period-object `_month` → `KeyError`.
3. Every month was skipped when `total_tss == 0`, so `_outcome_df` was empty and
   `pd.DataFrame(_outcome_rows)` had no columns.

Then three bad defensive patches on top (suffix columns, list-comprehension column drop)
each raised a fresh `KeyError`. The rebuild removed all of it rather than adding a fourth
layer.

**The rewrite** (`7aa3031`) — no `merge()` anywhere in the block:
- monthly denominator comes from `_mt.groupby("_month")["type_tss"].transform("sum")`,
  so there is no second frame and no column that can be lost;
- `_comp_all` uses `ctx.df_all` (full history) deliberately — "does the mix predict FTP?"
  is a long-term question and must not inherit the sidebar date window;
- month keys unify as `Period` (`_mt["_month"].dt.to_timestamp()`, and
  `pd.PeriodIndex(..., freq="M")` on the outcome side) instead of string juggling;
- `if/else` on `_outcome_df.empty` keeps `ftp_proxy` / `ftp_gain` present so the tables
  downstream always have their columns;
- debug expander removed — the data question it answered is settled;
- honest n-line replaces the hardcoded "≈72 observations": now
  `1,004 labelled sessions across 76 months (Nov 2019 → Sep 2026)`;
- pattern table now flags `⚠ n<5` (combo table already used `n≥3`) so a 3-month average
  cannot read as a finding.

**Verified locally before pushing** — do this every time, it is fast and it is the only
reason this landed on the first deploy. Script pattern lives in
`%TEMP%\opencode\test_composition.py`: run the block's exact logic against the real CSV.

```
rows: 1159  2019-11-16 -> 2026-09-22
caption -> 1,004 labelled sessions across 76 months (Nov 2019 -> Sep 2026)
pct sums per month (max dev from 100): 1.42e-14
chart1 span: 2019-11-01 -> 2026-08-01
outcome rows: 76 | ftp_gain non-null: 75
Apr/May present: 2020-05, 2021-04/05, 2022-04/05, 2023-04/05, 2024-04/05, 2025-04/05, 2026-04/05
```

Locally the pattern table is mostly thin, which is why the `n` flag matters:
`Single: FTP` n=19 (+1.5 W), `Single: SST` n=12 (+1.5 W), `Single: VO2MAX` n=8 (+0.4),
`No quality sessions` n=16 (−4.7), `Mixed: PIRAMIDAL+SST` n=2 (−10.0). Only the FTP and
SST rows earn a reading. `ftp_proxy` is best NP × 0.95 — a proxy, labelled as one
everywhere, never a measured threshold test.

---

## 6. DFA α1 is NOT buildable — closed, do not re-litigate

Four independent probes, all negative:
- `?types=rr` → HTTP 422 `Invalid stream type`. intervals.icu has no RR stream.
- `heartrate` stream is 1 Hz: 11,466 samples / 11,505 s ≈ **1.95 samples per beat** at
  117 bpm. Far too coarse for DFA.
- `average_dfa_a1` exists in the `Interval` schema but is **NULL on all 471** intervals
  pulled, as are `average_respiration`, `average_tidal_volume`, `average_epoc`,
  `average_smo2`, `average_lactate`, `average_thb`.
- `wellness.hrv` and `wellness.hrvSDNN` exist but are 100 % null.
- Hardware: Wahoo + COOSPO strap supplies no RR.

**Correction that was made to the user:** activity-level `variability` (n=160) is
intervals.icu's **power** variability index, *not* an HRV index. Proven by recomputing
CV of the 5 s rolling power from the raw `watts` stream — **Pearson r = 0.9704**.

Shipped instead, from fields `_norm_iv()` was already discarding: `ss_cp` / `ss_w_prime` /
`w5s_variability` → `ss_cp_w`, `ss_w_prime_kj`, `w5s_cv`. Units established empirically:
`ss_cp_w` = watts (129 W on a 4 h ride), `ss_w_prime_kj` = kJ, `w5s_cv` = fraction
(0.035 = 3.5 % CV). `ss_cp`/`ss_w_prime` are only trustworthy for long (>20 min) variable
intervals — on steady threshold reps they are model noise, which the caption already says.
Force sync pulled 3,674 rows at 99.6 % coverage.

---

## 7. Environment gotchas that cost time

- **`&&` is a parse error in this PowerShell.** Run `git add` / `git commit` / `git push`
  as separate calls.
- **`rm -rf` and `ls -la` do not exist.** Use `Remove-Item -Recurse -Force` and `dir`.
- **A successful `git push` still exits 1** when stderr is merged in (`2>&1`), because
  git writes progress to stderr. Judge success by the `main -> main` line, not the code.
  `git add` does the same with its `LF will be replaced by CRLF` warning.
- `git commit -m "multi line"` breaks on spaces — write the message to a file, use `-F`.
- **The model cannot read images.** `read` on a PNG returns "this model does not support
  image input", and `browser.screenshot` fails with "Screenshot needs a visible tab".
  Every UI question must be asked as a *text* question ("do you see this exact caption?").
  Do not promise to look at a screenshot.
- Streamlit does **not** hot-reload `views/*.py` locally either — kill every streamlit
  python and relaunch after a view edit, and make sure only one process is bound to the
  port (two processes serving pre-edit code cost a whole session once).
- `_CODE_TAG` in `views/intervals_view.py` is inside the `st.cache_data` key of the two
  cached entry points. Streamlit hashes **arguments, not function bodies**, so without a
  bump an edit inside `ml/` keeps serving old numbers for the full TTL. Bump it when
  `ml/type_comparison.py` or `ml/protocol_reps.py` change.

---

## 8. Standing rules — do not regress these

- Duration = length of **ONE** interval, **whole minutes, halves down**; sub-90 s keeps
  seconds; exact seconds always shown alongside. Single lever: `fmt_min` / `fmt_secs` in
  `ml/type_comparison.py`.
- **Never compare across training types or duration classes.**
- Thousands separators on big numbers.
- GitHub-dark + blue identity. Legends inside the canvas but in a margin lane, never over
  data. Never promise a line the chart does not draw.
- Do NOT touch `show()` / `style_figure` / `legend()` globally.
- ML honesty: time-ordered validation, persistence baseline, surfaced coefficients, n
  always shown, descriptive/associational only. "Showing nothing beats showing noise."
- Push must never expose the athlete ID or API key.
- **Patch blind never again.** Verify the block locally against the real CSV first, then
  push, then confirm the *caption/figure* the user should see.

---

## 9. Open items — carried forward, none started

1. **Unpinned build environment.** No `runtime.txt`, and every line in
   `requirements.txt` is `>=` with no ceiling, so Cloud can silently move to a different
   Python and pandas between rebuilds. Not today's bug, but it removes reproducibility.
   Pin the Python version and cap the majors. *Offered, not started.*
2. **History purge** of `data/combined_training_data.csv` + `data/power_curve.csv`.
   *Offered twice, never accepted, never performed. Do not do it unasked.*
3. **Pattern → FTP ML against the real rolling eFTP**, not the best-NP proxy. The
   pattern table is descriptive right now.
4. **`.rsc.json` plan** — add `"cycling-coach"` to `ownSkills` (currently `[]`). Show the
   exact plan and **WAIT** for acceptance.
5. **"28 weeks projection with weekly pattern of flats"** — should 4 / 8 / 28 / 52 weeks
   become first-class in the horizon selectbox, or stay 28-only? *Unanswered question.*
6. **Screenshot-unreadable** — the `browser.preview` path exists in the tool catalog; it
   was never tried as a workaround for reading UI images.

---

## 10. Addendum, same day — FTP auto-label + duplicate rides (commit `0db0799`)

Two plumbing bugs found while trying to make the user's real threshold work visible.
Both were inflating the Training page; both are fixed in `core/data.py`.

### 10a. The same ride was counted twice, 63 times

Nothing in the frame said "duplicate". The pairs have **different intervals.icu ids** and
different `source` (`WAHOO` = head unit, `UPLOAD` = manual upload of the same ride), so
`drop_duplicates` on any id column is blind to them. They also differ in TSS by up to 50,
because only the head-unit copy carries heart rate — so they are not byte-identical
either. What *is* identical: duration to the second, and distance to the centimetre.

Rule now in `load_data()`: same calendar day, duration within `DUPE_DUR_TOL_S` (90 s), and
**power within 3 W OR distance within 1 m**. The distance branch is not redundant — 10 of
the 63 pairs carry no `power_avg` on either copy, and those are exactly the rows a
power-based test cannot see. The more complete record wins, which keeps the head-unit copy
and its heart rate.

Plus 8 byte-identical rows in 2022/2023/2025, caught by an exact-key `drop_duplicates`.

| | before | after |
|---|---|---|
| rows | 1,164 | **1,093** |
| total TSS | 151,539 | **143,334** (−5.7 %) |
| hours | 2,947 | 2,764 |
| km | 106,426 | 102,374 |
| CTL / ATL / TSB | 130.3 / 135.4 / −5.1 | 130.2 / 135.4 / −5.2 |

A 5.7 % overstatement of lifetime volume. The 68 real two-ride days are untouched: 5
survive, no genuine pair lands inside both windows, and zero near-duplicates remain
(verified two ways).

**The measured threshold series did not move** — the best ~20 min number in all 10 months
is identical before and after. What was wrong was the effort **count** beside it: Mar 14 →
7, May 14 → 7, Apr 8 → 4. A lesson worth keeping: a duplicate that only ever appeared as a
*count* and never as a *value* will never show up in a before/after diff of the headline
number. Check the denominators.

### 10b. The FTP auto-label — one rule, declared and listed on the page

`_match_type` only trusts intervals.icu workout-name fields, so 110 sessions had no type and
appeared in **no** type-based view. Yesterday's ride (`Pepper it up`, 244 W peak over 22
min) was one of them. 9.2 % of all TSS sat in that blind spot.

Rule, applied only where there is no label: the session contains an **18–25 min**
peak-meter effort at **≥ 95 % of that session's own eFTP**. Per-session eFTP, never today's
— it is a rolling value, and applying the current one to an old ride is anachronistic.

The window and the fraction now live in `core/data.py` as `FTP_AUTO_MIN_S`,
`FTP_AUTO_MAX_S`, `FTP_AUTO_FRAC` and are **imported by `views/training.py`**, so the label
a session gets and the number the measured-threshold chart plots cannot drift apart.

It catches **7** sessions. That is the honest answer, not a shortfall — the recent log is
dominated by 7.5 min efforts at 120 % of eFTP, which are correctly *not* threshold, and
below 18 min the peak-meter number is not comparable across rides at all.

Nothing is overwritten silently: `training_type_raw` keeps what the API said, `label_source`
records `workout` / `auto-ftp` / `unlabelled` for every row, and the page carries a
disclosure listing all 7 caught sessions with their window and % of eFTP.

**Still open, deliberately.** The 110 unlabelled sessions are mostly steady "ronnestad"
rides. Labelling those by intensity factor is a much larger inference than the one rule
above, so it is *not* done. It is a decision for the user, not a guess to make.

### 10c. Test-harness gotcha

`st.expander` / `container` / `tabs` / `form` / `popover` are context managers. The stub in
`test_render.py` returned `None` for them, so **the body of every `with` block was silently
never executed** — the new disclosure rendered nothing under test and the failure surfaced
only when the stub was fixed. Any new UI code inside a `with` block is untested until the
stub handles it.

---

## 11. Addendum - the loader was throwing labels away (commit `652ede0`)

The athlete labelled 57 sessions from the Oct 2025 - Sep 2026 gap (AEROBIC BASE 36,
FTP 13, BILLAT 4, VO2MAX 3, PIRAMIDAL 1). All 57 were written to the CSV correctly and
**47 of them were then deleted by `load_data()` before any view could see them.** Two
independent bugs, both in `core/data.py`:

### 11a. The 60-day API overlay dropped every label inside the window

```python
api_dates = set(recent["date"].dt.date.astype(str))
base_df = base_df[~base_df["date"].dt.date.astype(str).isin(api_dates)]
df = pd.concat([base_df, recent], ignore_index=True)
```

Whole-**date** replacement, not row-level. The API row cannot carry a label - the account
has zero workout documents, so `_match_type` returns the blank token for all 37 rows.
So every session categorised inside the trailing 60 days lost its type on every reload.
That is the most recent two months: the present, not the history.

Fix: `_carry_csv_labels(base, recent)` copies the CSV's label onto the API row *before*
the CSV rows are dropped, matched on the same calendar day and the same 90 s duration
window as the duplicate rule (`_dur_seconds()` supplies the duration because `duration_s`
does not exist yet at that point - it is built later, after the merge). Count is exposed
as `df.attrs["labels_carried"]` = 32.

### 11b. The duplicate rule traded a label away for a heart rate

Survivor selection was `notna().count()` only. For the WAHOO-head-unit / manual-UPLOAD
pairs the head-unit copy has heart rate, so it won - and took the categorised twin's label
down with it. Worked in **both** directions depending on which copy was labelled.

Fix: survivor selection is unchanged (so no TSS, HR or distance number moves), but when
exactly one copy of a pair is categorised, that label is *donated* to the survivor.
`df.attrs["labels_recovered"]` = 15.

### 11c. `astype(str)` no longer stringifies NaN in this pandas - `.notna()` is load-bearing

```python
pd.Series(["END", float("nan")], dtype=object).astype(str)   ->  ['END', nan]   # NOT 'nan'
```

So `.isin({"nan", "", "-", "—"})` alone reports a **missing** label as a **real** one.
That silently gave every NaN row priority in the duplicate rule and made the 11b fix look
like it made things worse. The pre-existing `_has_label` was safe because it is guarded by
`_tt.notna()`; the new code needs the same guard. Grepped the repo: every other
`astype(str)` either `fillna("")` first or formats dates/ids, so nothing else is affected.

### 11d. Editing a committed CSV without churning it

A normal `read_csv` / `to_csv` round-trip rewrote ~400 unrelated cells at the 16th
significant digit and appended `.0` to integer ids (`strava_id`, `power_meter_serial`).
`dtype=str, keep_default_na=False` round-trips byte-identically and leaves a diff of
exactly the 57 edited cells. Use that whenever only a few cells of a committed data file
change. Verified with a per-column raw-text diff: only `training_type` moved, 57 cells,
all previously blank, **0 pre-existing labels overwritten**.

### 11e. Result

| | before | after |
|---|---|---|
| rows | 1,093 | 1,093 |
| TSS | 143,334 | 143,334 |
| labelled sessions | 992 | 1,039 |
| unlabelled | 86 | 54 |
| your 57 labels visible in-app | 10 | **57** |

8/8 tests pass; real `render()` draws 8 figures. Measured threshold series unchanged
(Dec 2025 202 W -> Sep 2026 244 W) - labels do not touch it.

BILLAT now has a 2026 row for the first time, which is what the year-over-year comparison
needs. Note the athlete's definition: BILLAT = "ronnestad", reclassified so it can be
compared with previous years - a long steady ride **ending with a short hard interval**.
All 4 new BILLAT sessions have a peak-meter best window of exactly 7.5 min at 112-116 % of
eFTP, which matches that shape precisely.

---

## 12. Addendum - the Evolution page (commit `f075fb5`)

`views/evolution.py` + `ml/year_over_year.py` + `tests/test_year_over_year.py`.
Registered as the 8th page in `app_modern.py`. Session grain (whole rides), which is
the grain the athlete's own labelling exists at - distinct from `ml/type_comparison.py`,
which is per-REP grain. Do not mix the two.

### 12a. The measurement that decided the design

Before writing any UI, count the cells. n >= 3 per (type x duration class x year):

| type | drawable cells | years |
|---|---|---|
| AEROBIC BASE | 22 | 8 |
| FTP | 15 | 7 |
| END | 17 | 6 |
| VO2MAX | 9 | 6 |
| FATMAX | 7 | 6 |
| SST | 7 | 5 |
| TORQUE | 7 | 4 |
| BILLAT | 5 | 4 (2021 2022 2025 2026) |
| TEMPO | 5 | 3 |
| PIRAMIDAL | 3 | 3 |
| Q-I INTERVALS | 2 | 1 - too thin, says so |

### 12b. Year-over-year WATTS does not exist. Half the original ask.

Threshold watts are **measured**, and the measurement starts **Dec 2025**. 138 readings
total: 3 in 2025 (all inside one December week) and 135 in 2026, 10 months. So:

- 2019-2024 have **no** threshold number to compare against;
- 2025's 3 readings are one week of one month.

There is no weak or modelled version of this comparison available, so the page states
the wall with the counts instead of drawing a watts line. `watts_gate()` returns
`can_compare_years = False` and the test asserts it.

**Gotcha that cost a cycle:** the first version of the gate used "a year with >= 3
readings counts as comparable". That declared 2025 comparable on 3 December-week
readings and then claimed a comparison existed. The gate now needs BOTH
`MIN_WATTS_YEAR_N = 12` readings AND `MIN_WATTS_YEAR_MONTHS = 3` distinct months.

### 12c. Decisions worth not relitigating

- **`power_np` is not offered as a measure.** Present for 139/139 sessions in 2020 but
  only 30/141 in 2026, so a series on it stops mid-2026 for a data reason, not a
  physiological one. `power_avg` has 1,039 rows across every year. (This is the same
  column that produced the fraudulent old FTP proxy - do not reach for it.)
- **The page reads `ctx.df_all`, not `ctx.df`.** The sidebar range defaults to
  "Last 6 months"; applying it here would delete 2019-2025 and leave a page whose
  entire subject was filtered away. The page says so in the header rather than
  silently ignoring the control.
- **`MIN_CELL_N = 3`, and a thin year gets NO point at all.** `connectgaps=False`
  then breaks the line there. Interpolating across a missing year is the "promising a
  line the chart does not show" failure.
- **"vs previous year" means the previous year in the SAME duration class that
  cleared the floor**, not year-1, and the year gap is printed. BILLAT 2022 -> 2025
  reads "3y". Without this a 3-year change reads as a 1-year change.
- **Session duration classes** are `under 45 / 45-90 / 90-150 / 150-240 / 240+` min.
  The whole-minute halves-DOWN rule governs REP lengths; at session scale a minute
  grid is meaningless. Exact `h:mm:ss` is reported alongside every cell.

### 12d. A second blank-label definition bit, and it would have been invisible

`ml/year_over_year.py` first defined its own blank-label test and kept **1,043**
sessions. `core.data` keeps **1,039**. The difference is the em dash in
`BLANK_TYPE_TOKENS = {"—", "-", "", "nan", "None"}`, which the local copy did not
have - plus the local copy's case-insensitive match would have pulled in extra
tokens the loader keeps.

The module now imports `BLANK_TYPE_TOKENS` from `core.data` and uses the loader's
exact predicate, and the test asserts parity (`1039 == 1039`). Same class of bug as
11c: two definitions of "missing" drifting apart, producing a *confident* wrong count
rather than an error. `ml/composition_intervals.py` already imports from `core.theme`,
so the `ml -> core` dependency is precedented.

### 12e. Verification

- 862 drawn points across every type x measure - all on cells clearing the floor, no
  duplicates, none outside the declared class set.
- 70 deltas, each with both years above the floor and the stated gap matching the two
  years actually compared.
- Real `render()` under stubbed Streamlit for AEROBIC BASE (2 figures / 21 markers),
  END (17), BILLAT (5) and Q-I INTERVALS (correct early return, 0 figures).
- Legends land in a margin lane (`y=1.0`, horizontal, top margin 74 -> 96), never over
  data, via the existing `_lane_legend` imported from `views/intervals_view.py` - not a
  second implementation of the lane rule.
- Two defects the render dump exposed and fixed: the exact-length table was listing
  years that were NOT drawable (a 1-session median length next to a 37-session one),
  and a column header read "Median average power".
- 9/9 tests pass, including the 8 pre-existing ones.

### 12f. Still not answerable

Pre-2025 threshold watts do not exist, so year-over-year *threshold* comparison is
impossible; only training-side comparison is real. Heat remains step 2 and is NOT a
third dimension of this page: adding heat bands to (type x duration x year) leaves
only 4 cells with n >= 15, so it would destroy every line. Heat gets its own section,
within one type and one duration class only, and `efficiency` (W/bpm, 757 rows) is the
only fitness-normalised heat-response variable that exists - `power_avg` alone is
confounded by eight years of fitness gain.
