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
- **PowerShell has no heredoc.** `git commit -F - <<'MSG'` is a parse error at `<<`.
  Write the message to a temp file first, then `git commit -F <path>`.
- **`core.autocrlf=true` and there is NO `.gitattributes`.** Measured 2026-10-01,
  and it matters for any future CSV edit. The committed blob of
  `data/combined_training_data.csv` is fixed at
  `c67b429900b9282125d4a19ecd995b7ff09e0540`, but the *working tree* is not
  reproducible from it. Byte-level:

  ```
  local checkout : 1,352,079 bytes  CR=1160  LF=1164  CRLF=1160  bareLF=4
  fresh clone    : 1,352,083 bytes  CR=1164  LF=1164  CRLF=1164  bareLF=0
  ```

  Same blob, 4 bytes apart on disk: autocrlf promoted the 4 bare LFs inside
  quoted description fields to CRLF on checkout. `git status` still calls the
  file clean in both, because git compares against the filtered form.
  **Never hardcode a CRLF count or a byte length as a CSV guard.** Measure the
  baseline in the current checkout, or better compare `git rev-parse
  HEAD:data/combined_training_data.csv` before and after — that is
  filter-independent. Verify remote state with `git rev-parse HEAD:<path>` on a
  clone, never by hashing the working tree. The 4 bare LFs are embedded
  newlines inside quoted `WorkoutDescription` fields and are legitimate
  content, not damage.
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

---

## 13. Addendum - the app-wide ImportError, and the amplifier behind it (commit `78d4e8b`)

The Evolution page took the WHOLE dashboard down, not just itself. All 8 pages
went to a red screen. Worth understanding, because it will happen again.

### 13a. What Cloud reported

```
app_modern.py line 31    from views import (evolution, fitness, forecast, ...)
views/evolution.py:31    from ml import year_over_year as yoy
ml/year_over_year.py:104 from core.data import BLANK_TYPE_TOKENS
ImportError
```

### 13b. Why one missing constant killed 8 pages

Look at the order inside `app_modern.py`:

| line | what happens |
|---|---|
| 31 | `from views import (evolution, fitness, ...)` - every page module imported |
| 67 | `st.navigation(_pages)` - page registry BUILT here |
| 90 | `nav.run()` - pages actually render here |

Every `views/*` module is imported at line 31 while the registry is being built,
long before any page runs. So a module-level ImportError inside ANY view is an
**app-wide** crash. The 7 pages that were perfectly healthy could not render at
all. A crash *inside* `render()` is page-local (everything already drawn
survives); a crash at *import* time is fatal to the run.

### 13c. Why it failed at all

`origin/main` == `HEAD` == `2b5d0fe`, and `core/data.py` on origin DOES define
`BLANK_TYPE_TOKENS` (line 51, added in `652ede0`). Verified by reading the blob
straight out of the remote. So the Cloud checkout was internally INCONSISTENT:
the new `ml/year_over_year.py` sitting on a `core/data.py` older than
`652ede0`. A stale-partial redeploy, not a bad commit.

Recovery was Cloud -> Deployments -> **Rerun** (keeps secrets; Delete does not),
then Ctrl+Shift+R.

### 13d. The fix, because "just Rerun" is not a fix

That import should never have been able to do this.

- `BLANK_TYPE_TOKENS` now lives in **`core/theme.py`**, whose only import is the
  leaf `src/config`. `core/data.py` imports it from there and **re-exports the
  same object** under the same name, so every existing
  `from core.data import BLANK_TYPE_TOKENS` keeps working and there is still
  exactly ONE definition.
- `ml/year_over_year.py` imports it from `core.theme`. The registry import can no
  longer be killed by anything living in `core/data.py`.
- **Why `core/theme.py` is the only correct home:** `core/data.py` imports from
  `core/theme`, so nothing in `core/` may import an `ml/` module without a
  circular import. A pure constant that `ml/` needs has to sit upstream of
  `core/data.py`, and `core/theme.py` is that layer.
- `tests/test_year_over_year.py` asserts the re-export IS the same object as
  `core/theme`'s. A second literal in `core/data.py` now FAILS the suite instead
  of silently costing 4 sessions (1043 vs 1039) the way it did in 12d.

### 13e. Deliberately left alone

`views/training.py` (`FTP_AUTO_*`) and `views/trends.py` (`safe_sum`, `safe_mean`)
still import `core/data.py` at module level. Those names have existed since the
repo began, so no plausible checkout skew removes them, and rewriting them would
be churn against an already-negligible risk. The dangerous pattern was importing
a NEWLY ADDED name that way, and there is now none.

### 13f. `load_data()` output is NOT reproducible across processes

`load_data()` merges a LIVE 60-day intervals.icu fetch over the local CSV, so:

| quantity | observed across runs minutes apart |
|---|---|
| rows | 1,088 - 1,093 |
| TSS | 143,334 (stable) |
| threshold readings | 138 - 144 |

Stable WITHIN one process, different BETWEEN them. My first verification script
asserted those numbers exactly and flapped, which cost a cycle and briefly looked
like the constant move had changed the data. **Never assert an API-derived count
across runs.** Assert things the CSV owns: labelled sessions (1,039) and the
agreement between the loader's blank-label predicate and the module's own.

This also means any threshold number quoted from this pipeline is a snapshot. The
STRUCTURAL finding is stable because 2025's 3 readings come from the CSV, outside
the 60-day overlay window - the overlay can only add 2026.
---

## 14. Addendum - second occurrence, and the amplifier is now removed (commit `fdeedd1`)

The same ImportError came back after `78d4e8b`, one module over:

```
ml/year_over_year.py:104  from core.theme import BLANK_TYPE_TOKENS   ->  ImportError
```

### 14a. The remote was verified BEFORE any code was touched

| check | result |
|---|---|
| `origin/main:core/theme.py` defines the constant | YES, line 26 |
| fresh `git clone` -> import `core.theme`, `core.data`, `ml.year_over_year`, `views.evolution` | all OK |
| anything from my working tree on `sys.path` during that clone test | none |
| line 104 of the CLONED `ml/year_over_year.py` | exactly the traceback line |

So Cloud *had* picked up `78d4e8b`. **This was never a code defect.**

### 14b. Why an inconsistent Cloud checkout is not reachable from git

`78d4e8b` moved `core/theme.py` and edited `ml/year_over_year.py` in **the same
commit**. A git checkout cannot have one file from a commit without the other.
Therefore the Cloud tree was inconsistent in a way git itself cannot produce -
a stale/partial container, not a bad commit.

**Lesson: verify the remote by cloning it, not by reading the local working
tree.** The local tree always reflects the fix and proves nothing.

### 14c. The fix is the amplifier, not the instance

`app_modern.py` pulled in all eight pages with one statement:

```python
from views import (evolution, fitness, forecast, intervals_view, overview,
                   sessions, training, trends)
```

executed at line 31, while BUILDING THE PAGE REGISTRY - hundreds of lines before
`st.navigation()` at line 67. A module-level ImportError there is an **app-wide**
crash. That is the whole reason a missing constant in one leaf module could blank
all eight pages.

Now each view is imported on its own via `importlib.import_module`:

- a view that fails to import is replaced by a stand-in that says so and names
  the exception;
- a `st.warning` names the broken page on **every** page;
- the other seven render normally;
- `except Exception`, not `except ImportError` - the point is not to know in
  advance what a page module may raise at import time.

Consequence: the class of failure that produced two app-wide outages can now only
ever cost **one** page, visibly.

### 14d. The stand-in withholds anything that might not be safe to display

Names the exception TYPE always. Shows the MESSAGE only for `ImportError`, which
is a name-or-path resolution failure and cannot carry athlete data. Any other
message is withheld and the user is pointed at Manage app -> Logs. This keeps the
standing rule that a push must never expose athlete ID or API key, even in an
error path.

### 14e. `tests/test_page_isolation.py` - testing the router, not a copy of it

The router's mechanism is `importlib.import_module(f"views.{name}")`, so patching
`importlib.import_module` **before** `app_modern` runs makes it genuinely fail for
a chosen module. That exercises the real router under a stubbed streamlit. Four
cases: healthy, `ImportError`, `RuntimeError`, and a `RuntimeError` carrying a
secret-looking string that must not reach the UI.

Every failure case asserts: 8 pages still registered, 7 still wired to *real*
module renders, the 8th wired to the stand-in, the stand-in renders without
raising, and a warning is on screen.

Suite is now **10/10**.

### 14f. Traps hit while writing that test (harness bugs, not app bugs)

- `functools.partial` keeps the callable in `.func`; `.args` holds
  `(head, ctx)`. Asserting on `fn.args[0]` silently inspected `head`.
- A streamlit stub used via `with st.sidebar:` needs `__enter__`/`__exit__`; a
  plain function is not a context manager.
- `except Exception` in the test harness needed the exception TYPE, not an
  instance (`type(exc).__name__`), because instances have no `__name__`.

All three produced confusing failures that looked like app defects. Same lesson
as before: when a check fails, first ask whether the check or the code is wrong.

### 14g. Deployment state

`fdeedd1` is pushed. If Cloud is still serving an inconsistent tree, the new
isolation means the seven healthy pages will load and Evolution alone will show
the "could not be loaded" panel - which is itself the diagnostic. Recovery is
Deployments -> **Rerun**; if that does not clear it, Delete + redeploy, which
**loses the secrets and requires re-adding them**.

## 15. Interval watts, not average watts

The user rejected the measure the whole Evolution page had been built on:
"i dont give a fuck about average watts". This section records what replaced it,
and the two facts that constrain it permanently.

### 15a. The six "unlabelled" ronnestad rows were duplicates, not sessions

`probe_billat_fix.py` established that the six shape-matched rows with a blank
`training_type` are exact duration twins (0 s apart) of records that survive
`load_data()`'s dedup:

| blank record | date | surviving twin | label |
|---|---|---|---|
| i156711936 | 2026-04-07 | i137901296 | VO2MAX |
| i156712011 | 2026-04-15 | i140095919 | FTP |
| i140666657 | 2026-04-17 | itself, label donated | FTP |
| i156712053 | 2026-04-18 | i140867318 | AEROBIC BASE |
| i156712110 | 2026-04-22 | i142054141 | FTP |
| i144107462 | 2026-04-29 | itself, label donated | FTP |

So "relabel the six unlabelled" is a **no-op** - the app already shows those
rides under VO2MAX/FTP/AEROBIC BASE via the surviving twin. No CSV edit was made.
The user then confirmed: **leave all 23 shape-matched sessions as they are**.
BILLAT stays at 30 sessions across 2020/2021/2022/2025/2026.

The 23 sessions the app shows for that shape are FTP 10, AEROBIC BASE 9, END 2,
PIRAMIDAL 1, BILLAT 1.

**The `data/` CSV is byte-identical to the last commit.** The `.bak` has been
deleted.

### 15b. Two sources that must never be joined

```
prescribed   WorkoutDescription, 239 parsed sessions, 2019-2025, ZERO in 2026
measured     interval_summary,   639 detected efforts, 2025-2026
```

They cover different years and meet in exactly one. A line through both would be
interpolating 2025-2026 out of nothing. They are separate functions, separate
censuses and separate tabs, and the prescribed series must **stop at 2025** -
carrying the last value forward is the single most tempting dishonesty in the
module, so `coverage_note()` returns `prescribed_last_year` explicitly and the
chart is expected to honour it.

### 15c. Cell design, chosen by census rather than preference

The forbidden comparison here is subtle: `3x1'` and `2x20'` are both "interval
watts". So the bucket is REP LENGTH, not session length. Two candidates were
measured before either was built:

```
A  (type x rep-class x year)                      34 drawable cells, 6 types >=2 years
B  (type x rep-class x year x session-dur-class)  35 drawable cells, 4 types >=2 years
```

**A wins.** Session-length pooling is disclosed per cell (`dur_mix` column,
surfaced in the hover) instead of being used as the bucket. Measured 2026 gives
26 drawable cells on `(type x rep-class)`.

Rep classes: `under 90s / 90s-5min / 5-10 min / 10-20 min / 20-30 min / 30+ min`.
`under 90s` is deliberately not subdivided and the exact seconds travel with
every row.

### 15d. Parser rules that must not regress

- `'` is **minutes**, `"` is **seconds**. Separate regex alternatives, never
  merged. Reading `10"` as ten minutes inflates a sprint session 60x.
- The gap between rep length and wattage may contain words but **not digits**.
  That is what stops `4x15' pujada (10' recup -150w)` from borrowing the
  recovery's watts.
- Refused rather than guessed: `pols` (cadence), `10k/h` (speed), compound
  structures where several efforts share one rep.
- `MS:` scopes the main set; `WU:`/`CD:`/`MD:`/`FINAL:` are cut off.
- Floors: rep >= 30 s and >= 100 W.
- A reversed range is ordered, not emitted as a negative span.

### 15e. The numbers, verbatim from the verified run

Prescribed target watts, lower end, median per (type x rep class x year). Only
cells that cleared n >= 3 are listed:

```
BILLAT   under 90s   2021 330(n=4)  -> 2022 350(n=10) -> 2025 375(n=8)   [gap 3y, stated]
FTP      10-20 min   2020 225(n=5)  -> 2021 260(n=12) -> 2022 265(n=4)
FTP      90s-5min    2021 305(n=4)  -> 2022 282(n=6)  -> 2023 265(n=4)  -> 2024 290(n=13)
FTP      5-10 min    2020 275(n=4)  -> 2022 295(n=5)
SST      10-20 min   2021 240(n=6)  -> 2022 245(n=5)  -> 2023 245(n=5)  -> 2024 245(n=4)
SST      5-10 min    2020 225(n=4)  -> 2021 245(n=6)
SST      20-30 min   2022 245(n=4)
VO2MAX   under 90s   2020 330(n=3)  -> 2021 350(n=5)  -> 2023 300(n=4)  -> 2024 320(n=8)
VO2MAX   90s-5min    2020 290(n=4)  -> 2021 300(n=13)
FATMAX   10-20 min   2023 195(n=5)  |  20-30 min 2023 195(n=4)  |  30+ min 2023 195(n=3)
FATMAX   5-10 min    2025 195(n=3)
TEMPO    10-20 min   2024 225(n=11) |  20-30 min 2022 225(n=4)  |  90s-5min 2024 225(n=4)
TORQUE   90s-5min    2021 255(n=4)
SPRINTS  under 90s   2025 450(n=4)
```

SST's 10-20 min chain is genuinely **flat** at 245 W for three straight years
after 2021 - a real finding, not a broken chart. FATMAX and TEMPO hold one
number for years, which is what a fixed aerobic target looks like.

Measured detected efforts, 2026, median W by length class, n in brackets:

```
AEROBIC BASE  under 90s 272(n=81) | 90s-5min 214(n=140) | 5-10 min 206(n=55) | 10-20 min 200(n=6)
BILLAT        under 90s 365(n=13) | 90s-5min 198(n=8)
END           under 90s 269(n=17) | 90s-5min 217(n=41)  | 5-10 min 208(n=20) | 10-20 min 195(n=3)
FATMAX        under 90s 330(n=8)  | 90s-5min 217(n=12)  | 5-10 min 223(n=4)
FTP           under 90s 276(n=36) | 90s-5min 229(n=77)  | 5-10 min 229(n=40) | 10-20 min 241(n=12) | 20-30 min 195(n=3)
PIRAMIDAL     under 90s 258(n=5)  | 90s-5min 218(n=11)  | 5-10 min 210(n=9)  | 10-20 min 214(n=3)
TEMPO         90s-5min 253(n=3)
VO2MAX        under 90s 266(n=6)  | 90s-5min 247(n=13)  | 5-10 min 249(n=3)
```

The honest read of FTP 2026: the 90s-5min and 5-10 min classes sit at **229 W**
and only the 10-20 min class rises to **241 W**, from n=12. That is a coherent
threshold-ish profile, but n=12 detected efforts is not a threshold test and
must not be called eFTP.

These are **detected** efforts, not prescribed workouts: on a long ride the head
unit also reports the rolling sections it found, which is why AEROBIC BASE has
140 efforts in the 90s-5min column and why TEMPO's only cell is 3 efforts. The
page says so in words; it must never be presented as "the interval the coach
asked for".


### 15f. Files

- `ml/interval_watts.py` - new engine. `parse_prescribed`, `parse_measured`,
  `prescribed`, `measured`, `prescribed_cells/evolution/yoy`,
  `measured_cells`, `audit`, `coverage_note`, `rep_class`, `fmt_rep`,
  `fmt_watts`. `MIN_CELL_N = 3`, deliberately a separate constant so
  `year_over_year.py` cannot silently loosen it.
- `views/interval_watts.py` - new page, slug `interval-watts`, title
  "Interval watts", icon U+1F3AF. **9 pages now.** Three tabs: Measured,
  Prescribed, What could be read. Reuses `_lane_legend`.
- `tests/test_interval_watts.py` - new, **synthetic frames only**, so it needs no
  credentials and runs anywhere. Real-data verification was done separately by
  `verify_interval_watts.py` and `render_interval_page.py`.
- `tests/test_page_isolation.py` - the hardcoded 8/7 counts are now **derived**
  from `AM._VIEW_NAMES`, plus duplicate-name and duplicate-slug checks.
- `app_modern.py` - `interval_watts` added to `_VIEW_NAMES` and `_SPECS`.

### 15g. More harness bugs, same lesson

- `st.columns(...)` in a stub must return a **list** of context managers. A
  single object produces `'_Ctx' object is not subscriptable` and `cannot
  unpack non-iterable`, which read as page defects and were not.
- The stub also needs a real pass-through `cache_data`, because `core.data`
  decorates its loaders with `@st.cache_data` at import time.
- `duration_s` is **not a CSV column** - `load_data()` derives it. On the raw CSV
  use `duration_h * 3600`.
- My own check `y["gap_years"].dropna() == y["year"] - y["prev_year"].dropna()`
  compared two differently-indexed series. The column was right; the check was
  wrong. A synthetic fixture with a thin 2023 now pins the multi-year gap.

### 15h. Still open

- **Interval power for the eFTP target.** 241 W in the 10-20 min class for 2026
  is the closest thing measured, but it is a median of detected efforts in
  FTP-labelled rides, not a validated threshold test. Do not call it eFTP.
- **Heat analysis** - untouched. Within one type and one duration class only.
- **Weekly planner with rest weeks** - untouched, and must follow the heat and
  interval findings rather than precede them.
- The 15 custom label values still appear in no type-based chart.
- The 48 empty-shell sessions still need a decision on blanking their distance.
- The Intervals page is still the one the user calls unusable.

## 16. Intervals on the Evolution page, and the peak-meter that actually holds them

The instruction was: *"in evolution i dont give a fuck about average watts, i want
to isolate the intervals just like in training, have charts with watts and then a
line to compare between days. x axis date, y in bars intervals, and a line to
compare average watts between intervals and days. simple."*

The mistake in the previous round was building a separate ninth page. Intervals
belong where the athlete looks for evolution, so the Evolution page now opens on
them: **`_interval_by_day(ctx)` is the first tab**, and the year-over-year body
moved wholesale into a second tab. It was re-indented by script rather than by
hand, and `git diff -w` confirms the move is whitespace-only - 196 insertions and
4 deletions against 726 lines of raw churn.

### 16a. Why the 20-minute intervals "were not there": the wrong source

Three separate causes, all confirmed against real data before anything was
edited.

1. **The auto-detector is the wrong instrument.** `interval_summary` reports
   what is UNUSUAL inside a ride, not what was prescribed. Across 659 detected
   efforts the common lengths are 10-14 s and 55-84 s - the accelerations at the
   start of each rep and the rolling sections the head unit segments out. Exactly
   **one** detected effort in the whole file sits between 19 and 21 minutes, and
   it is 213 W.
2. **The real 20-minute record lives in the peak-meter fields**,
   `icu_pm_ftp_watts` with `icu_pm_ftp_secs` - a third source, session grain,
   that reports the highest sustained average AND the window it was sustained
   over. It has a reading on **all 27 FTP sessions of 2026**, and 144 sessions
   overall (2025: 3, 2026: 141). My earlier column probe filtered on
   `20|peak|best|micro|interval` and missed it, because the name contains none of
   those tokens. `core/data.py:373` already used it, and `year_over_year.py:175`
   already collected it.
3. **The boundary is half-open, so 20:00 lands in `"20-30 min"`.** That is
   defensible and is now documented at `REP_CLASSES` rather than left to be
   rediscovered, because it is exactly the boundary a reader assumes the other
   way round. Every bar and every table row also carries its exact length, so a
   20:00 effort is visible as `20:00` and never only as a class name.

**FTP 20-30 min, nine days: 214, 218, 226, 240, 188, 242, 231, 241, 244 W, of
which four are a whole 20:00** (2026-04-17, 07-23, 09-15, 09-22). FTP 10-20 min
adds ten more days at 223-248 W.

### 16b. The peak-meter is a real peak - measured, not assumed

Before writing "the highest sustained average" into a caption, it was tested
against the session's own average watts on all 144 rows:

| field | n | >= session avg | median gap |
|---|---|---|---|
| `icu_pm_ftp_watts` | 144 | **144** | +54 W |
| `icu_rolling_p_max` | 144 | 144 | +500 W |
| `ss_p_max` | 144 | **0** | -153 W |

`ss_p_max` is below the session average on every single row, so it is not a peak
of the session and was not used. `icu_pm_ftp_watts` is above it on every row, and
its ratio to the session average is ~1.34-1.40 across every session-length band,
i.e. it does not drift with ride length - which is what a peak should do.

### 16c. Three real bugs, and three of my own checks being wrong

Per the standing rule: when a check fails, first decide whether the check or the
code is wrong. It was my checks, three times.

- **Wrong check, twice.** I asserted `avg <= bar` and wrote it as
  `bar <= avg`. Separately I asserted a `20:00` bar while the harness had
  selected AEROBIC BASE by default - a real finding about the harness, not the
  chart. And `rep_class(1800)` is `"30+ min"`, not `"20-30 min"`; 1800 s is a
  whole 30:00.
- **Real bug: positional index leaking.** `effort_best()` rebuilds its frame
  from a list, so its index is positional. `avg.loc[idxs]` was therefore reading
  *whatever rows happened to sit at those positions* - the average watts beside
  each bar belonged to unrelated sessions. Fixed by carrying an explicit `src`
  column holding each session's own index in `df`, and pinned by a test.
- **Real bug: the picker could promise bars it would not draw.** `day_options`
  counted SESSIONS while `day_series` drew DAYS, so on a day with two rides the
  dropdown said "4 day(s)" and the chart drew 2 bars. Fixed by collapsing both
  onto one private `_effort_by_day()` frame; `days` and `sessions` are now both
  reported, and a test asserts the picker count equals the bar count and the
  picker median equals the median of the bars shown.
- **Real fragility, caught only by the synthetic fixture.**
  `out["year"] = pd.to_datetime(out["date"]).dt.year` raises on a column that
  legitimately mixes `"2026-04-17"` and `"2026-05-01 08:30:00"`. It is derived
  from the already-parsed normalised `day` now. This only failed on synthetic
  data - on the real file `date` is already a datetime - which is the argument
  for keeping credential-free tests.

A day holding two sessions keeps its **harder** effort, never the mean: the mean
describes neither ride, `n_sessions` records that it happened, and the session
average shown beside it is the mean of the day's sessions. Sessions are also
normalised to the calendar day, because two rides at 08:30 and 18:00 are one day
to the athlete and two bars otherwise.

### 16d. Files

- `ml/interval_watts.py` - `effort_best`, `_effort_by_day`, `day_series`,
  `day_options`; `BEST_W_COL` / `BEST_S_COL`; `REP_CLASSES` boundary documented.
- `views/evolution.py` - `_interval_by_day(ctx)` and the two-tab restructure.
- `tests/test_interval_watts.py` - sections 7, 8 and 9: the per-day rules on a
  7-session synthetic frame, the boundary, and eight degenerate inputs.

Verification was `tests/test_interval_watts.py` (all pass, no credentials) and
`verify_evolution.py`, which stubs streamlit, forces four picker selections
including FTP / 20-30 min, and asserts on the real figures: bars present, exact
length labelled, line points equal bar count, no bar above its own session
average, date axis, thousands separators, table row per bar. The stub needs
`st.session_state` to be a real mapping - `core/context.py` calls `.get()` on it
before any page code runs.

### 16e. Still open

- **Training page categories** still do not match the agreed 11 types.
  `views/training.py:331-400` invents its own `pattern`/`combo` taxonomy
  ("Single-dominant", "Mixed", "Top-3 types by TSS share") derived from TSS
  rather than from `MAIN_TYPES`. Not touched yet.
- **Ronnestad -> BILLAT.** Still needs the explicit decision, and the two
  requests pull against each other: relabelling the 23 shape-matched sessions
  would move ~10 FTP and ~9 AEROBIC BASE sessions out of the very FTP view this
  chart now serves. The non-destructive option - keep the athlete's label and
  add ronnestad as a derived flag - is still on the table.
- **Heat analysis** and the **weekly planner with rest weeks**: untouched.
- The separate `views/interval_watts.py` ninth page still exists alongside the
  new Evolution tab. Overlap should be resolved on the athlete's word.
- eFTP retargeting: 241 W in the 10-20 min class is still a median of detected
  efforts, not a validated threshold test. Do not call it eFTP.
