# Resume notes — updated Fri 25 Sep 2026 (session 3); original save Tue 23 Sep 2026

## Fri 25 Sep session 3 — Training types tab rebuilt: SESSION as the unit + per-interval explorer

**Server**: `python -m streamlit run app_modern.py --server.port 8601 --server.headless true`,
background shell `sh_0d898d33d001PboaJM1usYOJCQ`, ONE instance only. Earlier today two
processes bound 8601 at once and served pre-edit code — always kill every streamlit
python before relaunching. Streamlit does NOT hot-reload `views/*.py` → restart after
any view edit.

### The core decision: one row per DAY is the unit
`ml/type_comparison.py` gained `session_frame(g)` (one row per date: mean W, sets, reps,
median rest, mean IF/TSB/temp) and `all_sessions(s)` (the same for every series in ONE
groupby). **First/Last/Best W, Δ %, the Theil–Sen slope, its CI and the family
small-multiples are now all computed on session means** — the conclusions table and the
explorer chart are the same numbers by construction, so they can never disagree. `Sets`
still counts sets; a day with two sets is ONE observation, not two.
- `MIN_CI_SESSIONS = 5`: below 5 distinct days the bootstrap CI is left blank (a 3-session
  series used to print ±0.00 and look certain). The CI condition counts DISTINCT t values.
- `Ridden` now derives from the SAME median IF printed beside it (was the modal band over
  sets) → no row can say "IF 88" next to "<88 %". >50 % flagged sets ⇒ "n/a".
- Performance: `run_type_comparison` 43.6 s → **8.7 s** (77 per-series pandas groupbys
  replaced by one; the small multiples stopped re-filtering the whole frame). Dead
  `_slope_w_month` removed; `series_trend` is session-based.
- `_CODE_TAG` in `views/intervals_view.py` is part of the `st.cache_data` key of the two
  cached entry points — Streamlit hashes ARGUMENTS, not function bodies, so without it an
  edit inside `ml/` kept serving old numbers for the full 30-min TTL. Bump it when
  `ml/type_comparison.py` / `ml/protocol_reps.py` change.

### Per-interval explorer (ml/protocol_reps.py + views/intervals_view.py)
Training type → duration class → 5 KPI cards, then:
1. **every individual interval as a bar**, one slot per session, ◇ = that session's mean
   (no smoothing, no invented points);
2. **session-average line over time** with the Theil–Sen trend and its bootstrap 95 % CI
   band, optional temperature / TSB right-axis overlays;
3. "Plain reading" callout, the quality callouts, the per-session table (Open/Close W,
   fade, best/worst, trend W, °C, TSB) and per-interval / per-set expanders.
Both analysis calls are inside the Streamlit cache, so changing the type or a range no
longer re-runs the 9 s bootstrap.

### Three real bugs found by looking at the live DOM (not by tests)
1. **"138 intervals in 138 sets"** — the per-set row counted ROWS, so every set reported
   itself as N sets. One group = one set now (`n_sets: 1`).
2. **Legends clipped outside the canvas.** A horizontal Plotly legend is ONE row and does
   not wrap: measured live, the family small multiples (257 px wide) carried legends of
   336–625 px, and the session-trend legend 616 px in a 530 px figure. New `_fit_legend()`
   estimates the row (≈5.0 px/char at font 10 + 28 px per swatch, calibrated on the
   rendered DOM) and stacks it vertically when it cannot fit the narrowest column the page
   can produce (250 px small multiples, 430 px full width). Labels shortened: "Session
   avg", "Session avg (mean of its reps)", "Trend 95 % CI", "Theil–Sen trend +X W/month".
3. **"Fade +0.0 %" on single efforts** — a 1-rep set has no opening or closing window, so
   the fade was 0 by arithmetic, not physiology. Sets with <2 reps now report NaN and the
   KPI says "single efforts — one rep, nothing to fade between".
Also: the fallback path (member rows not in the cache) no longer passes the SET count off
as the INTERVAL count; `Intervals` is blank rather than invented.

### The data after the screen: 866 clean sets / 134 sessions / 35 series
Only **3 of 35** series clear the arrow bar, and that is the finding:

| series | sets/sessions | first→last | trend | verdict |
|---|---|---|---|---|
| FTP/threshold 5 min | 24 / 19 | 186 → 234 W (+25.2 %) | **+6.58 ±3.46 W/mo ↑** | the one clean improvement |
| sprints ≤15 s | 73 / 51 | 418 → 454 W | **+6.21 ±4.46 W/mo ↑** | real watts, IF unusable |
| single efforts 6 min | 48 / 36 | 212 → 191 W (−9.9 %) | **−2.38 ±1.26 W/mo ↓** | the one clean decline |
| Ronnestad 30/15 | 5 / 5 | 295 → 323 W | +14.61 ±16.71 → | direction right, evidence not there |
| VO₂ max 3 min | 32 / 27 | 195 → 226 W | +0.07 ±2.96 → | endpoints rose, trend flat |
| VO₂ max 4 min | 32 / 25 | 219 → 227 W | −0.22 ±4.65 → | flat |
| FTP 4 / 6 / 7 min | 16/12, 9/7, 5/4 | ≈flat | all → | flat |

19 of 35 have a CI at all (the rest have <5 distinct days) and say · or →. Within-set
fade (closing 2 reps vs opening reps 2–3): FTP/threshold 5 min +0.4 % median (worst
13.5 %), Ronnestad 30/15 +2.0 % (worst 4.8 %), VO₂-named 3 min +2.8 %, 4 min +2.6 %,
FTP 6 min +7.8 %.
Quality screen: 75 sets excluded (61 gaps too long = not one protocol, 14 duplicate rides =
the 4 real double-sync pairs), 83 flagged not deleted (76 IF >150 %, 7 sub-15 s).
Intensity labels: only "single efforts 1 min" is genuinely VO₂ (median IF 106 %); the
VO₂-named 3–4 min sets ride at 85–86 % IF = sub-threshold tempo work.

### Where the rider actually is (derived, all 19 series with ≥5 sessions)
Last-3-session mean vs that series' OWN season median (not vs the first day):
structured multi-rep sets **+1.8 %** (median), unstructured "single efforts" **−1.0 %**;
the two series that clear the arrow bar are also the two furthest above their own median
(FTP/threshold 5 min **+9.9 %**, sprints ≤15 s **+8.2 %**); the two at the bottom are
single efforts 1 min **−7.9 %** and 6 min **−4.8 %**. Dropping the FIRST session barely
moves any slope (e.g. 6 min −2.38 → −2.51 W/mo), so the "first→last looks down, trend
says flat" shape is a real high-start/low-finish season, NOT a first-point artefact.
Coach read: the structured block is the part that is progressing; the unstructured work
is sitting ~5–8 % below its own median at the short end.

### Verification
- Offline: `py_compile` green on views/intervals_view.py, ml/type_comparison.py,
  ml/protocol_reps.py, ml/set_evolution.py, ml/vo2_estimate.py, views/forecast.py.
  `test_session_unit.py` (new) — `session_frame` vs `all_sessions` agree on all 77
  series, bar-mean vs session-mean diff 0.0, summary vs explorer first/last/best/slope/CI
  identical. `test_figures.py` (new) — every figure builds, legend orientation asserted
  against the label widths. `test_type_comparison.py`, `test_vo2.py`,
  `test_protocol_reps.py` pass. Console needs `$env:PYTHONIOENCODING='utf-8'` (cp1252
  chokes on the arrow character).
- Live DOM on `tab_70775130-9327-4ae4-a435-7d60cda62408` (`/intervals`): 4 tabs, no
  `stException`, 22 figures; the explorer's two new figures carry exactly the traces they
  promise (138 bars + 5 session diamonds; CI band + session line + Theil–Sen line).
- **Screenshots could NOT be captured this session**: `browser.screenshot` refused with
  "needs a visible tab" and `document.visibilityState` stayed "hidden" even with the Edge
  window foregrounded and resized. Verification was DOM/geometry only (the method agreed
  earlier). Worth retrying when the desktop session paints again.

### Session 3b — two readability bugs the user found BY LOOKING (not by tests)

Both were reported from the rendered page after the first proof screenshots, and both
were invisible to the automated checks because the checks asserted the *model*, not
the paint.

**Bug 4 — the 11-duration small multiple was unreadable: "it overlaps legend and
dots".** Root cause: `legend()` in `core/components.py` parks the legend INSIDE the
plot area at `x=0.005, y=0.995`. On a two-per-row column (~257 px) with six names it
lay straight across the markers. Two fixes, both new and local to the view:
- `_lane_legend(fig, labels, min_w, size, cols)` replaces the old `_fit_legend`. A
  legend now gets a **margin lane**: one row in the TOP margin when the names fit
  (y=1.0, yanchor=bottom, `margin.t` grown), or wrapped onto rows of `cols` in the
  top margin via `entrywidthmode="fraction"` (which is the only way a horizontal
  Plotly legend can wrap), or a stack in the RIGHT margin widened to the longest
  name. MUST be called **after** `style_figure` — that call replaces the whole
  margin dict, so the lane has to be applied at the call site, not in the builder.
  `_family_figure` therefore returns `(fig, labels)`; the three call sites moved.
- The small-multiple grid is no longer a fixed 2-per-row. Families with **>2
  durations get their own full-width row** (`TYPE_FIG_H_FULL = 350`); ≤2 still
  pair up. Plot area on the 6-series families: **181 px → 382 px wide**, 186 px tall,
  legend in 2 rows of 3 in the top margin, over nothing. Two-per-row is only ~257 px
  total, and six lines plus six names cannot both be legible in 257 px.
- `cols` is capped by the longest label so no entry is ever truncated to "…":
  `cols = max(1, min(cols, int(430 / need)))` — 430 px is the narrowest full-width
  column this page produces. This caught a real truncation: "Session avg (mean of its
  reps)" needs 178 px of a 153 px third, so that name is now "Session avg (of its
  reps)".
- The side-lane branch is now only reachable from a genuinely narrow column. The
  trend figure deliberately uses `cols=3`, never a right lane: with temperature/TSB
  on, a right lane would sit on the y2 axis.

**Bug 5 — "there's an icon for the average watts of the session but can't see a
figure or average number".** The diamonds WERE drawn (14.3 px, measured in the DOM,
5 of them) but were a light fill with a blue ring among 138 thin blue bars — they read
as just another bar top, and carried no number. Now each session gets a **dotted
orange line across its own slot AT that session's mean**, a 15 px diamond with a 3 px
orange ring on it, and the wattage printed above (`markers+text`, `cliponaxis=False`).
Caption and legend both say "Session average" in words.

### Verification (session 3b)
- `py_compile` green. `test_figures.py` rewritten: for every figure on the tab it
  applies `style_figure` then `_lane_legend` and asserts the legend is in a MARGIN
  (y==1.0/yanchor==bottom, or x==1.0/xanchor==left), that the margin has room for the
  rows, that no name is truncated, and that the plot keeps ≥380 px × ≥150 px. Also
  asserts the session-average trace is `markers+text` with one "N W" label per session
  at size ≥14 with a ≥2 px ring. All 13 figures pass; exit 0.
- `test_fade_honesty.py`, `test_session_unit.py`, `test_protocol_reps.py`,
  `test_type_comparison.py` all exit 0 (no number changed — this pass is layout only).
- Server restarted (one instance, `health=ok`). **Live DOM verification of 3b could
  not be completed: the desktop browser disconnected mid-session** ("No desktop
  browser is connected to this session"). The model-level geometry was measured
  (458 px canvas → 382 px plot); the PAINT was not re-checked this pass.

### Still pending (unchanged)
- Proof screenshots of the bold Forecast-page redesign, and of this tab.
- User answers: (a) re-run pattern→FTP ML against real rolling eFTP? (b) accept the
  `.rsc.json` plan (add `"cycling-coach"` to `ownSkills`) — show the exact plan and WAIT;
  (c) the "28 weeks projection with weekly pattern of flats" horizon-selectbox question.
- All work UNCOMMITTED.

---

## Fri 25 Sep session 4 — display rules: whole minutes, big numbers separated

The user approved the tab work and asked for exactly two display changes:
1. *"not a fan of the grouping in identical sets over time, just round it to one digit
   no decimals in time, just minuts without decimals"*;
2. *"things like elevation or tss i would use , to separate numbers ie 135,000 not
   135000"*; then *"ready to upload to github and create a live app in streamlit"*.

### Rule 1 — whole minutes, one lever (`ml/type_comparison.py: fmt_min`)
`fmt_min(secs, decimals=0)` is now the single formatting lever: under 90 s keeps seconds
(a 30 s effort is not "0 min"), everything longer is whole minutes by the **same
halves-DOWN class rule `dur_bucket` uses** (`np.ceil(s/60 - 0.5)`). That matters: a
printed duration and the group it belongs to can never disagree (2:59 → "3 min" and it
lands in "3 min"; 3:30 → "3 min" and it also lands in "3 min"). `decimals=1` is kept for
an audit table that wants the exact form. New sibling `fmt_secs()` = exact whole seconds.
`ml/set_evolution.py:_dur_label` uses the same rule.
- **The trap found before coding**: the identical-sets groups are **10-SECOND buckets**,
  so a whole-minute label alone is NOT unique — 190 groups collapse to 98 whole-minute
  prefixes, and two dropdown entries would have read identically (`1 × 3 min · 19
  sessions` and `1 × 3 min · 21 sessions`). Fix: `_sig_label()` = `2 × 4 min · 240 s`, the
  clean minute plus the exact rep length in seconds. The **grouping did NOT change** — no
  number on that tab moved, no cluster merged, and a test asserts no label hides a >60 s
  spread. Live dropdown now reads `2 × 4 min · 240 s · 7 sessions · 10 sets · VO₂ max sets`.
- **Nothing is lost by the rounding**: every table that enumerates measured lengths gained
  an exact-seconds column (`Length s` / `Rep s` / `Dur s` / `Duration s`), the "Intervals
  measured …" line is now `3–4 min (175–235 s, median 205 s)` via `_span_txt()`, and the
  two KPI foots print `3 × 5 min (302 s)`. The 60–89 s band deliberately stays in seconds
  (75 s is not "1 min").
- Window echoes use `:g`, not `.0f`/`.1f`: `3–6 min · 90–120% FTP`, and a genuine 3.5
  boundary still prints `3.5`. The only decimals left on the page are the two strings that
  *explain* the rule ("up to 3.5 min → 3 min").

### Rule 2 — thousands separators
The 135,000 was `views/trends.py` year cards (`fmt = f"{{:.{dec}f}}{unit}"` with dec=0).
Now `dec == 0` → `{:,.0f}` for value, delta and foot. Live Trends page: **zero**
unseparated 5+ digit numbers; values read 235,904 / 138,176 / 97,728 / 1,372,325.
Also `views/overview.py` Weekly TSS (×2) and `views/sessions.py` `Elev (m)` →
`NumberColumn(format="localized")` (Streamlit 1.58 documents it; unverifiable from the
DOM because the dataframe grid paints to a virtualised surface, not text nodes).

### Verification
- New `test_display_rules.py` (exit 0): 14 `fmt_min` cases, agreement with
  `dur_bucket`/`dur_class_label` over 200+ values, 190/190 unique dropdown labels, no
  group hiding a >60 s spread, and the trends formatter incl. the 135,000 case.
- `test_figures`, `test_fade_honesty`, `test_session_unit`, `test_protocol_reps`,
  `test_type_comparison`, `test_set_evolution`, `test_vo2` all exit 0.
- Live DOM: 0 exceptions on /intervals, /trends, /sessions; band label reads
  `Best 3–6 min effort`; only the 2 rule-explaining decimals remain.
- `_CODE_TAG` bumped to `whole-min-2026-09-25c` (ML code changed → cache must not serve
  pre-edit rows for 30 min). Server restarted, one instance, `health=ok`.
- `browser.screenshot` still fails ("needs a visible tab") — the desktop window is not
  paintable from here, so this pass is DOM-verified, not screenshot-verified.

### Deployment facts gathered (for the GitHub / Streamlit Cloud step — NOT done yet)
- Secrets: nothing hardcoded anywhere; `probe*.py` read `os.environ`.
  `core/data.py:api_creds()` already tries `st.secrets` **first**, then `.env` → the
  Cloud dashboard secrets path works as-is. `.env` is gitignored.
- `requirements.txt` and `.streamlit/config.toml` (dark theme) already exist and cover
  the app. Entry point for Cloud = `app_modern.py` (set in the dashboard; `app.py` is the
  original monolith and must not be modified).
- **Publishing blocker**: `data/combined_training_data.csv` (1.3 MB of real ride history)
  and `data/power_curve.csv` are **already tracked**, so a push publishes the rider's
  training data. `.gitignore` lists them but ignore does not untrack. Also untracked and
  not ignored: `.claude/`, `.rsc.json` (42 KB), `01-TOOLS/_TEMPLATE`, `02-DOCS/wiki`,
  `probe*.py` — harness clutter, not app code.
- Without a data file the non-interval pages have nothing to show, so a live app needs a
  decision: ship the real data, ship an anonymised demo dataset, or accept empty pages.

---

## Fri 25 Sep session 2 — Intervals page: identical-set comparison + duration evolution + VO₂max
Server relaunched this session: background shell `sh_0d7f90552001NbpRFKRTJqpPP1`, port 8601,
headless — carries ALL of the below. (Earlier `sh_0d7e34d8c001uO1AEcIMJ8QU73` was killed for
the restart.)

### New/changed files (all mine — `app.py`, `src/`, `main.py`, `README.md` untouched)
1. **core/interval_data.py — schema v2**: `IV_COLUMNS + group_id` (intervals.icu repeat-group
   key, e.g. `29s@323w94rpm`), `ACT_COLUMNS + temp` (`average_temp`, °C, 240/259 coverage),
   new `fetch_profile()` (DOB 1977-10-13, sex M, icu_weight 56, height 1.75 — 24 h cache).
   Stale v1 caches (no group_id) auto-discard → one-time full re-pull DONE: 3,674 rows,
   195 activities, group_id on 100 % of WORK rows.
2. **NEW ml/set_evolution.py**:
   - `build_sets`: group_id → SETS; measured rest = MEDIAN per-gap recovery where
     prev AND next WORK row share the same key (the naive first/last-seq version summed
     warm-up gaps → reported 94 s rest for a Ronnestad that is really ~16 s — fixed).
   - `add_signatures`: sig = rep bucket (5 s ≤90 s else10 s) × reps (exact ≤10 else /5);
     style heuristics: Ronnestad (30/15), Billat (30/30), FTP/threshold, VO₂, sprints,
     long efforts. Multi-rep clusters sorted BEFORE singles (workouts first).
   - `run_set_evolution`: OLS W/month + Pearson/Spearman, detrended context associations
     (temp, TSB, CTL, acwr, prior-7d TSS, set HR, decoupling) only when n ≥ 8; fresh-vs-
     fatigued TSB split needs ≥ 4 sets/side else honest "not enough" callout.
   - `run_duration_evolution`: monthly BEST watts per duration class, row-normalised heatmap
     (dark=class low, green=class high, blanks stay blank) + W/month slope table.
3. **NEW ml/vo2_estimate.py**: VO₂ = 60·P ÷ (GE·20.1 kJ/L·1000). User answered: NO lab test
   → GE 21 % centre, band 18–24 % shown as sensitivity; maximality = "you review the data".
   PVO2 = best 180–360 s effort → **271 W · 4.75 W/kg (301 s, 2026-05-27)** → 3.85 L/min =
   **68 mL/kg/min @57 kg** (79 @18 % / 59 @24 %). Confidence judged from the best
   HR-recorded window effort → 166 bpm = 98 % of all-time max 170 → **Strong**. Evidence
   table lists top-10 window efforts with "% of all-time HR". Age-predicted HR (Tanaka
   208−0.7·48) = 174. P3/P4/P5/P6 disagreement (3.60–3.85 L/min) surfaced as uncertainty.
   Anchor ladder (top-down): 70 elite / 60 high-well-trained / 50 trained / 40 avg-fit.
   User's own guess ("3 min ≈ 260–275 W") vs data: **268 W best 3-min** ✓.
4. **views/intervals_view.py**: Section 1 → 3 tabs (🧩 Identical sets over time = default,
   📈 Evolution by duration class, 🔍 Duration-window search = old code unchanged);
   NEW Section 5 "🫁 VO₂max estimate from interval power" (power–duration curve, log x with
   bucket labels, implied-L/min right axis, PVO₂ hline pill, sensitivity + disagreement
   tables, evidence table, formula 🧮 + caveats ⚠️ callouts). Session keys: `iv_sig`,
   `iv_on_temp`, `iv_on_tsb`, `vo2_weight` (default 57), `vo2_ge` (21.0). Page subtitle now
   mentions VO₂max.

### Verification
- Offline: `test_set_evolution.py` + `test_vo2.py` (in `C:\Users\levod\AppData\Local\Temp\opencode\`)
  both pass — Ronnestad-style `30s × 30` cluster (4 sessions, +13.5 W/mo), Billat `30s × 20`,
  FTP ×24; duration trends: 30 s +8.95, 1 m +6.75, 5 m +3.55, 8 m +3.73 W/month (↑ across
  the board, 2 m flat); view + modules import clean; py_compile green.
- **Live DOM pass: DONE (Fri 25 Sep)** on `tab_13b46bf0-afc4-45de-bf11-76fa1cb22756` (`/intervals`):
  no `stException`; 3 tab labels present (🧩 Identical sets / 📈 Evolution by duration / 🔍 Duration-window search);
  VO₂ section fully rendered (weight widget, GE slider, 4 KPI cards, curve, 3 tables, 3 callouts);
  sets tab = 11 figures, duration tab = 7 figures (heatmap + strain fan + 2 band + 3 composition), 10 dataframes.
  **Screenshots captured**: sets KPI+chart+plain-reading, VO₂ KPI/profile, VO₂ curve-top (wide),
  VO₂ sensitivity+disagreement+evidence+confidence+formula+caveats, duration heatmap, slope table
  (files under `C:\Users\levod\AppData\Local\Temp\opencode-browser-*\0\screenshot.png`).
  Gotchas learned: `stMain` (not window) is the scroll container; plotly titles are SVG text (search
  `f.textContent`); section headings are CSS-uppercased (search `INTERVAL POWER`); screenshot frames can
  lag/deliver stale surfaces when the Edge window isn't actively painting — verify via DOM geometry
  (`getBoundingClientRect`) instead; sidebar re-expanded itself mid-session (harmless, chart self-heals
  to container width), collapse toggle via `el.click()` was unreliable.
- Left the page on the 🧩 tab (default) after verification.

### Still pending (unchanged from session 1)
- Proof screenshots of the bold Forecast-page redesign (both charts DOM-verified earlier).
- User answers: (a) re-run pattern→FTP ML vs real rolling eFTP? (b) accept `.rsc.json` plan
  (add `"cycling-coach"` to `ownSkills`) — WAIT for acceptance; (c) "28 weeks projection with
  weekly pattern of flats" horizon-selectbox question.
- All work UNCOMMITTED; git shows only pre-existing `.gitignore` + `data/combined_training_data.csv`.

---

# Original notes — end of session Tue 23 Sep 2026

## Server
- Run: `python -m streamlit run app_modern.py --server.port 8601 --server.headless true`
  (from this project dir). Background shell `sh_0cf9e6527001WDYb6ZsC0c54QV`, PID 12996.
- Health at save time: `http://localhost:8601/_stcore/health` → `ok`.
- Background shells have died once before tonight — if it's down tomorrow, just relaunch
  (first healthy response can take 30–110 s; script run adds ~30–60 s before charts appear).
- **Streamlit does NOT hot-reload `views/*.py` edits → restart the server after any view change.**

## Browser state
- Forecast tab: `tab_13b46bf0-afc4-45de-bf11-76fa1cb22756` — loaded clean, no exceptions.
- Verified live in DOM: PMC 9 traces ↔ 9 legend entries; eFTP 9 traces ↔ 7 legend entries
  (legend-honesty invariant holds).
- Screenshot tool fails if the Edge window is not visible/focused → bring it to the front first
  (`browser.tabs.focus` + desktop window foreground).

## Tonight's completed work (live in `views/forecast.py`, both files parse)
Bold redesign of the two Forecast charts:
- **PMC (`_render_pmc`)**
  - Seam: solid, width 3, `#d29922`, both panels.
  - Projection region: 2 rects `rgba(88,166,255,0.20)` + dotted cyan border 1.5.
  - Pills (bordered labels): `today` 13 px moved to BOTTOM of its line (it used to collide with
    the legend → garbled text), value pills `132 (+2)` / `133 (+2)` / `-2 (+0)` at 15 px.
  - `PROJECTED` watermarks ×2, `textangle=-90`, sizes 30 / 22.
  - Recent-pace corridors width 9, opacity 0.3; plan dashes width 4.5.
  - Right margin widened to `r=88` so pills don't clip.
- **eFTP (`_render_eftp`)**
  - Bands fill alpha 0.18 / 0.36 + 1 px edge strokes (lo90/lo50 are `showlegend=False` helpers).
  - Forecast line width 4.5; star marker `mode=markers` size 20; `230 W` pill 15 px.
  - `today` vline solid width 2.5, label pinned to BOTTOM of the line.
- All of the above were DOM-verified on the live page (annotations/shapes/traces present).

## KEY DIAGNOSIS from tonight (why the screenshots looked squashed/broken)
Measured in the live tab:
- Viewport is only **800 px** wide and the **Streamlit sidebar is EXPANDED**:
  sidebar `left=0, width=300` → `stMain left=300 width=500` → **figures render at 458 px**
  (fig0 `left=316, right=774`).
- The charts are NOT broken — the pane is just narrow.
- Fix before taking proof screenshots: click `[data-testid="stSidebarCollapseButton"]`
  (present in DOM) and/or widen the desktop window, then screenshot fig0 (PMC) and fig1 (eFTP).

## Tomorrow — task list
1. Collapse sidebar + foreground window → take fresh PMC and eFTP screenshots as proof.
2. User's last remark: *"you have my data and have invented the projection of 28 weeks with a
   weekly pattern of flats"* — clarify what he saw:
   - PMC horizon selectbox: `["21 days", "28 days", "42 days"]` (CHECK the `index=` default!),
     PMC x-axis ran to **2026-12-08** — verify horizon semantics (calendar days vs ride days).
   - eFTP horizon selectbox: `["4 weeks", "8 weeks", "12 weeks"]`, x-axis ran to **2027-01-05**
     — check which option was active.
   - "weekly pattern of flats" is most likely the **prescribed plan** (weekly TSS target spread
     with rest days = 0) — by design, not invented. Explain / offer to make it clearer in the UI.
3. Still awaiting user answers:
   - (a) Re-run pattern→FTP ML against real rolling eFTP? (yes/no)
   - (b) `.rsc.json` → add `"cycling-coach"` to `ownSkills`: show exact plan, WAIT for
        acceptance before writing.
4. Hard constraints (do not violate):
   - Never modify originals: `app.py`, `src/`, `main.py`, `README.md` — new files only.
   - Do NOT touch `show()` / `style_figure` globally (user approved the app-wide look).
   - ML honesty rules (time-ordered validation, persistence baseline, surfaced coefficients).
   - Visual scope = Forecast page charts only.
   - rsc changes: plan first, acceptance before write.

## Quick reference
- Health: `http://localhost:8601/_stcore/health`
- Forecast tab id: `tab_13b46bf0-afc4-45de-bf11-76fa1cb22756`
- Main files: `views/forecast.py` (redesign target, ~550 lines),
  `core/components.py` (`show()` L381, `legend()` L392), `core/theme.py` (`style_figure` L82,
  `H_HERO=780`, `H_STD=420`), `ml/pmc_projection.py`, `ml/ftp_forecast.py` (`run_forecast`),
  `app_modern.py` (6-page router), `src/intervals_api.py` (fetch).
- Plotly 6.8.0: `add_vrect` line_dash/line_color, `add_vline` line_width/annotation_position,
  `textangle=-90` — all validated working (all pass through `**kwargs`).
- Work is UNCOMMITTED in git; originals untouched (git shows only the pre-existing
  `.gitignore` + `data/combined_training_data.csv` modifications).
