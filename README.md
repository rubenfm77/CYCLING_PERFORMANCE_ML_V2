# 🚴 Cycling Performance ML

> **6+ years · 1,161 activities · 1,085 ride days · 2,941 hours · real athlete data**
> Road cycling with power meter · Catalonia, Spain · Post-surgery rebuild (Jun 2025)

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?logo=streamlit)](https://cycling-performance-ml-gwl7kzbkctmdgnatg2jvnt.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌐 Live Dashboard

**[Open Dashboard →](https://cycling-performance-ml-gwl7kzbkctmdgnatg2jvnt.streamlit.app/)**

Real-time cycling performance dashboard built with Streamlit. Updates automatically via Intervals.icu API pipeline.

**Run locally:**

```bash
streamlit run app_modern.py     # the 6-page dashboard
```

> **On Streamlit Cloud:** set **Main file path** to `app_modern.py` in the app's
> settings, and add `INTERVALS_ATHLETE_ID` and `INTERVALS_API_KEY` under
> **Settings → Secrets**. `core/data.py` reads `st.secrets` first and falls back to a
> local `.env`, so the same code runs in both places. Then hit **Rerun** — the app picks
> up the new commit on its own.

---

## 📁 Project Structure

```
cycling-performance-ml/
├── data/                          # Ride data (see ⚠️ Data Privacy below)
│   ├── JOIN_STRAVA_TP.xlsx        # Historical joined dataset
│   ├── combined_training_data.csv # Merged API + historical
│   ├── interval_cache.csv         # Cached intervals.icu interval detail
│   └── wellness_data.csv          # Intervals.icu wellness
├── src/                           # Original pipeline (unchanged)
│   ├── config.py                          # Constants, zones, colours, athlete profile
│   ├── data_loader.py                     # Load + clean + feature engineering
│   ├── pmc.py                             # Performance Management Chart (CTL/ATL/TSB)
│   ├── ftp_analysis.py                    # Random Forest — what drives FTP gains (session level)
│   ├── monthly_composition_analysis.py    # Composition mix → FTP outcomes (window level)
│   ├── wkg_progression.py                 # W/kg regression + 12-week forecast
│   ├── clustering.py                      # K-Means + PCA — session archetypes
│   ├── fatigue_detection.py               # Isolation Forest — overreach detection
│   └── intervals_api.py                   # Live Intervals.icu API pipeline
├── core/                          # Shared foundation
│   ├── data.py                    # CSV + API load, st.secrets/.env credentials
│   ├── interval_data.py           # Interval-level fetch + cache (v2: group_id, temp)
│   ├── theme.py                   # GitHub-dark palette, chart heights, style_figure
│   ├── components.py              # metric_card, callout, dataframe, show, legend
│   └── pmc.py                     # CTL/ATL/TSB engine
├── ml/                            # The analysis models
│   ├── set_evolution.py           # Set detection, signatures, quality gates
│   ├── type_comparison.py         # Training-type × duration series, Theil–Sen fits
│   ├── protocol_reps.py           # Per-interval bars, session means, trends
│   ├── interval_forecast.py       # Walk-forward best-effort forecast
│   ├── exertion_forecast.py       # RPE / TSS → performance model
│   ├── composition_intervals.py   # Interval mix → performance model
│   ├── vo2_estimate.py            # ACSM VO2max from CP and weight
│   ├── pmc_projection.py          # Forward PMC under a planned week
│   └── ftp_forecast.py            # eFTP forecast
├── views/                         # Streamlit pages (one module per page)
│   ├── overview.py  sessions.py  fitness.py  training.py
│   ├── trends.py    intervals.py  forecast.py
├── app.py                         # Original single-file dashboard (unchanged)
├── app_modern.py                  # 6-page dashboard — the one to run
├── main.py                        # Run full ML pipeline
├── METRICS_GLOSSARY.md            # Every metric explained in plain language
├── requirements.txt
└── README.md
```

---

## 📊 Dashboard Features

### Today's Status
- CTL / ATL / TSB with 7-day deltas
- eFTP from Intervals.icu
- W/kg 10-session rolling average
- Colour-coded fatigue state with automatic recommendation

### 🩺 VO2max Estimate *(new)*
- Estimated via the ACSM leg-cycling equation — VO2max (ml/kg/min) = 10.8 × (Watts/kg) + 7
- Uses each session's Critical Power (`icu_pm_cp`) and logged body weight
- CP reflects a sustainable effort, not the shorter maximal ramp-test power the formula was originally built around — likely undershoots true VO2max somewhat
- Only populated where Intervals.icu has fitted a power-curve model to recent activities
- Power-based estimate, not a lab VO2max test — track it as a directional trend, not an absolute number

### 🔬 Objective Fatigue Signals

*Why objective metrics beat RPE for this athlete: heat, sleep, and motivation contaminate perceived effort. W/BPM doesn't lie.*

- **28-day rolling W/BPM baseline** — your personal efficiency benchmark
- **Session dots coloured vs baseline** — green = adapting, red = fatigued
- **Hot session detection** (>28°C) — orange marker so heat isn't misread as fatigue
- **Automatic fatigue alert** — if W/BPM drops >5% below baseline, dashboard flags it regardless of TSB
- **Fatigue overrides recommendation** — physiological signal beats TSB number

### 🇳🇴 Norwegian Method Compliance
- Weekly true threshold sessions (IF ≥ 0.85) vs Z3 drift (IF 0.75–0.85)
- IF distribution histogram — validates training polarisation
- Target: 2 threshold sessions/week + pure Z2 everything else

### 📈 Performance Management Chart
- CTL / ATL with correct EWM formula (42-day / 7-day)
- TSB coloured by fatigue state
- Weekly TSS bars with overreach reference lines

### 🎯 FTP Development Analysis
- FTP Stimulus Score by training type (IF² × Duration × 100)
- TSS split: quality vs base volume
- Weekly FTP stimulus trend with 4-week rolling average

### 🔀 Training Composition Analysis *(new)*
- Stacked bar: % TSS by training type per calendar month — see how the mix has shifted over 6 years
- FTP proxy trend aligned beneath so you can visually correlate composition shifts with fitness peaks
- Pattern comparison table: Single-dominant vs Mixed months and their next-month FTP outcome
- Top combination ranking: best-performing 3-type combos by average next-month FTP gain
  (sample-size caveats shown inline — combos with n < 3 flagged)

### ⚡ Power Curve
- Best efforts at 5s / 1min / 5min / 10min / 20min / 30min / 60min
- W/kg at each duration with target reference lines
- Identifies the gap between neuromuscular ceiling and aerobic threshold

### 📈 FTP Progression
- Monthly estimated FTP (best NP × 0.95) from 2019 to present
- Surgery structural break annotated
- Annual peak FTP comparison

### 🏔️ Elevation Stats
- Weekly elevation with 3000m/week target line
- Annual elevation totals
- Career total and best single session

### 📊 Year vs Year Comparison
- This year vs same period last year: TSS, sessions, power, W/kg, elevation, hours
- Monthly TSS comparison chart
- Monthly W/kg comparison chart

---

## 🧩 Intervals Page — the interval-level analysis

The deepest page in the dashboard, built on intervals.icu's **interval-level** API rather
than session summaries. Four tabs, all over the full cached history so trends have room:

### 🧩 Identical sets over time
- Reps sharing a `group_id` form one **set**; sets are matched across rides by **rep
  length (within 10 s) × rep count** — so `2 × 4 min · 240 s` is one repeated protocol
- Rest between reps is measured from the recovery rows, which is what separates a
  **Ronnestad** (30 s on / 15 s off) from a **Billat** (30 s on / 30 s off)
- **One matching protocol selected at a time**, with every rep plotted, plus a robust
  trend and a ±28-day comparison
- Family names are heuristics derived from the measured numbers, and are always printed
  next to those raw numbers

### 🏷️ Training types over time
- **One row = one training type at one duration class**, with its own sessions, its own
  watts and its own trend — a heatmap of best watts per month plus a sortable summary
- **Never compared across types or across duration classes**: a 3-minute sprint and a
  3-minute tempo ride are different series by construction
- Select any series for the **per-interval explorer**: every single rep as a bar, the
  session-average as a dotted line with a numbered diamond, and a Theil–Sen trend with
  a bootstrap CI band

### 📈 Evolution by duration class
- Monthly **best** watts per duration class as a heatmap, each row coloured by its own
  worst→best range, actual watts printed in every cell

### 🔍 Duration-window search
- Exact duration × intensity window (the same match intervals.icu's own interval search
  uses), computed locally so **every matched effort** is visible, not just a count
- Feeds the walk-forward forecast in the section below

### 📐 The analysis rules this page holds itself to
| Rule | Why it exists |
|---|---|
| **Session is the unit** of every time-based number | One ride must never outvote another; table and chart are the same aggregation |
| **Duration = one rep**, never total workout time | The classic way interval analysis lies |
| **Whole minutes, halves down** — `3:30` is "3 min" | One function prints the label and makes the group, so they cannot disagree |
| **Exact seconds beside every rounded minute** | A rounded "3 min" never hides that the reps ran 2:50–3:55 |
| **Theil–Sen + bootstrap CI**, not least squares | One outlier rep can't drag the trend |
| **↑ / ↓ only if the slope clears its own CI** and is ≥ 1 W/month; otherwise `→` or `·` | A trend that can't clear its error bar isn't a trend |
| **Fade is `n/a`** for a single rep | Inventing "+0.0 %" would be a fabricated number |
| **"Ridden" band** printed next to the family name | The name describes the protocol shape; the band describes how hard it was actually ridden |
| **Excluded vs flagged** kept distinct | Deleting a set over one soft column would delete the sessions that matter |
| **n always shown** | No number without its sample size |

---

## 🔬 ML Analyses

| Module | Method | Question answered |
|---|---|---|
| `ftp_analysis.py` | Random Forest + feature importance | Which training types most drive FTP? *(session level)* |
| `monthly_composition_analysis.py` | Correlation + RF (exploratory) | Does the *mix* of types per month outperform any single dominant type? |
| `wkg_progression.py` | Linear regression + Ridge | Where is W/kg heading? |
| `clustering.py` | K-Means + PCA | Hidden session archetypes? |
| `fatigue_detection.py` | Isolation Forest | When is fatigue genuine overreach? |
| `ml/type_comparison.py` | Theil–Sen + bootstrap CI | Are my watts rising on *this* work, *at this* duration? |
| `ml/interval_forecast.py` | Walk-forward vs persistence | What can honestly be said about this window in 28 days? |
| `ml/exertion_forecast.py` | Time-ordered backtest + ridge | Does RPE / TSS add anything over persistence? |
| `ml/pmc_projection.py` | Exact EWMA recursion | Where does fitness go under a *planned* week? |

> **Small-sample caveat on composition analysis:** ~72 independent calendar-month
> windows over 6 years.  Any finding with |r| < 0.3 or p > 0.10 should be treated
> as noise at this sample size.  The RF is exploratory — it surfaces which
> composition features *might* matter, not which ones *do* matter.

### 🚦 The honesty rules every model in `ml/` holds itself to

These are not decoration — each one has a test that fails if the code stops obeying it
(`test_session_unit.py`, `test_fade_honesty.py`, `test_figures.py`, `test_display_rules.py`).

| Rule | What it prevents |
|---|---|
| **Time-ordered walk-forward validation**, embargoed by target date | A model learning from its own future |
| **Persistence ("do nothing") is the baseline every model must beat** | Shipping a model that has learned nothing about you |
| **"No model beat the flat line" is a real, reachable answer** | Manufacturing a forecast out of noise |
| **Negative skill is displayed, not hidden** | Cherry-picking only the models that looked good |
| **Coefficients are surfaced, not summarised** | Letting a model claim influence it can't justify |
| **Descriptive / associational only** | Causal language the data cannot support |
| **n is printed with every number** | A trend from 4 sessions reading like a law |
| **"Showing nothing beats showing noise"** | A chart that draws a line where there is none |

---

## 🚀 Quick Start

```bash
git clone https://github.com/rubenfm77/cycling-performance-ml.git
cd cycling-performance-ml
pip install -r requirements.txt

# Add your data
cp /path/to/JOIN_STRAVA_TP.xlsx data/

# Set up Intervals.icu API credentials (never committed — .env is gitignored)
echo "INTERVALS_ATHLETE_ID=your_id" > .env
echo "INTERVALS_API_KEY=API_KEY:your_key" >> .env

# Fetch latest data
python src/intervals_api.py

# Run the 6-page dashboard
streamlit run app_modern.py

# Run the original single-file dashboard
streamlit run app.py

# Run ML pipeline
python main.py
```

> 🔒 **Credentials never leave your machine.** `.env` is in `.gitignore`, is untracked,
> and no file in this repository contains your athlete ID or API key. On Streamlit Cloud
> they go in **Settings → Secrets** instead, which `core/data.py` reads in preference
> to `.env`.

---

## 📋 Weekly Workflow

Every Sunday after your long ride:
```bash
python src/intervals_api.py      # fetch fresh data
streamlit run app_modern.py      # open dashboard
```

Check:
1. TSB — am I fresh or fatigued?
2. W/BPM vs baseline — objective fatigue signal
3. Norwegian compliance — 2 quality sessions done?
4. FTP stimulus — is training mix driving FTP up?

---

## 🏆 Key ML Findings & Conclusions

Six years and four ML models converge on a clear picture of what actually drives cycling performance — and where most amateur athletes go wrong.

**FTP is built through volume, not isolated intensity**
- **FTP stimulus** best predicted by `IF² × duration` — not intensity alone
- **Duration beats intensity**: RF assigns 49% importance to TSS, only 1.4% to IF alone
- Sessions that combine quality with duration consistently precede FTP breakthroughs

**The biggest training mistake at amateur level is Z3 drift**
- **Z3 drift** (IF 0.75–0.85) is the most common pattern and the least productive — high fatigue, low adaptation
- Most "easy" rides that drift into Z3 would produce better results ridden 10–15 W lower in true Z2

**Session archetypes and adaptation**
- **PIRAMIDAL + FTP + SST** cluster together as the highest-adaptation archetype
- K-Means identifies 4 distinct session types; polarised training consistently maps to the best-gains cluster

**Fatigue detection**
- **W/BPM** is the most reliable fatigue signal — outperforms TSB alone
- **Isolation Forest** correctly flags all overreach periods and the post-surgery detraining arc
- When W/BPM drops >5% below the 28-day baseline, performance is degraded regardless of what TSB says

**Post-surgery recovery**
- The Ridge regression model projects W/kg recovery to pre-surgery levels within 6 months — conditional on Norwegian-compliant training load
- Annual peak FTP comparison confirms a consistent winter base-building pattern: CTL peaks in March, race-form peaks May–June

### 🔀 Training Composition Analysis (monthly-window results)

> ⚠️ **Small-sample caveat:** 74 calendar-month windows over 6 years; 73 with
> a valid next-month outcome.  All correlations are non-significant (ns) at
> this sample size — findings are directional signals, not statistical proof.
> Reproduce with `python main.py --module composition`.

**What it measures:** FTP proxy = `max(NP) × 0.95` per month.  Outcome =
next month's proxy minus this month's.  Training pattern = dominant type(s)
among quality (FTP-driver) TSS.

**Correlation result:** No composition feature reaches statistical significance
at n=73.  Strongest signal: `pct_SST` r=+0.222 (ns), `pct_VO2MAX` r=−0.220 (ns).
The RF regressor produced R²=−0.094 (worse than predicting the mean) — there
is not enough data for a reliable ML model at the monthly window level.

**Mixed vs single-dominant windows — actual numbers:**

| Pattern | n months | Avg next-month FTP Δ (W) | Median |
|---|---|---|---|
| Mixed: PIRAMIDAL+SST | 2 | **+11.9** | +11.9 |
| Mixed: BILLAT+FTP | 3 ✓ | **+6.7** | +5.7 |
| Mixed: FTP+TEMPO | 1 | +4.8 | — |
| No quality sessions | 11 | +2.3 | +3.8 |
| Single: SST | 10 | +2.2 | −1.0 |
| Single: FTP | 15 | +1.6 | +0.9 |
| Single: BILLAT | 2 | 0.0 | — |
| Single: TEMPO | 10 | −0.9 | −1.4 |
| Single: Q-I INTERVALS | 7 | −4.1 | −4.8 |
| Single: VO2MAX | 7 | **−9.0** | −3.8 |
| Single: PIRAMIDAL | 2 | **−10.5** | −10.5 |

✓ = n≥3 (minimum threshold for directional reliability)

**Best combination with n≥3:** AEROBIC BASE+END+FATMAX → avg +8.9W (n=3 ✓).
All top-ranked combos (END+PIRAMIDAL+TORQUE +27.5W, END+FTP+SST +15.7W) have n=1–2 — unreliable.

**Honest interpretation:**
- The data loosely supports diversifying quality work across types rather than hammering a single high-intensity mode in isolation.
- Pure VO2MAX blocks and solo PIRAMIDAL months associate with the worst next-month outcomes (−9W and −10.5W respectively) — possibly because they accumulate fatigue without the base volume needed to convert it.
- Single: FTP (largest sample, n=15) shows only +1.6W — mixing FTP with base and support types outperforms in every n≥3 pattern.
- These findings **contradict the per-session ranking** from `ftp_analysis.py` (which ranks VO2MAX and PIRAMIDAL highly for stimulus score) — the composition view suggests those types work best as part of a mixed block, not as the sole monthly focus.
- The stacked-bar chart in the Streamlit dashboard lets you visually match composition periods to your 275W peak years.

---

## 🧩 What the interval-level data actually shows

866 screened sets · 134 sessions · 35 training-type × duration series. Of the 35 series,
19 have enough sessions for a confidence interval, and **only 3 clear their own error
bar**. Those three are the only trends in this dataset I'd defend to a coach:

| Series | Sets / sessions | First → last | Robust trend | Verdict |
|---|---|---|---|---|
| **FTP / threshold 5 min** | 24 / 19 | 186 → **234 W** (+25.2 %) | **+6.58 ±3.46 W/mo** | ↑ **real** — the clearest gain in the data, and fade stayed flat (+0.4 %) |
| **Sprints ≤ 15 s** | 73 / 51 | 418 → **454 W** | **+6.21 ±4.46 W/mo** | ↑ **real** — biggest sample in the dataset, and it holds |
| **Single efforts 6 min** | 48 / 36 | 212 → **191 W** (−9.9 %) | **−2.38 ±1.26 W/mo** | ↓ **real** — and the interesting part: **fade went +7.8 %**, i.e. you got *stronger* through the set |
| Ronnestad 30 s on / 15 s off | 5 / 5 | 295 → 323 W | +14.61 ±16.71 | → not established (CI spans zero) — best single rep 354 W |
| VO₂ max 3 min | 32 / 27 | 195 → 226 W | +0.07 ±2.96 | → not established |
| VO₂ max 4 min | 32 / 25 | 219 → 227 W | −0.22 ±4.65 | → not established |

**How to read the three real ones together:** the *sustained* work got stronger
(threshold +25 %, no fade) and the *sprint* work got stronger, while the *middle*
6-minute effort got weaker but with **positive fade** — you're losing the first rep and
holding the rest, which reads as pacing and freshness rather than a fitness loss. The
VO₂-named 3 and 4 minute sets sit at **85–86 % median IF** — sub-threshold in reality,
despite the name; the only genuinely VO₂ work in the data is the 1-minute single efforts
(median IF 106 %).

**Data quality, stated plainly:** 75 sets excluded (61 multi-rep sets with minutes-to-hours
gaps, 14 duplicate rides from double-syncing — the 4 real double-sync pairs) and 83 flagged
but kept (76 with IF > 150 %, 7 sub-15 s reps). Excluded means "cannot be aggregated
validly"; flagged means "one column is soft, the rest is usable".

> These are **descriptive trends in observed history** — not a forecast, and not a claim
> about *why* anything changed. Sample sizes are printed next to every number in the app.

---

## ⚙️ Data Sources

- **Strava** → distance, elevation, speed, HR
- **TrainingPeaks** → TSS, IF, NP, power zones
- **Intervals.icu API** → eFTP, CTL, ATL, efficiency, wellness, power-curve model (`icu_pm_cp`, used for VO2max estimate), and the **interval-level** data behind the Intervals page (each rep's length, average/NP watts, HR, IF, load, temperature, and the `group_id` that groups reps into sets)
- **Wahoo ELEMNT** → raw power, L/R balance, temperature, cadence

---

## 📖 Metrics Reference

See **[METRICS_GLOSSARY.md](METRICS_GLOSSARY.md)** for a plain-language explanation of every metric in the dashboard — what it is, why it matters, and how to monitor it.

---

## ⚠️ Data Privacy

Being precise, because this is a public repository:

- **Never committed:** `.env` — your `INTERVALS_ATHLETE_ID` and `INTERVALS_API_KEY`.
  It is listed in `.gitignore`, is untracked, and no file in this repository contains
  either value. On Streamlit Cloud the same two values live in the app's encrypted
  secrets store instead.
- **Excluded by `.gitignore`:** `data/*.xlsx`, `data/wellness_data.csv`,
  `data/interval_cache.csv`, `outputs/`, `__pycache__/`.
- **Currently public, and it should not be:** `data/combined_training_data.csv` — the
  merged ride history, ~1,100 activities from 2019 onward, including activity names,
  dates, elevation, bike, device and power-meter serials, heart-rate and power zones.
  It was committed before those exclusions were added, and a `.gitignore` entry does
  **not** remove an already-tracked file.

> 🛑 **To remove it:** the file has to be untracked *and* purged from history
> (`git rm --cached data/combined_training_data.csv data/power_curve.csv`, then a
> history rewrite and a force-push), after which a deployment would need a dataset
> supplied another way. Say the word and it is a ten-minute job.

---

## 📋 Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
openpyxl>=3.1
streamlit>=1.28
plotly>=5.17
requests>=2.31
python-dotenv>=1.0
```

---

## 🧪 Tests

The analysis rules above are enforced by tests, not by good intentions — if the code
stops obeying one of them, a test fails.

| Test | What it locks down |
|---|---|
| `test_session_unit.py` | Every time-based number is computed per **session**, and the table matches the chart |
| `test_fade_honesty.py` | Fade is `n/a` for a single rep — no invented `+0.0 %` |
| `test_figures.py` | Every legend sits in a **margin**, never over the data; every session average is drawn *and* numbered |
| `test_display_rules.py` | Whole minutes with no decimals, exact seconds kept beside each, thousands separators on big numbers |
| `test_type_comparison.py` | No series ever mixes training types or duration classes |
| `test_protocol_reps.py` | Per-rep bars, session means and trends agree with the underlying rows |
| `test_set_evolution.py` | Set detection, signatures and the quality gates behave |
| `test_vo2.py` | The VO₂max estimate matches its stated formula |

---

*Catalonia, Spain · 2019–2026 · rubenfm77*
