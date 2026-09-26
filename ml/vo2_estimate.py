# ml/vo2_estimate.py — VO2max estimate from interval power. (NEW FILE)
"""
VO2max from the power-duration curve built out of intervals.icu's interval
rows — watts converted to oxygen cost through gross efficiency:

    VO2 (L/min) = P (W) × 60 s/min ÷ (GE × 20.1 kJ per L O2 × 1000)
    e.g. P = 300 W, GE = 0.21:  300 × 60 ÷ (0.21 × 20 100) = 4.26 L/min

Honesty contract (this app's rules, applied):
  * NO gas exchange was measured — this is an ESTIMATE (±5–10 % vs lab)
    resting on two assumptions, both surfaced in the UI:
    1. gross efficiency — no lab anchor exists (user confirmed), default
       21 % with the full 18–24 % sensitivity band shown;
    2. best 3–6 min power ≈ power at VO2max (PVO2) — EVIDENCE-CHECKED via
       how close the best window effort's HR came to the rider's all-time
       max; confidence is labelled Strong / Medium / Low / Unknown.
  * short efforts carry anaerobic power — P3 vs P6 disagree; all window
    durations are shown, the spread IS part of the uncertainty.
  * reference anchors are qualitative (general adult / trained cyclist /
    elite), not lab percentile tables — no fabricated precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.type_comparison import fmt_min, fmt_secs

KJ_PER_L_O2 = 20.1            # ≈5.05 kcal per litre of O2 (mixed diet)
DEFAULT_GE_PCT = 21.0
GE_BAND_PCT = (18.0, 24.0)
CURVE_BINS = [
    (30, "30 s"), (60, "1 min"), (120, "2 min"), (180, "3 min"),
    (240, "4 min"), (300, "5 min"), (360, "6 min"), (480, "8 min"),
    (600, "10 min"), (720, "12 min"), (1200, "20 min"),
]
PVO2_WINDOW = (180, 360)       # classic 3–6 min range where VO2 reaches max
HR_STRONG = 0.97               # ≥97 % of all-time max HR → truly maximal
HR_MEDIUM = 0.93
ANCHORS = [           # evaluated top-down; first match wins
    (70.0, "elite-cyclist territory"),
    (60.0, "high — well-trained cyclist territory"),
    (50.0, "fit / trained cyclist territory"),
    (40.0, "average-fit adult territory"),
    (0.0, "below average for adults"),
]


def vo2_l_per_min(watts: float, ge_frac: float) -> float:
    """Gross-efficiency conversion: mechanical watts → litres O2 per minute."""
    return 60.0 * float(watts) / (ge_frac * KJ_PER_L_O2 * 1000.0)


def _prep(iv: pd.DataFrame) -> pd.DataFrame:
    w = iv[iv["iv_type"] == "WORK"].copy()
    for c in ("secs", "avg_w", "np_w", "hr_max", "intensity", "cad_avg"):
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w.dropna(subset=["secs", "avg_w"])
    w = w[w["avg_w"] > 0]
    w["date"] = pd.to_datetime(w["date"])
    return w


def _curve(w: pd.DataFrame) -> pd.DataFrame:
    """Best average power near each reference duration (±12 %)."""
    rows = []
    for secs, lab in CURVE_BINS:
        near = w[(w["secs"] >= secs * 0.88) & (w["secs"] <= secs * 1.12)]
        if len(near):
            b = near.loc[near["avg_w"].idxmax()]
            rows.append({"label": lab, "secs": secs,
                         "watts": float(b["avg_w"]), "date": b["date"],
                         "hr_max": float(b["hr_max"])
                         if pd.notna(b["hr_max"]) else np.nan})
        else:
            rows.append({"label": lab, "secs": secs, "watts": np.nan,
                         "date": pd.NaT, "hr_max": np.nan})
    return pd.DataFrame(rows)


def run_vo2_estimate(iv: pd.DataFrame, weight_kg: float,
                     ge_pct: float = DEFAULT_GE_PCT,
                     age: int | None = None) -> dict:
    if iv is None or not len(iv):
        return {"ok": False,
                "reason": "No interval rows cached yet — sync first."}
    w = _prep(iv)
    if not len(w):
        return {"ok": False,
                "reason": "No WORK intervals with watts in the cache."}

    win = w[(w["secs"] >= PVO2_WINDOW[0]) &
            (w["secs"] <= PVO2_WINDOW[1])].sort_values("avg_w",
                                                        ascending=False)
    if not len(win):
        return {"ok": False,
                "reason": f"No WORK efforts between "
                          f"{fmt_min(PVO2_WINDOW[0])}–"
                          f"{fmt_min(PVO2_WINDOW[1])} in the cache — do one "
                          f"hard 3–6 min effort to anchor this estimate."}

    curve = _curve(w)
    best = win.iloc[0]
    pvo2_w = float(best["avg_w"])

    alltime_hr = float(w["hr_max"].max()) if w["hr_max"].notna().any() \
        else np.nan
    eff_hr = float(best["hr_max"]) if pd.notna(best["hr_max"]) else np.nan
    age_pred = (208.0 - 0.7 * age) if age else np.nan

    # Maximality evidence: judge the best window effort that HAS HR — if the
    # absolute best lacks a strap reading, fall back to the strongest
    # HR-recorded one and say so.
    with_hr = win[win["hr_max"].notna()]
    ref = with_hr.iloc[0] if len(with_hr) else None
    if ref is None or np.isnan(alltime_hr):
        conf, conf_why = "Unknown", ("no HR recorded on any effort in the "
                                     "3–6 min window")
    else:
        eff_hr = float(ref["hr_max"])
        frac = eff_hr / alltime_hr if alltime_hr else 0.0
        ref_txt = (f"{fmt_min(ref['secs'])} / "
                   f"{ref['avg_w']:.0f} W on "
                   f"{pd.Timestamp(ref['date']).date()}")
        if frac >= HR_STRONG:
            conf = "Strong"
            conf_why = (f"best HR-recorded window effort ({ref_txt}) "
                        f"reached {eff_hr:.0f} bpm = {frac * 100:.0f} % of "
                        f"your all-time max ({alltime_hr:.0f} bpm)")
        elif frac >= HR_MEDIUM:
            conf = "Medium"
            conf_why = (f"{ref_txt} reached {eff_hr:.0f} bpm = "
                        f"{frac * 100:.0f} % of all-time max "
                        f"({alltime_hr:.0f}) — close, not maximal")
        else:
            conf = "Low"
            conf_why = (f"{ref_txt} reached only {eff_hr:.0f} bpm = "
                        f"{frac * 100:.0f} % of all-time max "
                        f"({alltime_hr:.0f}) — looks paced; treat the "
                        f"estimate as a lower bound")

    ge = ge_pct / 100.0
    vo2_c = vo2_l_per_min(pvo2_w, ge)
    mlkg = vo2_c * 1000.0 / float(weight_kg)

    # Sensitivity: gross efficiency band (user's chosen anchor ±).
    vals = sorted({GE_BAND_PCT[0], ge_pct, GE_BAND_PCT[1]})
    sens = pd.DataFrame([
        {"GE (%)": g,
         "VO₂ (L/min)": round(vo2_l_per_min(pvo2_w, g / 100.0), 2),
         "mL/kg/min": round(vo2_l_per_min(pvo2_w, g / 100.0) * 1000.0
                            / float(weight_kg))}
        for g in vals
    ])
    # window durations → disagreement IS the uncertainty
    p_rows = []
    for r in curve.to_dict("records"):
        if PVO2_WINDOW[0] <= r["secs"] <= PVO2_WINDOW[1] and \
                not np.isnan(r["watts"]):
            p_rows.append({"Effort": r["label"],
                           "Best (W)": round(r["watts"]),
                           "W/kg": round(r["watts"] / float(weight_kg), 2),
                           "VO₂ (L/min) @GE": round(
                               vo2_l_per_min(r["watts"], ge), 2)})
    dur_table = pd.DataFrame(p_rows)

    # Evidence table: the window's strongest efforts with HR maximality.
    ev = win.head(10).copy()
    evidence = pd.DataFrame({
        "Date": ev["date"].dt.strftime("%Y-%m-%d"),
        "Dur": ev["secs"].map(fmt_min),
        "Dur s": ev["secs"].map(fmt_secs),
        "Avg W": ev["avg_w"].round(0),
        "W/kg": (ev["avg_w"] / float(weight_kg)).round(2),
        "IF %": ev["intensity"].round(0),
        "HR max": ev["hr_max"].round(0),
        "% all-time HR": (ev["hr_max"] / alltime_hr * 100).round(0)
        if not np.isnan(alltime_hr) else np.nan,
    })

    anchor = next((txt for cut, txt in ANCHORS if mlkg >= cut),
                  ANCHORS[0][1])
    formula = (f"{pvo2_w:.0f} W × 60 ÷ ({ge:.2f} × {KJ_PER_L_O2} kJ/L × "
               f"1000) = {vo2_c:.2f} L/min ÷ {weight_kg:.1f} kg = "
               f"{mlkg:.0f} mL/kg/min")
    caveats = (
        "No gas exchange — watts are converted with an **assumed gross "
        "efficiency** (no lab test exists; 21 % centre, band 18–24 % above). "
        "Anaerobic contribution inflates shorter efforts, so P3–P6 "
        "disagree; the spread is shown rather than hidden. Body weight is "
        "self-reported (set in the widget above). A paced effort "
        "under-states true VO₂max — read the confidence label before the "
        "number."
    )

    return {
        "ok": True, "curve": curve, "evidence": evidence,
        "pvo2_w": pvo2_w, "pvo2_basis": (
            f"best effort in {fmt_min(PVO2_WINDOW[0])}–"
            f"{fmt_min(PVO2_WINDOW[1])}: {fmt_min(best['secs'])} on "
            f"{pd.Timestamp(best['date']).date()}"),
        "wkg_pvo2": pvo2_w / float(weight_kg),
        "vo2_l": vo2_c, "vo2_mlkg": mlkg,
        "ge_pct": ge_pct, "weight": float(weight_kg),
        "sens": sens, "dur_table": dur_table,
        "conf": conf, "conf_why": conf_why,
        "alltime_hr": alltime_hr, "age_pred": age_pred, "age": age,
        "anchor": anchor, "formula": formula, "caveats": caveats,
        "n_window": int(len(win)),
    }
