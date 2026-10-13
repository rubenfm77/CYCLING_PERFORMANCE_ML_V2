# views/plan.py — Weekly plan from weeks like this one.
#
# Four blocks, no coaching model: (1) now — CTL/ATL/TSB, 28-day TSS, quality
# share, main types, form reading; (2) what similar past weeks were followed
# by — association with n and medians, never a causal claim; (3) three
# ranked week templates with TSS ranges, quality sessions in the athlete's
# bins, rest days and the outcome line; (4) the chosen template as
# intervals.icu paste blocks, one per workout, dated onto next week.

import pandas as pd
import streamlit as st

from core.components import (callout, dataframe, metric_card, page_header,
                             section)
from core.theme import C
from ml import interval_watts as iw
from ml import plan as PL
from ml import weekpattern as WP

_FAMBIN = [("ftp_n", "FTP"), ("vo2_n", "VO2MAX"), ("billat_n", "BILLAT"),
           ("sst_n", "SST"), ("tempo_n", "TEMPO")]


def _nominal_for(mix, family):
    """Top template of one family: (nominal min, reps, rest min or NaN)."""
    import numpy as np
    m = mix[mix["family"] == family] if mix is not None and len(mix) else None
    if m is None or not len(m):
        return None
    r = m.iloc[0]
    try:
        rest = float(r.get("rest_med"))
    except (TypeError, ValueError):
        rest = float("nan")
    return (float(r["nominal"]), int(r.get("reps_i", 1) or 1), rest)


def _resolve_sessions(pat, mix, refs, ftp, df_all):
    """Pattern row -> pasteable sessions. Never empty, never invented quietly.

    Each pattern family resolves against the base-year mix (the athlete's own
    nominals, reps and measured rests); a family missing from the mix falls
    back to the book's staple for that bin, noted as book-standard. A pattern
    with no quality sessions becomes a recovery week: a recovery spin plus
    Z2 endurance blocks at the athlete's own easy-ride duration — still
    pasteable, still honest about what it is.
    """
    import numpy as np
    out, notes = [], []
    for col, fam in _FAMBIN:
        n = int(pat.get(col, 0) or 0)
        for _ in range(n):
            got = _nominal_for(mix, fam)
            if got is None:
                if fam not in PL.BOOK_DEFAULTS:
                    notes.append(f"{fam}: in the pattern but neither in the "
                                 "base mix nor the book staples — skipped")
                    continue
                nom, reps = PL.BOOK_DEFAULTS[fam]
                rest = PL.book_rest_for(fam, nom)
                notes.append(f"{fam}: book-standard "
                             f"{reps}x{PL.fmt_dur(nom)} — not in the base mix")
            else:
                nom, reps, rest = got
                if not (np.isfinite(rest) and rest > 0):
                    rest = PL.book_rest_for(fam, nom)
                else:
                    rest = rest / 60.0
            ref = refs.get("best", {}).get(float(nom))
            out.append(PL.build_session(
                fam, nom, reps, rest, ftp=ftp,
                ref=(ref if ref else None)))
    if not out:
        notes.append("recovery week: no quality sessions in this pattern — "
                     "recovery spin plus endurance blocks instead of blanks")
        easy_h = PL.endurance_med_duration(df_all)
        out.append(PL.build_session(
            "REC", 45.0, 1, None, ftp=ftp))
        for _ in range(2):
            out.append(PL.build_session(
                "Z2", easy_h * 60.0, 1, None, ftp=ftp))
    return out, notes


def _staple_sessions(refs, ftp):
    """Book staples for a climber when nothing else exists to learn from."""
    out = []
    for fam, nom, reps in [("FTP", 20.0, 2), ("VO2MAX", 5.0, 5),
                           ("SST", 12.0, 3)]:
        ref = refs.get("best", {}).get(float(nom))
        out.append(PL.build_session(
            fam, nom, reps, PL.book_rest_for(fam, nom), ftp=ftp,
            ref=(ref if ref else None)))
    return out, ["book staples for a climber (threshold + VO2 + sweet spot) "
                 "— no pattern or mix to learn them from"]


def render(head, ctx):
    df_all = ctx.df_all
    page_header(
        "📋",
        "Plan — weeks like this one, and what followed them",
        "Association, not prescription: past weeks under similar season and "
        "fitness, what the fitness model did next, and the patterns ranked "
        "by that — with n attached so small samples read as small.",
    )

    W = WP.weekly_frame(df_all)
    if not len(W):
        callout("No weeks", "No sessions to build weeks from — sync first.",
                C["yellow"], icon="⏸️")
        return
    stt = WP.current_state(df_all)

    # ── 1. NOW ────────────────────────────────────────────────────────────
    section("📍 Now")
    c0 = st.columns(4)
    with c0[0]:
        metric_card("CTL / ATL / TSB",
                    f"{stt['ctl']:.0f} / {stt['atl']:.0f} / {stt['tsb']:+.0f}",
                    f"form reading: {stt['form']} · {stt['date']}",
                    accent=C["accent"])
    with c0[1]:
        metric_card("Last 28 days", f"{stt['tss_28']:.0f} TSS",
                    f"{stt['sessions_28']} sessions",
                    accent=C["muted"])
    with c0[2]:
        metric_card("Quality share", f"{stt['quality_share'] * 100:.0f} %",
                    "sessions under FTP/VO2MAX/BILLAT/SST/TEMPO",
                    accent=C["green"])
    with c0[3]:
        metric_card("Main types",
                    ", ".join(stt["main_types"]) if stt["main_types"] else "—",
                    "28-day sessions by label", accent=C["muted"])
    import numpy as _np0
    _acwr0 = stt.get("acwr", float("nan"))
    _acwr0_txt = (f"{_acwr0:.2f}" if _np0.isfinite(_acwr0) else "—")
    st.caption(
        "Form reading off TSB bands (fresh ≥ +10, productive −10…+10, "
        "fatigued −30…−10, overreached < −30) — a conventional reading, not "
        "a diagnosis. eFTP is only carried from 2025 on, so long-horizon "
        "outcomes below use the CTL model, which is a model of load, not "
        "fitness itself. "
        f"ACWR (7-day vs 28-day load): {_acwr0_txt} "
        f"— cap 1.2 post-surgery, 1.3 otherwise."
    )
    # ── 2. WHAT SIMILAR WEEKS WERE FOLLOWED BY ────────────────────────────
    section("🔁 What weeks like this were followed by")
    S, tag, crit = WP.find_similar(W, stt["month"], stt["ctl"], stt["tsb"])
    st.caption(
        f"**{len(S)} similar weeks** (season ±{crit['season_months']} month(s), "
        f"starting CTL ±{crit['ctl_band']:.0f}, TSB ±{crit['tsb_band']:.0f} — "
        f"criteria: **{tag}**). Outcome = the fitness model's change over "
        "the next 28 days. Read every median WITH its n: a +8 on n=1 is one "
        "week that happened once, not a recipe."
    )
    if not len(S):
        callout("No matches", "No past week starts like this one — the "
                "templates below fall back to the base-year mix.",
                C["yellow"], icon="⏸️")
    else:
        show_s = S[["week", "tss", "quality", "ftp_n", "vo2_n", "billat_n",
                    "rest_days", "d_ctl_28", "d_eftp_28"]].copy()
        show_s["week"] = pd.to_datetime(show_s["week"]).dt.strftime("%Y-%m-%d")
        dataframe(show_s.rename(columns={
            "week": "Week", "tss": "TSS", "quality": "Quality",
            "ftp_n": "FTP", "vo2_n": "VO2", "billat_n": "Billat",
            "rest_days": "Rest days", "d_ctl_28": "ΔCTL next 28d",
            "d_eftp_28": "ΔeFTP next 28d"}),
            height=min(420, 60 + 34 * len(show_s)))
    R = WP.rank_patterns(S)
    ranked = R[R["ranked"]] if len(R) else R
    st.caption(
        f"**{len(ranked)} ranked pattern(s)** (≥{WP.MIN_WEEKS_RANK} weeks "
        "each — fewer than that is listed, never ranked). Ranked by median "
        "next-28-day CTL change; eFTP medians ride along only where the "
        "file carries eFTP."
    )

    # ── 3. THREE TEMPLATES ────────────────────────────────────────────────
    section("🏅 Three week templates, ranked by what followed")
    P = iw.prescribed(df_all)
    yt = PL.year_table(df_all, P, None)
    base_year = int(yt["year"].iloc[0]) if len(yt) else None
    mix = PL.prescribed_mix(P, base_year) if base_year else None
    if (mix is None or not len(mix)) and base_year:
        try:
            from core.interval_data import read_intervals as _riv
            _iv = _riv((pd.Timestamp.now().normalize()
                        - pd.Timedelta(days=370)).isoformat())
            mix = PL.measured_mix(df_all, _iv, base_year)
        except Exception:                                    # noqa: BLE001
            mix = None
    try:
        from core.interval_data import read_intervals as _riv2
        _iv2 = _riv2((pd.Timestamp.now().normalize()
                      - pd.Timedelta(days=370)).isoformat())
    except Exception:                                        # noqa: BLE001
        _iv2 = None
    refs = PL.measured_refs(df_all, _iv2)

    pats = (ranked.head(3).to_dict("records") if len(ranked)
            else (R.head(3).to_dict("records") if len(R) else []))
    ftp_set = float(refs.get("ftp", 240.0)) if isinstance(refs, dict) else 240.0
    templates = []
    for i, p in enumerate(pats, 1):
        sess, notes = _resolve_sessions(p, mix, refs, ftp_set, df_all)
        templates.append({
            "rank": i,
            "ranked": bool(p.get("ranked", False)),
            "n": int(p["n"]),
            "tss": (float(p["tss_lo"]), float(p["tss_hi"])),
            "rest_days": int(p["rest_days"]),
            "d_ctl": (float(p["d_ctl_med"]), float(p["d_ctl_lo"]),
                      float(p["d_ctl_hi"])),
            "d_eftp": (p.get("d_eftp_med"), int(p.get("d_eftp_n", 0) or 0)),
            "mix": {fam: int(p.get(col, 0) or 0) for col, fam in _FAMBIN},
            "sessions": sess, "notes": notes,
        })
    if not templates:
        sess, notes = _staple_sessions(refs, ftp_set)
        _bt = WP.base_week_tss(df_all, base_year) if base_year else (
            float("nan"),) * 3
        templates.append({"rank": 1, "ranked": False, "n": 0,
                          "tss": (_bt[1], _bt[2]),
                          "rest_days": 2, "d_ctl": (float("nan"),) * 3,
                          "d_eftp": (None, 0),
                          "mix": {"FTP": 1, "VO2MAX": 1, "SST": 1},
                          "sessions": sess,
                          "notes": notes + [
                              "no similar week to learn from — TSS range is "
                              f"the {base_year} weeks' own IQR, sessions the "
                              "book staples"]})
    if not templates:
        callout("No template", "Neither similar weeks nor a base-year mix "
                "exist — sync interval rows first.", C["yellow"], icon="⏸️")
        return
    for t in templates:
        import numpy as np
        mix_txt = " + ".join(f"{v}× {k}" for k, v in t["mix"].items() if v)
        eftp_txt = (f" · median ΔeFTP {t['d_eftp'][0]:+.0f} W "
                    f"(n={t['d_eftp'][1]})" if t["d_eftp"][1] else "")
        tss_txt = (f"TSS {t['tss'][0]:.0f}–{t['tss'][1]:.0f}"
                   if np.isfinite(t['tss'][0]) and np.isfinite(t['tss'][1])
                   else "TSS —")
        out_txt = (f"median next-28d ΔCTL {t['d_ctl'][0]:+.1f} "
                   f"(IQR {t['d_ctl'][1]:+.1f}…{t['d_ctl'][2]:+.1f})"
                   if np.isfinite(t['d_ctl'][0]) else "no outcome to cite")
        st.markdown(
            f"**Option {t['rank']}{' (ranked)' if t['ranked'] else ' (single example — not ranked)'}** · "
            f"{tss_txt} · {mix_txt or 'recovery week'} · "
            f"{t['rest_days']} rest days · "
            f"based on N={t['n']} similar week(s) → "
            f"{out_txt}{eftp_txt}"
        )
        for note in t["notes"]:
            st.caption(f"· {note}")
    choice = st.radio("Template to paste", [f"Option {t['rank']}" for t in
                                            templates],
                      index=0, key="plan_template", horizontal=True)
    tpl = next(t for t in templates
               if f"Option {t['rank']}" == choice)

    # ── 4. INTERVALS.ICU FORMAT ───────────────────────────────────────────
    section("📋 Paste into intervals.icu")
    st.caption(
        "Library > Add workout > paste one block (keep the blank lines — "
        "intervals.icu parses the steps from them), then drag it onto the "
        "date beside it. Every block carries %FTP and watts at your current "
        f"FTP setting ({ftp_set:.0f} W); steps carry %FTP only so the paste "
        "survives your next FTP change. Recovery days are riding easy or "
        "rest — the pattern's rest-day count, not a prescription."
    )
    nxt = pd.Timestamp.now().normalize() + pd.Timedelta(
        days=(7 - pd.Timestamp.now().dayofweek) % 7)
    wds = PL.weekday_pattern(df_all, base_year) if base_year else ["Tue",
                                                                   "Thu",
                                                                   "Sat"]
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dated = []
    for i, s in enumerate(tpl["sessions"]):
        day = wds[i % len(wds)] if wds else order[i % 7]
        date = nxt + pd.Timedelta(days=(order.index(day)
                                        - nxt.dayofweek) % 7)
        dated.append({**s, "day": day, "date": date.date()})
    for s in dated:
        st.markdown(f"**{s['date']} ({s['day']}) — {s['title']}** · "
                    f"{s['purpose']} · ~{s['total_min']:.0f} min total"
                    + (f" · rest {PL.fmt_dur(s['rest_min'])}"
                       if s["reps"] > 1 else ""))
        st.caption(s["reference"])
        st.code(s["text"], language="text")
    st.markdown("**Every interval of this week, one row each:**")
    dataframe(PL.session_table(dated), height=min(420, 60 + 34 * len(dated)))
    import numpy as _np
    _acwr_txt = (f"{stt.get('acwr', float('nan')):.2f}"
                 if _np.isfinite(stt.get("acwr", float("nan"))) else "—")
    st.caption(
        "**Adaptation rules (book).** Missed 1 session → skip it, don't "
        "stack; missed 3+ days → repeat last week at −10% TSS. "
        f"TSB {stt['tsb']:+.0f}: below −25, convert the next key session to "
        "Z2. "
        f"ACWR {_acwr_txt} (7d vs 28d; cap 1.2 post-surgery, 1.3 otherwise): "
        "above the cap, hold next week at the 28-day average load. Two "
        "consecutive failed key sessions → recovery week now, calendar "
        "aside. At 48 with a 2:1 load:recovery rhythm, every third week "
        "should look like the recovery template above."
    )
    st.caption(
        "**Why this template.** Its shape is the shape past weeks like this "
        "one had; its outcome line is the median of what followed those "
        "weeks, with n and spread attached — association, not causation, and "
        "a +8 on n=1 is one good month that happened once. Durations come "
        f"from the {base_year} mix (the nominals you actually ride), rests "
        "from the medians you took, watts from your own bests as %FTP. "
        "Limits: eFTP outcomes exist only from 2025 on; CTL is a load model, "
        "not fitness; duplicate sync copies count once."
    )
