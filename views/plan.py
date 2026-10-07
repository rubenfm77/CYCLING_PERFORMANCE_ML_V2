# views/plan.py — Training plan built from the athlete's own history.
#
# One page, three blocks: WHY this base year (the open score), the plan
# itself (dated sessions, each as an intervals.icu paste block), and the
# interval summary with the explanation of every choice. Copy a block with
# its own button, paste it into intervals.icu (Library > Add workout >
# paste the text — sections must stay separated by blank lines), drag it
# onto the date shown beside it.

import pandas as pd
import streamlit as st

from core.components import (callout, dataframe, metric_card, page_header,
                             section)
from core.interval_data import read_intervals, sync_intervals
from core.theme import C
from ml import plan as PL
from ml import interval_watts as iw


def render(head, ctx):
    df_all = ctx.df_all
    page_header(
        "📋",
        "Plan — your own training, replayed forward",
        "No coaching model, no invented sessions: the base year's own "
        "interval mix, your own rests, your own bests as %FTP, laid over a "
        "stated 4 + 4 + 3 + 1 progression.",
    )

    # The interval rows the mix and the references are counted from. Same
    # sync discipline as the Intervals page; without it the plan falls back
    # to the coach's prescribed comments alone and says so.
    now = pd.Timestamp.now().normalize()
    sync_iso = (now - pd.Timedelta(days=370)).isoformat()
    iv_full, synced = None, False
    try:
        with st.spinner("Syncing interval rows from intervals.icu…"):
            sync_intervals(sync_iso)
            iv_full = read_intervals(sync_iso)
            synced = iv_full is not None and len(iv_full) > 0
    except Exception as exc:                                # noqa: BLE001
        st.caption(f"Interval sync unavailable ({type(exc).__name__}) — "
                   "the plan below is built from the coach's comments only.")
    if not synced:
        try:
            iv_full = read_intervals(sync_iso)
            synced = iv_full is not None and len(iv_full) > 0
        except Exception:                                    # noqa: BLE001
            iv_full = None

    P = iw.prescribed(df_all)
    M = iw.measured(df_all)

    # ── 1. WHY this base year ─────────────────────────────────────────────
    section("🏆 The base year, scored in the open")
    yt = PL.year_table(df_all, P, M)
    if not len(yt):
        callout("No history", "No sessions to score — sync first.",
                C["yellow"], icon="⏸️")
        return
    st.caption(
        "**score = FTP-labelled sessions + key-family prescribed sessions + "
        "measured efforts of 10 minutes and up.** Every component is shown, "
        "so the ranking can be argued with instead of trusted. The default "
        "base year is the top row; the picker overrides it."
    )
    show_cols = ["year", "sessions", "ftp_labelled", "prescribed_key",
                 "measured_long", "ftp_setting", "score"]
    dataframe(yt[show_cols], height=min(420, 60 + 34 * len(yt)))
    years = [int(y) for y in yt["year"]]
    base_year = st.selectbox("Base year", years, index=0, key="plan_year",
                             help="The year whose interval mix the plan replays.")
    brow = yt[yt["year"] == base_year].iloc[0]
    kc = st.columns(3)
    with kc[0]:
        metric_card("Base year", f"{base_year}",
                    f"score {int(brow['score'])} · "
                    f"{int(brow['sessions'])} sessions", accent=C["accent"])
    with kc[1]:
        metric_card("Structure source",
                    ("coach comments"
                     if int(brow["prescribed_key"]) > 0 else "measured sets"),
                    f"{int(brow['prescribed_key'])} prescribed key sessions",
                    accent=C["muted"])
    with kc[2]:
        metric_card("Watts source", "your own bests",
                    f"FTP {PL.measured_refs(df_all, iv_full)['ftp']:.0f} W · "
                    "%FTP targets", accent=C["green"])

    # ── 2. the mix ────────────────────────────────────────────────────────
    section("🧱 The mix, counted — not chosen")
    mix = PL.prescribed_mix(P, base_year)
    mix_src = "the coach's prescribed comments"
    if not len(mix) and iv_full is not None and len(iv_full):
        mix = PL.measured_mix(df_all, iv_full, base_year)
        mix_src = "your measured sets (no coach comments that year)"
    if not len(mix):
        callout("No structure", f"No countable interval mix for {base_year} "
                "— pick another base year.", C["yellow"], icon="⏸️")
        return
    st.caption(f"From {mix_src} of {base_year}: one row per "
               "(family × length × reps), most-ridden first. Only combos "
               "ridden this often make the plan — nothing is invented to "
               "fill a week.")
    show_mix = mix.head(8).copy()
    show_mix["rep length"] = (show_mix["rep_secs_med"]
                              .map(lambda s: iw.fmt_rep(s)))
    rest = pd.to_numeric(show_mix.get("rest_med"), errors="coerce")
    show_mix["rest"] = rest.map(
        lambda s: f"{s / 60:.0f} min" if pd.notna(s) else "—")
    dataframe(show_mix[["family", "nominal", "reps_i", "sessions",
                        "rep length", "rest"]]
              .rename(columns={"family": "Type", "nominal": "Min",
                               "reps_i": "Reps", "sessions": "Sessions",
                               "rep length": "Rep", "rest": "Rest taken"}),
              height=min(360, 60 + 34 * len(show_mix)))

    refs = PL.measured_refs(df_all, iv_full)
    wds = PL.weekday_pattern(df_all, base_year)
    st.caption(f"Hard sessions land on **{', '.join(wds)}** — the weekdays "
               f"you actually ride quality sessions in {base_year}, most "
               "frequent first.")

    # ── 3. the plan ───────────────────────────────────────────────────────
    section("📅 The plan — paste each block into intervals.icu")
    c3a, c3b = st.columns(2)
    with c3a:
        start = st.date_input("Week 1 starts",
                              value=(now + pd.Timedelta(
                                  days=(7 - now.dayofweek) % 7)).date(),
                              key="plan_start")
    with c3b:
        weeks = st.selectbox("Weeks", [8, 12, 16], index=1, key="plan_n")
    plan = PL.build_plan(mix, refs, wds, pd.Timestamp(start),
                         weeks=int(weeks),
                         rest_fallback=PL.rest_fallback_for(mix))
    st.caption(
        "Library > Add workout > paste the block (keep the blank lines — "
        "intervals.icu parses the steps from them), then drag the workout "
        "onto the date beside it. Targets are %FTP: they follow your current "
        f"FTP setting ({refs['ftp']:.0f} W), so the same text stays right "
        "when your FTP moves."
    )
    for wk in plan:
        with st.expander(f"Week {wk['week']} · {wk['phase']} — {wk['note']}",
                         expanded=(wk["week"] == 1)):
            for s in wk["sessions"]:
                st.markdown(f"**{s['date']} ({s['day']}) — {s['title']}** · "
                            f"{s['target']}"
                            + (f" · rest {PL.fmt_dur(s['rest_min'])}"
                               if s["reps"] > 1 else ""))
                st.caption(s["reference"])
                st.code(s["text"], language="text")

    # ── 4. summary + why ──────────────────────────────────────────────────
    section("📝 Every interval, and why it is there")
    summ = PL.plan_summary(plan)
    dataframe(summ, height=min(480, 60 + 34 * len(summ)))
    st.caption(
        f"**Why this plan.** Base year {base_year} scored "
        f"{int(brow['score'])} ({int(brow['ftp_labelled'])} FTP-labelled + "
        f"{int(brow['prescribed_key'])} prescribed key + "
        f"{int(brow['measured_long'])} measured long). Its mix (above) "
        "decides the sessions; its weekday pattern decides the days; the "
        "rests are the medians you actually took "
        f"({PL.rest_fallback_for(mix):.0f} min fallback for long sets); the "
        f"targets are %FTP against your {refs['ftp']:.0f} W setting, with "
        "your own best beside each session as the reality check. Weeks 1-4 "
        "replay the mix as ridden, 5-8 add one rep, 9-11 hold the longest "
        "variant, 12 halves everything — a plain progression, stated, not "
        "copied. **Limits:** no measured interval rows exist before 2025, "
        "coach comments end in 2025, one ride a day is drawn hardest-first "
        "where a double day exists, and duplicate sync copies are counted "
        "once (see the Intervals quality screen)."
    )
