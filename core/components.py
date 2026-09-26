# core/components.py — the modern UI kit. (NEW FILE, rewritten for the
# multipage layout: adds the persistent top bar, page headers and the
# navigation pill styling on top of the original card/badge system.)
#
# Design contract (agreed with the user): KEEP the GitHub-dark + blue identity
# (#0d1117 / #161b22 / #58a6ff, blue headings, blue metric values) but present
# it with fresh components — gradient hero panels with glow, cards with an
# accent strip + hover lift, icon-chip section headers, pill nav bar.

import inspect
import math
import re

import streamlit as st
import pandas as pd

from core.theme import C, CHART_CONFIG

# ── CSS (injected once per run, before any page content) ─────────────────────
_CSS = """
<style>
  /* ── Foundation: GitHub-dark blue identity, modernized ───────────────── */
  :root {
    --bg:#0d1117; --panel:#161b22; --line:#21262d; --line2:#30363d;
    --blue:#58a6ff; --txt:#c9d1d9; --mut:#8b949e;
  }
  html, body { background: var(--bg); }
  .stApp {
    background-color: var(--bg);
    font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', Arial, sans-serif;
  }
  .stSidebar { background-color: #12171f; border-right: 1px solid var(--line); }
  h1, h2, h3, h4 { color: #58a6ff !important; font-weight: 700; letter-spacing: -0.01em; }

  ::selection { background: rgba(88,166,255,0.35); }
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-track { background: #0d1117; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 8px; }
  ::-webkit-scrollbar-thumb:hover { background: #484f58; }

  /* ── Persistent top bar: brand + live status + state/KPI chips ───────── */
  .cm-topbar {
    display: flex; justify-content: space-between; align-items: center;
    gap: 14px; flex-wrap: wrap;
    padding: 8px 2px 12px; margin-bottom: 2px;
    border-bottom: 1px solid var(--line);
  }
  .cm-brand {
    font-size: 1.18rem; font-weight: 800; color: #58a6ff;
    letter-spacing: -0.01em; display: flex; align-items: center; gap: 10px;
  }
  .cm-brand-sub {
    font-size: 0.64rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #8b949e;
    background: #161b22; border: 1px solid #30363d;
    border-radius: 999px; padding: 3px 10px;
  }
  .cm-top-right { display: flex; align-items: center; gap: 8px;
                  flex-wrap: wrap; justify-content: flex-end; }

  .cm-live {
    display: inline-flex; align-items: center; gap: 7px;
    font-size: 0.66rem; font-weight: 800; letter-spacing: 0.1em;
    color: #3fb950; background: rgba(63,185,80,0.10);
    border: 1px solid rgba(63,185,80,0.35);
    border-radius: 999px; padding: 5px 12px;
  }
  .cm-live .cm-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #3fb950;
    box-shadow: 0 0 8px rgba(63,185,80,0.9); animation: cm-pulse 2.2s infinite;
  }
  .cm-live.cm-off { color: #8b949e; background: rgba(139,148,158,0.08);
                    border-color: #30363d; }
  .cm-live.cm-off .cm-dot { background: #8b949e; box-shadow: none; animation: none; }
  @keyframes cm-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

  .cm-chip {
    display: inline-flex; align-items: baseline; gap: 7px;
    background: #161b22; border: 1px solid #21262d;
    border-radius: 999px; padding: 5px 13px;
  }
  .cm-chip-k { font-size: 0.6rem; letter-spacing: 0.12em;
               color: #8b949e; font-weight: 700; }
  .cm-chip-v { font-size: 0.84rem; font-weight: 800; color: #58a6ff; }

  /* ── Top navigation pill bar (st.radio, horizontal) ──────────────────── */
  div[data-testid="stRadio"] { padding-top: 0; margin-bottom: 4px; }
  div[data-testid="stRadio"] div[role="radiogroup"] {
    background: #11161d; border: 1px solid var(--line);
    border-radius: 14px; padding: 6px; gap: 6px !important;
  }
  div[data-testid="stRadio"] label {
    background: transparent; border: 1px solid transparent;
    border-radius: 10px; padding: 8px 18px !important;
    transition: all 0.15s ease;
  }
  div[data-testid="stRadio"] label p {
    color: #8b949e; font-weight: 600; font-size: 0.9rem;
  }
  div[data-testid="stRadio"] label:hover { background: #161b22; border-color: #30363d; }
  div[data-testid="stRadio"] label:hover p { color: #58a6ff; }
  div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    border-color: rgba(120,190,255,0.55);
    box-shadow: 0 3px 14px rgba(56,139,253,0.35);
  }
  div[data-testid="stRadio"] label:has(input:checked) p {
    color: #ffffff !important; font-weight: 700;
  }

  /* ── Hero panel (overview) + compact page header ─────────────────────── */
  .cm-hero {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, #161b22 0%, #151b26 55%, #131a28 100%);
    border: 1px solid #262c36; border-left: 4px solid #58a6ff;
    border-radius: 16px; padding: 22px 26px; margin: 4px 0 14px;
  }
  .cm-hero::after {
    content: ""; position: absolute; top: -170px; right: -130px;
    width: 390px; height: 390px; pointer-events: none;
    background: radial-gradient(circle, rgba(88,166,255,0.14), transparent 65%);
  }
  .cm-hero h1 { margin: 0; font-size: 1.65rem; }
  .cm-hero .cm-sub { color: #8b949e; font-size: 0.86rem; margin-top: 6px; }
  .cm-hero .cm-right { text-align: right; position: relative; z-index: 1; }
  .cm-hero-sm {
    display: flex; gap: 16px; align-items: center; padding: 16px 24px;
  }
  .cm-hero-sm h2 { margin: 0; font-size: 1.22rem; }
  .cm-hero-ic {
    width: 44px; height: 44px; flex: none; border-radius: 13px;
    font-size: 1.25rem; display: flex; align-items: center; justify-content: center;
    background: rgba(88,166,255,0.10); border: 1px solid rgba(88,166,255,0.35);
  }

  /* ── KPI card: accent strip, hover lift, auto height (never clipped) ─── */
  .cm-card {
    position: relative; overflow: hidden;
    background: linear-gradient(180deg, #161b22 0%, #141a22 100%);
    border: 1px solid var(--line); border-radius: 14px;
    padding: 15px 17px; min-height: 120px;
    transition: transform 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease;
  }
  .cm-card::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--acc, #58a6ff), rgba(0,0,0,0) 72%);
    opacity: 0.75;
  }
  .cm-card:hover {
    transform: translateY(-2px); border-color: #30363d;
    box-shadow: 0 10px 26px rgba(0,0,0,0.45);
  }
  .cm-card .cm-label {
    font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #8b949e; font-weight: 700;
    display: flex; align-items: center; gap: 6px;
  }
  .cm-card .cm-chipdot {
    width: 7px; height: 7px; flex: none; border-radius: 2.5px;
    background: var(--acc, #58a6ff);
  }
  .cm-card .cm-value {
    font-size: 1.85rem; font-weight: 800; color: #58a6ff;   /* original blue value */
    margin-top: 7px; line-height: 1.08; letter-spacing: -0.02em;
  }
  .cm-card .cm-value small { font-size: 0.95rem; font-weight: 500;
                             color: #8b949e; margin-left: 3px; }
  .cm-card .cm-delta { font-size: 0.78rem; margin-top: 6px; font-weight: 700; }
  .cm-card .cm-foot  { font-size: 0.72rem; color: #8b949e;
                       margin-top: 5px; line-height: 1.35; }
  .cm-card .cm-spark { position: absolute; right: 10px; bottom: 8px; opacity: 0.85; }

  /* ── Section header: icon chip + title + fading divider ──────────────── */
  .cm-section { display: flex; align-items: center; gap: 10px; margin: 28px 0 12px; }
  .cm-sec-chip {
    width: 27px; height: 27px; flex: none; border-radius: 9px; font-size: 0.88rem;
    display: flex; align-items: center; justify-content: center;
    background: rgba(88,166,255,0.10); border: 1px solid rgba(88,166,255,0.32);
  }
  .cm-sec-title {
    font-size: 0.9rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; color: #58a6ff; white-space: nowrap;
  }
  .cm-sec-line { flex: 1; height: 1px;
                 background: linear-gradient(90deg, #30363d, rgba(48,54,61,0)); }

  /* ── State badge / callout ───────────────────────────────────────────── */
  .cm-badge {
    border-radius: 12px; padding: 15px 18px; margin: 6px 0 12px;
    border: 1px solid; border-left-width: 4px;
    font-size: 0.95rem; position: relative; overflow: hidden;
  }
  .cm-badge .cm-badge-title { font-weight: 700; font-size: 1.02rem; margin-bottom: 4px; }
  .cm-badge .cm-badge-body  { color: #c9d1d9; font-size: 0.88rem; line-height: 1.45; }
  .cm-badge .cm-badge-meta  { color: #8b949e; font-size: 0.76rem; margin-top: 8px; }

  /* ── Distribution stat blocks ────────────────────────────────────────── */
  .cm-stat {
    background: #161b22; border: 1px solid #21262d; border-radius: 14px;
    padding: 16px 18px; text-align: center;
    transition: transform 0.16s ease;
  }
  .cm-stat:hover { transform: translateY(-2px); }
  .cm-stat .cm-stat-n { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em; }
  .cm-stat .cm-stat-l { font-size: 0.74rem; color: #8b949e; margin-top: 4px; }
  .cm-stat .cm-stat-p { font-size: 0.72rem; margin-top: 6px; font-weight: 700; }

  /* ── Buttons, tables, sidebar chrome ─────────────────────────────────── */
  div[data-testid="stButton"] > button {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    color: #c9d1d9; font-weight: 600; transition: all 0.15s ease;
  }
  div[data-testid="stButton"] > button:hover {
    border-color: #58a6ff; color: #58a6ff; background: #161b22;
  }
  [data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 12px;
                                overflow: hidden; }
  [data-baseweb="select"] > div { border-radius: 10px; }
</style>
"""


def inject_css() -> None:
    """Call once, early in the entry script (before st.navigation runs)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def _tone_color(tone: str | None) -> str:
    return {"good": C["green"], "bad": C["red"], "warn": C["yellow"]}.get(tone or "", C["muted"])


def sparkline(values, color: str = None, width: int = 104, height: int = 26) -> str:
    """Tiny inline SVG sparkline — KPI context without a full Plotly chart."""
    color = color or C["accent"]
    vals = []
    for v in list(values)[-40:]:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isnan(fv):
            vals.append(fv)
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    pts = []
    for i, v in enumerate(vals):
        x = 2 + i / (len(vals) - 1) * (width - 4)
        y = height - 2 - (v - lo) / rng * (height - 4)
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    last = pts[-1].split(",")
    return (
        f'<svg class="cm-spark" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="1.6" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{last[0]}" cy="{last[1]}" r="2.2" fill="{color}"/></svg>'
    )


def topbar(head, sync: dict) -> None:
    """Persistent app header: brand, live-sync status, state + KPI chips.

    Rendered by the router on EVERY page, so the coach always sees the
    current state/load regardless of where they are.
    """
    ok = bool(sync.get("ok"))
    live = (
        '<span class="cm-live"><span class="cm-dot"></span>LIVE · INTERVALS.ICU</span>'
        if ok else
        '<span class="cm-live cm-off"><span class="cm-dot"></span>LOCAL DATA</span>'
    )

    def _chip(k: str, v: str, col: str = "#58a6ff") -> str:
        return (f'<div class="cm-chip" style="--cc:{col}"><span class="cm-chip-k">{k}</span>'
                f'<span class="cm-chip-v" style="color:{col}">{v}</span></div>')

    eftp = f"{head.eftp_val:.0f}W" if head.eftp_val else "—"
    chips = (
        _chip("STATE", head.state, head.state_color)
        + _chip("CTL", f"{head.ctl_now:.1f}")
        + _chip("ATL", f"{head.atl_now:.1f}")
        + _chip("TSB", f"{head.tsb_now:+.1f}")
        + _chip("eFTP", eftp)
        + _chip("W/KG", f"{head.wkg_val:.2f}")
    )
    st.markdown(
        f'<div class="cm-topbar">'
        f'<div class="cm-brand">🚴 Cycling Performance'
        f'<span class="cm-brand-sub">pro coach view</span></div>'
        f'<div class="cm-top-right">{live}{chips}</div></div>',
        unsafe_allow_html=True,
    )


def page_header(icon: str, title: str, subtitle: str) -> None:
    """Compact hero for secondary pages: icon chip + title + guidance line."""
    st.markdown(
        f'<div class="cm-hero cm-hero-sm"><div class="cm-hero-ic">{icon}</div>'
        f'<div><h2>{title}</h2><div class="cm-sub">{subtitle}</div></div></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value, delta: str | None = None, tone: str | None = None,
                foot: str | None = None, accent: str | None = None,
                spark=None, spark_color: str | None = None, icon: str | None = None):
    """KPI tile. tone in {'good','bad','warn',None} colours the delta."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        value = "—"
    acc = accent or C["accent"]
    label_html = f"{icon} {label}" if icon else label
    delta_html = ""
    if delta:
        delta_html = f'<div class="cm-delta" style="color:{_tone_color(tone)}">{delta}</div>'
    foot_html = f'<div class="cm-foot">{foot}</div>' if foot else ""
    spark_html = sparkline(spark, spark_color or acc) if spark is not None else ""
    st.markdown(
        f'<div class="cm-card" style="--acc:{acc}">'
        f'<div class="cm-label"><span class="cm-chipdot"></span>{label_html}</div>'
        f'<div class="cm-value">{value}</div>'
        f'{delta_html}{foot_html}{spark_html}</div>',
        unsafe_allow_html=True,
    )


def stat_block(n: str, label: str, pct: str, color: str):
    st.markdown(
        f'<div class="cm-stat">'
        f'<div class="cm-stat-n" style="color:{color}">{n}</div>'
        f'<div class="cm-stat-l">{label}</div>'
        f'<div class="cm-stat-p" style="color:{color}">{pct}</div></div>',
        unsafe_allow_html=True,
    )


def badge(title: str, body: str, color: str, meta: str | None = None):
    meta_html = f'<div class="cm-badge-meta">{meta}</div>' if meta else ""
    st.markdown(
        f'<div class="cm-badge" style="border-color:{color};background:{color}14;">'
        f'<div class="cm-badge-title" style="color:{color}">{title}</div>'
        f'<div class="cm-badge-body">{body}</div>{meta_html}</div>',
        unsafe_allow_html=True,
    )


def callout(title: str, body: str, color: str, icon: str = "ℹ️"):
    # This element is injected as raw HTML, and markdown is not parsed
    # inside block-level HTML — **bold** would show as literal asterisks.
    # Convert the two inline marks callers actually use.
    body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", str(body))
    body = re.sub(r"`([^`]+)`", r"<code>\1</code>", body)
    st.markdown(
        f'<div class="cm-badge" style="border-color:{color}66;background:{color}10;">'
        f'<div class="cm-badge-title" style="color:{color}">{icon} {title}</div>'
        f'<div class="cm-badge-body">{body}</div></div>',
        unsafe_allow_html=True,
    )


def section(label: str):
    """Section header: icon chip + title + fading divider line.

    A leading emoji in `label` becomes the chip icon; labels without one
    get a neutral marker.
    """
    token = label.split(" ", 1)[0]
    if token and not token[0].isascii():
        icon, text = token, (label.split(" ", 1)[1] if " " in label else "")
    else:
        icon, text = "▸", label
    st.markdown(
        f'<div class="cm-section"><span class="cm-sec-chip">{icon}</span>'
        f'<span class="cm-sec-title">{text}</span><span class="cm-sec-line"></span></div>',
        unsafe_allow_html=True,
    )


def show(fig, height: int | None = None):
    """Render a Plotly figure compatibly across Streamlit versions."""
    if height:
        fig.update_layout(height=height)
    params = inspect.signature(st.plotly_chart).parameters
    if "width" in params:
        st.plotly_chart(fig, config=CHART_CONFIG, width="stretch")
    else:  # older API
        st.plotly_chart(fig, config=CHART_CONFIG, use_container_width=True)


def legend(fig, side: str = "right", horizontal: bool = False, size: int = 11):
    """Park the legend INSIDE the figure canvas so it can never be clipped.

    app.py relied on plotly defaults; an earlier pass put legends at y=1.06 —
    outside the canvas — so every reference-line key was invisible.
    """
    x, anchor = (0.995, "right") if side == "right" else (0.005, "left")
    d = dict(x=x, y=0.995, xanchor=anchor, yanchor="top",
             bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363d",
             font=dict(size=size, color="#c9d1d9"))
    if horizontal:
        d.update(orientation="h")
    fig.update_layout(legend=d)
    return fig


def dataframe(df: pd.DataFrame, height: int = 380, **kwargs):
    params = inspect.signature(st.dataframe).parameters
    if "width" in params:
        st.dataframe(df, width="stretch", height=height, **kwargs)
    else:
        st.dataframe(df, use_container_width=True, height=height, **kwargs)
