# app_modern.py — Cycling Performance Dashboard, modern edition (NEW FILE)
# =====================================================================
# Run with:  python -m streamlit run app_modern.py
#
# app.py is untouched and still runs with:  python -m streamlit run app.py
#
# This file is the ROUTER of a real multipage app (st.navigation / st.Page):
#   - persistent chrome on every page: top bar (brand + live status + state/
#     KPI chips), the navigation pill bar, and the sidebar filters;
#   - seven pages with real URLs (/fitness, /forecast, /intervals, ...):
#     Overview · Fitness · Forecast · Intervals · Training · Trends · Sessions.
# Design: keep the GitHub-dark + blue identity of app.py, presented with
# modern components (gradient hero, accent-strip cards, icon section headers).

from functools import partial

import streamlit as st

st.set_page_config(
    page_title="Cycling Performance",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.components import inject_css, topbar          # noqa: E402
from core.context import frames, render_sidebar         # noqa: E402
from core.data import load_data                         # noqa: E402
from core.metrics import headline                       # noqa: E402
from views import (fitness, forecast, intervals_view, overview, sessions,  # noqa: E402
                   training, trends)

inject_css()

# ── Data + shared sidebar (widgets live here so state survives page changes) ─
df_all = load_data()
render_sidebar(df_all)
ctx = frames(df_all)
head = headline(ctx)

# ── Page registry ────────────────────────────────────────────────────────────
_SPECS = [
    # slug, icon, title, callable, is_default
    ("home",     "🏠", "Overview", partial(overview.render, head, ctx), True),
    ("fitness",  "📈", "Fitness",  partial(fitness.render, head, ctx),  False),
    ("forecast", "🔮", "Forecast", partial(forecast.render, head, ctx), False),
    ("intervals", "🔬", "Intervals", partial(intervals_view.render, head, ctx),
     False),
    ("training", "🏋️", "Training", partial(training.render, head, ctx), False),
    ("trends",   "📊", "Trends",   partial(trends.render, head, ctx),   False),
    ("sessions", "📅", "Sessions", partial(sessions.render, head, ctx), False),
]

_pages, _by_slug = [], {}
for _slug, _icon, _title, _fn, _is_default in _SPECS:
    if _is_default:
        _p = st.Page(_fn, title=_title, icon=_icon, default=True)
    else:
        _p = st.Page(_fn, title=_title, icon=_icon, url_path=_slug)
    _pages.append(_p)
    _by_slug[_slug] = _p

# Hidden native menu — our own pill bar below is the navigation UI.
nav = st.navigation(_pages, position="hidden")
current = nav.url_path or "home"     # default page has url_path ""

# Keep the pill selection synced when the page changes via URL
# (deep link, browser refresh, back/forward) rather than via the pill itself.
if st.session_state.get("_nav_last") != current:
    st.session_state["_nav"] = current
    st.session_state["_nav_last"] = current

# ── Persistent chrome ────────────────────────────────────────────────────────
topbar(head, df_all.attrs.get("api_status", {}))

_labels = {slug: f"{icon} {title}" for slug, icon, title, _, _ in _SPECS}
choice = st.radio(
    "Navigation", list(_labels), key="_nav", horizontal=True,
    label_visibility="collapsed", format_func=lambda s: _labels[s],
)

# Pill clicked → real page switch (URL updates, execution restarts on target).
if choice != current and choice in _by_slug:
    st.switch_page(_by_slug[choice])

# ── Active page ──────────────────────────────────────────────────────────────
nav.run()

st.markdown("---")
st.caption("app_modern.py — multipage dashboard (7 pages, real URLs) · "
           "`app.py` remains untouched and runnable")
