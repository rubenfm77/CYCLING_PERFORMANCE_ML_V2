# app_modern.py — Cycling Performance Dashboard, modern edition (NEW FILE)
# =====================================================================
# Run with:  python -m streamlit run app_modern.py
#
# app.py is untouched and still runs with:  python -m streamlit run app.py
#
# This file is the ROUTER of a real multipage app (st.navigation / st.Page):
#   - persistent chrome on every page: top bar (brand + live status + state/
#     KPI chips), the navigation pill bar, and the sidebar filters;
#   - eight pages with real URLs (/fitness, /forecast, /intervals, ...):
#     Overview · Fitness · Forecast · Intervals · Training · Evolution ·
#     Trends · Sessions.
# Design: keep the GitHub-dark + blue identity of app.py, presented with
# modern components (gradient hero, accent-strip cards, icon section headers).

import importlib
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

# ── Page modules, imported ONE AT A TIME ──────────────────────────────────────
# This is not a style choice. Every page module used to be pulled in by a single
# `from views import (a, b, c, ...)` statement, and this file imports them all
# while BUILDING THE PAGE REGISTRY — hundreds of lines before st.navigation().
#
# A module-level ImportError in any one view is therefore an APP-WIDE crash. Not
# the broken page: all eight. A missing constant in one leaf module once took the
# whole dashboard down while the seven healthy pages could not render at all, and
# nothing in the UI could say why.
#
# Importing each module separately makes a page that cannot load visible and
# isolated: the app still runs, the other pages still work, and the broken one
# reports the exception instead of silently drawing nothing.
#
# `except Exception` rather than `except ImportError` is deliberate. The entire
# point is not to know in advance what a page module might raise on import; any
# of it must not be allowed to reach the user as a blank dashboard.
_VIEW_NAMES = ("evolution", "fitness", "forecast", "intervals_view", "overview",
               "sessions", "training", "trends")

_views, _view_errors = {}, {}
for _name in _VIEW_NAMES:
    try:
        _views[_name] = importlib.import_module(f"views.{_name}")
    except Exception as _exc:                                    # noqa: BLE001
        _view_errors[_name] = _exc


def _unavailable(name, exc):
    """Stand-in render() for a page module that failed to import.

    Names the exception TYPE always. Shows the message only for ImportError,
    which is a name-or-path resolution failure and cannot carry athlete data;
    any other exception's message is withheld, because nothing guarantees it is
    safe to put on screen. The full traceback is in Streamlit's logs under
    'Manage app'.
    """
    def _render(head, ctx):
        st.error(f"The **{name}** page could not be loaded.", icon="🚨")
        st.caption(f"`{type(exc).__name__}` raised while importing "
                   f"`views/{name}.py`. The rest of the dashboard is unaffected.")
        if isinstance(exc, ImportError):
            st.code(str(exc))
        else:
            st.caption("Full traceback: Manage app → Logs.")
    return _render


def _render_of(name):
    """The page's real render(), or an honest stand-in if it failed to import."""
    if name in _views:
        return _views[name].render
    return _unavailable(name, _view_errors[name])

inject_css()

# ── Data + shared sidebar (widgets live here so state survives page changes) ─
df_all = load_data()
render_sidebar(df_all)
ctx = frames(df_all)
head = headline(ctx)

# ── Page registry ────────────────────────────────────────────────────────────
_SPECS = [
    # slug, icon, title, callable, is_default
    ("home",     "🏠", "Overview", partial(_render_of("overview"), head, ctx), True),
    ("fitness",  "📈", "Fitness",  partial(_render_of("fitness"), head, ctx),  False),
    ("forecast", "🔮", "Forecast", partial(_render_of("forecast"), head, ctx), False),
    ("intervals", "🔬", "Intervals", partial(_render_of("intervals_view"), head, ctx),
     False),
    ("training", "🏋️", "Training", partial(_render_of("training"), head, ctx), False),
    ("trends",   "📊", "Trends",   partial(_render_of("trends"), head, ctx),   False),
    ("sessions", "📅", "Sessions", partial(_render_of("sessions"), head, ctx), False),
    ("evolution", "\U0001F4C8", "Evolution",
     partial(_render_of("evolution"), head, ctx), False),
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

# A page that failed to import is a fault, not a detail: say so on every page
# rather than letting the user discover a missing page by looking for it.
if _view_errors:
    st.warning(
        f"**{len(_view_errors)} of {len(_SPECS)} pages could not be loaded:** "
        + ", ".join(sorted(_view_errors))
        + ". Everything else works normally. The reason is in the logs under "
          "Manage app.",
        icon="⚠️",
    )

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
st.caption("app_modern.py — multipage dashboard (8 pages, real URLs) · "
           "`app.py` remains untouched and runnable")
