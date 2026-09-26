# core/context.py — shared app context for the multipage dashboard. (NEW FILE)
#
# The sidebar is rendered ONCE in the entrypoint (app_modern.py) so Streamlit
# keeps the filter widgets' state across page switches (the documented
# st.navigation pattern). Pages read the filtered frames via frames().

from datetime import datetime, timedelta
from types import SimpleNamespace

import streamlit as st

from core.data import load_data, load_wellness
from core.theme import MAIN_TYPES

RANGES = {
    "Last 30 days": 30,
    "Last 90 days": 90,
    "Last 6 months": 180,
    "Last 12 months": 365,
    "All time": 9999,
}
DEFAULT_RANGE = "Last 6 months"


def render_sidebar(df_all) -> None:
    """Persistent sidebar: controls, live sync status, data provenance."""
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        if st.button("🔄 Refresh data", key="refresh"):
            st.cache_data.clear()
            st.rerun()

        st.markdown("### 🔌 Data source")
        _sync = df_all.attrs.get("api_status", {})
        _base_rows = df_all.attrs.get("base_rows", 0)
        if _sync.get("ok"):
            st.success(f"Live · intervals.icu — {_sync.get('rows', 0)} activities "
                       f"synced (last {_sync.get('days', 60)} days)")
            if _sync.get("newest"):
                st.caption(f"Newest ride in feed: {str(_sync['newest'])[:16]}")
        elif _sync.get("error"):
            st.error("intervals.icu sync failed — showing local data only")
            st.caption(str(_sync.get("error"))[:180])
        else:
            st.warning("intervals.icu not connected")
        st.caption(f"Local base: {_base_rows} rows · merged total: {len(df_all)}")

        st.markdown("### Range")
        _names = list(RANGES)
        st.selectbox(
            "Time range", _names,
            index=_names.index(DEFAULT_RANGE), key="_range",
        )

        st.markdown("### Session filter")
        available_types = ["All"] + sorted(
            df_all["training_type"].dropna().unique().tolist()
        )
        st.multiselect(
            "Training types", available_types, default=["All"], key="_types",
            help="In app.py this filter was ignored — here it is wired to every "
                 "range-scoped chart and the weekly stats.",
        )

        st.markdown("---")
        st.caption(
            "Filters apply to charts & weekly stats. "
            "Fitness state and next-week advice always use full history."
        )
        st.markdown("---")
        st.markdown(f"**Updated** {datetime.now().strftime('%d %b %Y, %H:%M')}")
        st.markdown(f"**{len(df_all)} sessions** in base data")
        st.markdown(
            f"**{df_all['date'].min().date()} → {df_all['date'].max().date()}**"
        )


def frames(df_all=None) -> SimpleNamespace:
    """Apply sidebar range/type filters and return every frame pages need."""
    if df_all is None:
        df_all = load_data()
    wellness = load_wellness()

    range_name = st.session_state.get("_range", DEFAULT_RANGE)
    days = RANGES.get(range_name, 180)
    cutoff = datetime.now() - timedelta(days=days)
    types = st.session_state.get("_types", ["All"])

    df = df_all[df_all["date"] >= cutoff].copy()
    if "All" not in types:
        df = df[df["training_type"].isin(types)].copy()

    if "All" in types:
        df_all_f = df_all
    else:
        df_all_f = df_all[df_all["training_type"].isin(types)].copy()

    df_main_all = df_all_f[df_all_f["training_type"].isin(MAIN_TYPES)].copy()
    df_main = df[df["training_type"].isin(MAIN_TYPES)].copy()

    return SimpleNamespace(
        df_all=df_all,
        wellness=wellness,
        df=df,
        df_all_f=df_all_f,
        df_main=df_main,
        df_main_all=df_main_all,
        date_range=range_name,
        cutoff=cutoff,
        types=types,
    )
