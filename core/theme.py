# core/theme.py — SINGLE SOURCE OF TRUTH for styling constants.
# New file: app.py keeps its own local C dict untouched. Anything that needs
# colors/heights in the modern dashboard imports from here, so the 240W/235W
# and 56.8kg/57.0kg mismatches from app.py cannot happen again.

from src.config import ATHLETE, MAIN_TYPES, ZONES, TYPE_COLOURS

# ── Palette ──────────────────────────────────────────────────────────────────
C = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "panel2":  "#1c2128",
    "grid":    "#21262d",
    "border":  "#30363d",
    "text":    "#c9d1d9",
    "muted":   "#8b949e",
    "accent":  "#58a6ff",
    "green":   "#3fb950",
    "orange":  "#f0883e",
    "red":     "#f85149",
    "yellow":  "#d29922",
    "purple":  "#bc8cff",
    "teal":    "#39c5cf",
}

# ── Athlete constants — reconciled, ONE value each ───────────────────────────
# app.py disagrees with itself (FTP 240 in power-curve chart, 235 in FTP
# progression; weight 57.0 vs 56.8). athlete-profile.md says FTP 240 W
# validated June 2026, weight 57 kg. Those are the values used here.
FTP_CURRENT = 240            # W, validated June 2026 (athlete-profile.md)
FTP_TARGET  = ATHLETE["ftp_target"]   # 275 W, pre-injury reference
WEIGHT_KG   = ATHLETE["weight_kg"]    # 57.0
SURGERY     = ATHLETE["surgery_date"] # "2025-06-01"

# ── Norwegian-method thresholds ─────────────────────────────────────────────
IF_Z2_MAX       = 0.75   # below = pure Z2
IF_THRESHOLD    = 0.85   # at/above = true threshold ("quality")
IF_VO2          = 0.95   # at/above = VO2max
HOT_TEMP_C      = 28.0   # heat flag for efficiency chart

# ── Chart geometry — original app.py sizes (350–420px panels, 750px PMC) ─────
H_CARD   = 340   # small single-purpose chart
H_STD    = 420   # default panel chart
H_PAIR   = 440   # side-by-side pair
H_HERO   = 780   # multi-row stacked figure

# ── State colours (TSB-derived fatigue state) ────────────────────────────────
STATE_COLORS = {
    "Undertrained":      C["muted"],
    "Overreached":       C["red"],
    "Deep Block":        C["orange"],
    "Build Phase":       C["yellow"],
    "Neutral":           C["accent"],
    "Fresh":             C["green"],
    "Peak/Detrain Risk": C["purple"],
}

STATE_GUIDE = {
    "Undertrained":      ("⚠️", "Build baseline volume — CTL too low"),
    "Overreached":       ("🚨", "Rest immediately — TSB below -30"),
    "Deep Block":        ("💪", "Hard training block — monitor recovery closely"),
    "Build Phase":       ("✅", "Most productive training zone — keep pushing"),
    "Neutral":           ("⚖️", "Balanced load — maintain or start taper"),
    "Fresh":             ("🟢", "Race-ready window — quality sessions now"),
    "Peak/Detrain Risk": ("⚡", "Peak form — race or start next block"),
}

FTP_DRIVERS = ["FTP", "SST", "TEMPO", "PIRAMIDAL", "VO2MAX", "BILLAT", "Q-I INTERVALS"]

# Training-type colours come from src.config.TYPE_COLOURS (already canonical);
# exposed here so views never hardcode hex values.
TYPE_COLORS = dict(TYPE_COLOURS)

# ── Plotly house style ──────────────────────────────────────────────────────
CHART_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
}


def style_figure(fig, title: str | None = None, height: int = H_STD,
                 showlegend: bool | None = None):
    """Apply the one true chart style. Call after adding traces."""
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=C["panel"],
        font=dict(family="Inter, Segoe UI, Helvetica, sans-serif",
                  color=C["text"], size=12),
        height=height,
        margin=dict(l=52, r=24, t=74 if title else 48, b=46),
        hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["border"],
                        font=dict(color=C["text"], size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"],
                    font=dict(color=C["muted"], size=11)),
        xaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"],
                   tickfont=dict(color=C["muted"], size=11)),
        yaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"],
                   tickfont=dict(color=C["muted"], size=11)),
    )
    if title:
        layout["title"] = dict(
            text=title, x=0.005, xanchor="left",
            font=dict(color=C["text"], size=14, family="Inter, Segoe UI, sans-serif"),
            pad=dict(t=10, l=2),
        )
    if showlegend is not None:
        layout["showlegend"] = showlegend
    fig.update_layout(**layout)
    fig.update_xaxes(showline=False)
    fig.update_yaxes(showline=False)
    return fig


__all__ = [
    "C", "STATE_COLORS", "STATE_GUIDE", "TYPE_COLORS", "FTP_DRIVERS",
    "FTP_CURRENT", "FTP_TARGET", "WEIGHT_KG", "SURGERY",
    "IF_Z2_MAX", "IF_THRESHOLD", "IF_VO2", "HOT_TEMP_C",
    "H_CARD", "H_STD", "H_PAIR", "H_HERO",
    "CHART_CONFIG", "style_figure", "ZONES", "MAIN_TYPES",
]
