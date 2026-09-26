# src/config.py — Global constants, zone definitions, colour palette, and training type metadata.

# ── Athlete ───────────────────────────────────────────────────────────────────
ATHLETE = {
    "weight_kg":    57.0,
    "ftp_current":  235,      # W — working FTP as of Jun 2026
    "ftp_target":   275,      # W — pre-accident FTP to recover
    "surgery_date": "2025-06-01",
    "dob_year":     1975,     # approximate for context
}

# ── FTP zones (% of FTP) ──────────────────────────────────────────────────────
ZONES = {
    "Z1": (0.00, 0.55, "Active Recovery",   "#8b949e"),
    "Z2": (0.55, 0.75, "Endurance",         "#58a6ff"),
    "Z3": (0.75, 0.87, "Tempo",             "#d29922"),
    "Z4": (0.87, 0.95, "Threshold",         "#f0883e"),
    "Z5": (0.95, 1.05, "VO2max",            "#f85149"),
    "Z6": (1.05, 1.20, "Anaerobic",         "#bc8cff"),
    "Z7": (1.20, 9.99, "Neuromuscular",     "#ff7b72"),
}

# ── Training types ────────────────────────────────────────────────────────────
MAIN_TYPES = [
    "END", "AEROBIC BASE", "FTP", "FATMAX", "VO2MAX",
    "SST", "TEMPO", "TORQUE", "Q-I INTERVALS", "BILLAT", "PIRAMIDAL",
]

TYPE_META = {
    "END":           {"category": "Volume",    "primary_zone": "Z2", "ftp_driver": False},
    "AEROBIC BASE":  {"category": "Volume",    "primary_zone": "Z2", "ftp_driver": False},
    "FTP":           {"category": "Intensity", "primary_zone": "Z4", "ftp_driver": True},
    "SST":           {"category": "Intensity", "primary_zone": "Z3-Z4", "ftp_driver": True},
    "TEMPO":         {"category": "Intensity", "primary_zone": "Z3", "ftp_driver": True},
    "PIRAMIDAL":     {"category": "Mixed",     "primary_zone": "Z2-Z4", "ftp_driver": True},
    "VO2MAX":        {"category": "Intensity", "primary_zone": "Z5", "ftp_driver": True},
    "BILLAT":        {"category": "Intensity", "primary_zone": "Z5", "ftp_driver": True},
    "Q-I INTERVALS": {"category": "Mixed",     "primary_zone": "Z4-Z5", "ftp_driver": True},
    "FATMAX":        {"category": "Metabolic", "primary_zone": "Z2", "ftp_driver": False},
    "TORQUE":        {"category": "Strength",  "primary_zone": "Z3-Z4", "ftp_driver": False},
}

# ── PMC constants ─────────────────────────────────────────────────────────────
CTL_SPAN = 42   # days — fitness
ATL_SPAN = 7    # days — fatigue

TSB_ZONES = {
    "overreached":   (-999, -30),
    "deep_block":    (-30,  -10),
    "build_phase":   (-10,    0),
    "neutral":       (  0,   10),
    "fresh":         ( 10,   25),
    "peak":          ( 25,  999),
}

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "bg":       "#0d1117",
    "panel":    "#161b22",
    "grid":     "#21262d",
    "text":     "#c9d1d9",
    "muted":    "#8b949e",
    "accent":   "#58a6ff",
    "green":    "#3fb950",
    "orange":   "#f0883e",
    "red":      "#f85149",
    "yellow":   "#d29922",
    "purple":   "#bc8cff",
}

TYPE_COLOURS = {
    "END":           "#58a6ff",
    "AEROBIC BASE":  "#3fb950",
    "FTP":           "#f85149",
    "SST":           "#f0883e",
    "TEMPO":         "#d29922",
    "PIRAMIDAL":     "#bc8cff",
    "VO2MAX":        "#ff7b72",
    "BILLAT":        "#79c0ff",
    "FATMAX":        "#56d364",
    "TORQUE":        "#ffa657",
    "Q-I INTERVALS": "#e3b341",
}

# ── Plot defaults ─────────────────────────────────────────────────────────────
PLOT_STYLE = {
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "text.color":        "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "grid.color":        "#21262d",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "figure.dpi":        130,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
}