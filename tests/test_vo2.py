import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import pandas as pd

from core.interval_data import read_intervals
from ml.vo2_estimate import run_vo2_estimate, vo2_l_per_min

iv = read_intervals((pd.Timestamp.now().normalize()
                     - pd.Timedelta(days=370)).isoformat())
print("rows:", len(iv))

res = run_vo2_estimate(iv, 57.0, ge_pct=21.0, age=48)
if not res["ok"]:
    print("NOT OK:", res["reason"])
    sys.exit(1)

print("PVO2 basis:", res["pvo2_basis"], "->", round(res["pvo2_w"]), "W",
      round(res["wkg_pvo2"], 2), "W/kg")
print("VO2:", round(res["vo2_l"], 2), "L/min =",
      round(res["vo2_mlkg"]), "mL/kg/min |", res["anchor"])
print("confidence:", res["conf"], "-", res["conf_why"])
print("alltime_hr:", res["alltime_hr"], "age_pred:", res["age_pred"])
print("\ncurve:")
print(res["curve"][["label", "watts", "date", "hr_max"]].to_string(index=False))
print("\nsensitivity:")
print(res["sens"].to_string(index=False))
print("\ndisagreement:")
print(res["dur_table"].to_string(index=False))
print("\nevidence:")
print(res["evidence"].head(6).to_string(index=False))
print("\nformula:", res["formula"])

# edge cases: slider at band ends + no-HR weight
r18 = run_vo2_estimate(iv, 57.0, ge_pct=18.0, age=48)
r24 = run_vo2_estimate(iv, 57.0, ge_pct=24.0, age=48)
print("band ends ok:", r18["vo2_mlkg"], r24["vo2_mlkg"],
      "| sens rows:", len(r18["sens"]), len(r24["sens"]))

from views import intervals_view
print("view import OK")
