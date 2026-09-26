# ml/ — forecasting for the modern dashboard. (NEW PACKAGE)
#
# Deliberately separate from src/: the original exploratory modules stay
# untouched. Everything in this package must obey the honesty rules spelled
# out in ml/ftp_forecast.py:
#   * no predicting a formula from its own ingredients (target leakage),
#   * time-ordered validation with a target-date embargo (no random folds),
#   * always compete against a real baseline (persistence) and report
#     negative skill honestly,
#   * surface coefficients so the coach sees what the model weighs.
#
# ml/pmc_projection.py is intentionally NOT machine learning: it continues
# the dashboard's exact EWMA under an assumed TSS plan (pure bookkeeping).
