"""
Live data helpers for the Customer Churn Dashboard.

The repository's IBM Telco dataset is a static snapshot, so this module does
not pretend to provide a true external real-time feed. Instead, it supports a
live dashboard workflow by re-reading the current source CSV and scoring the
latest rows with the saved XGBoost model whenever the user refreshes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from model.preprocessing import (
    ALL_CATEGORICAL_COLS,
    ALL_NUMERICAL_COLS,
    engineer_features,
    clean_data,
    load_raw_data,
)


RISK_THRESHOLDS = (
    (0.30, "LOW"),
    (0.60, "MEDIUM"),
    (0.80, "HIGH"),
)


def risk_level(probability: float) -> str:
    """Convert churn probability to the dashboard's four risk tiers."""
    for threshold, label in RISK_THRESHOLDS:
        if probability < threshold:
            return label
    return "CRITICAL"


def load_and_score_live_data(data_path: str | Path, model) -> tuple[pd.DataFrame, str]:
    """Reload the source CSV, run feature engineering and score all customers."""
    raw = load_raw_data(str(data_path))
    cleaned = clean_data(raw)
    featured = engineer_features(cleaned)

    feature_cols = ALL_NUMERICAL_COLS + ALL_CATEGORICAL_COLS
    X = featured[feature_cols]

    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    featured["churn_probability"] = np.round(probabilities, 4)
    featured["prediction"] = predictions
    featured["risk_level"] = [risk_level(float(p)) for p in probabilities]

    refreshed_at = datetime.now(timezone.utc).astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )

    return featured, refreshed_at
