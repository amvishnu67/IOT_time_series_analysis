"""
Preprocessing utilities for the IoT hourly temperature series.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

DEFAULT_FREQ = "h"


def load_temperature_series(
    path: str | Path,
    datetime_col: str = "Datetime",
    value_col: str = "Hourly_Temp",
    freq: str = DEFAULT_FREQ,
) -> pd.Series:
    """
    Load the temperature series, set datetime index, enforce frequency.
    """
    df = pd.read_csv(path, parse_dates=[datetime_col])
    missing_cols = {col for col in (datetime_col, value_col) if col not in df.columns}
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    df = df[[datetime_col, value_col]].dropna()
    df = df.sort_values(datetime_col)
    df = df[~df[datetime_col].duplicated(keep="first")]
    y = df.set_index(datetime_col)[value_col].asfreq(freq)
    return y


def fill_gaps(y: pd.Series, method: str = "interpolate") -> pd.Series:
    """
    Fill missing timestamps/values without dropping rows.
    """
    if method == "ffill":
        filled = y.ffill().bfill()
    elif method == "interpolate":
        filled = y.interpolate(method="time").ffill().bfill()
    else:
        raise ValueError(f"Unknown gap-fill method: {method}")
    return filled


def clip_outliers(
    y: pd.Series, lower_quantile: float = 0.01, upper_quantile: float = 0.99
) -> pd.Series:
    """
    Winsorize extremes to keep frequency intact.
    """
    lower, upper = y.quantile([lower_quantile, upper_quantile])
    return y.clip(lower, upper)


def train_validation_split(y: pd.Series, val_steps: int = 24 * 7) -> Tuple[pd.Series, pd.Series]:
    """
    Chronological split preserving order (no shuffling).
    """
    if val_steps <= 0 or val_steps >= len(y):
        raise ValueError("val_steps must be positive and smaller than series length.")
    return y.iloc[:-val_steps], y.iloc[-val_steps:]


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred lengths must match.")
    err = y_true - y_pred
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err**2).mean()))
    return {"mae": mae, "rmse": rmse}


__all__ = [
    "load_temperature_series",
    "fill_gaps",
    "clip_outliers",
    "train_validation_split",
    "evaluate_forecast",
    "DEFAULT_FREQ",
]
