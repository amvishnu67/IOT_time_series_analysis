import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from preprocess import clip_outliers, fill_gaps, train_validation_split


def test_fill_keeps_frequency_and_values():
    idx = pd.date_range("2022-01-01", periods=5, freq="h")
    y = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0], index=idx)

    filled = fill_gaps(y, method="interpolate")

    assert filled.isna().sum() == 0
    freq = filled.index.freqstr or filled.index.inferred_freq
    assert freq.lower() == "h"
    assert np.isclose(filled.loc["2022-01-01 01:00"], 2.0)


def test_clip_outliers_preserves_length():
    idx = pd.date_range("2022-01-01", periods=10, freq="h")
    y = pd.Series([1] * 8 + [1000, 2000], index=idx)

    clipped = clip_outliers(y, lower_quantile=0.1, upper_quantile=0.9)

    assert len(clipped) == len(y)
    assert clipped.max() < 2000


def test_train_validation_split_order():
    idx = pd.date_range("2022-01-01", periods=10, freq="h")
    y = pd.Series(range(10), index=idx)

    train, val = train_validation_split(y, val_steps=3)

    assert len(train) == 7 and len(val) == 3
    assert train.index[-1] < val.index[0]
