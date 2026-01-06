"""
CLI entry point: load data, clean, train SARIMA, evaluate, and save a forecast plot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Allow running as a script (`python src/train.py`) without installation
SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from preprocess import (  # noqa: E402
    clip_outliers,
    evaluate_forecast,
    fill_gaps,
    load_temperature_series,
    train_validation_split,
)


def fit_model(train: pd.Series, order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int]):
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def plot_forecast(train: pd.Series, val: pd.Series, fc_mean: pd.Series, conf: pd.DataFrame, output_path: Path) -> None:
    context = pd.concat([train.iloc[-24 * 7 :], val])

    plt.figure(figsize=(12, 4))
    plt.plot(context, label="Observed", color="#1f77b4")
    plt.plot(fc_mean.index, fc_mean.values, label="Forecast", color="#d62728")
    plt.fill_between(conf.index, conf.iloc[:, 0], conf.iloc[:, 1], color="#ff9896", alpha=0.3, label="95% CI")
    plt.title("Hourly Temperature Forecast (Validation Horizon)")
    plt.xlabel("Datetime")
    plt.ylabel("Hourly Temperature")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SARIMA on hourly temperature series.")
    parser.add_argument("--data-path", default="data/Dataset.csv", help="CSV with Datetime and Hourly_Temp columns.")
    parser.add_argument("--output-dir", default="reports/figures", help="Directory to save forecast plot.")
    parser.add_argument("--val-steps", type=int, default=24 * 7, help="Validation horizon in hours.")
    parser.add_argument("--gap-fill", choices=["interpolate", "ffill"], default="interpolate", help="Gap filling method.")
    parser.add_argument("--lower-quantile", type=float, default=0.01, help="Lower quantile for clipping.")
    parser.add_argument("--upper-quantile", type=float, default=0.99, help="Upper quantile for clipping.")
    parser.add_argument("--order", nargs=3, type=int, metavar=("p", "d", "q"), default=(1, 1, 1), help="ARIMA (p,d,q).")
    parser.add_argument(
        "--seasonal-order",
        nargs=4,
        type=int,
        metavar=("P", "D", "Q", "s"),
        default=(1, 0, 1, 24),
        help="Seasonal (P,D,Q,s).",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "forecast.png"

    y = load_temperature_series(data_path)
    y = fill_gaps(y, method=args.gap_fill)
    y = clip_outliers(y, lower_quantile=args.lower_quantile, upper_quantile=args.upper_quantile)

    train, val = train_validation_split(y, val_steps=args.val_steps)

    fit = fit_model(train, order=tuple(args.order), seasonal_order=tuple(args.seasonal_order))
    fc = fit.get_forecast(steps=args.val_steps)
    fc_mean = fc.predicted_mean
    conf = fc.conf_int(alpha=0.05)

    metrics = evaluate_forecast(val, fc_mean)
    print(f"Validation MAE: {metrics['mae']:.3f}")
    print(f"Validation RMSE: {metrics['rmse']:.3f}")

    plot_forecast(train, val, fc_mean, conf, plot_path)
    print(f"Saved forecast plot to {plot_path}")


if __name__ == "__main__":
    main()
