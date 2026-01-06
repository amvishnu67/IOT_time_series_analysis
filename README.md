# IoT Temperature Forecasting

Forecast hourly temperature readings from an IoT sensor stream with a clean, reproducible workflow suitable for showcasing to recruiters.

## Project layout
- `data/Dataset.csv` – sample hourly temperature data (Datetime, Hourly_Temp).
- `notebooks/IoT_data_analysis_clean.ipynb` – narrated EDA + forecasting walkthrough.
- `src/preprocess.py` – data loading, frequency enforcement, gap filling, outlier clipping, time-series split, metrics.
- `src/train.py` – CLI entry point to train a SARIMA model and save plots/metrics.
- `reports/figures/` – generated figures (forecast plot).
- `tests/` – minimal regression tests for preprocessing.

## Quickstart
```bash
# Install dependencies (uses your conda env 'iot_ts' as requested)
conda run -n iot_ts python -m pip install -r requirements.txt

# Train and evaluate (forecast last 7 days by default)
conda run -n iot_ts python src/train.py --data-path data/Dataset.csv --output-dir reports/figures
```
Artifacts:
- `reports/figures/forecast.png` – forecast vs. actual with confidence intervals.
- Console prints MAE/RMSE; you can redirect to a log if desired.

## Modeling approach
1) Load and sort the hourly series, enforce an hourly index with `.asfreq("H")`.
2) Fill missing timestamps via time-based interpolation (configurable).
3) Clip extreme outliers by quantiles to preserve frequency.
4) Split chronologically (default: last 168 hours as validation).
5) Fit a SARIMA `(1,1,1)x(1,0,1,24)` using `statsmodels`.
6) Forecast the validation horizon; compute MAE/RMSE and plot with confidence intervals.

## Notebook usage
Open `notebooks/IoT_data_analysis_clean.ipynb` in Jupyter/VS Code and run all cells. The notebook mirrors the script while adding EDA (seasonality, stationarity checks) and inline commentary.

## Testing
```bash
conda run -n iot_ts python -m pytest
```
The tests assert that preprocessing keeps the hourly frequency and fills gaps without dropping data.

## Notes
- Replace `data/Dataset.csv` with your own IoT series (must include `Datetime` and `Hourly_Temp` columns).
- Tweak SARIMA orders or add `pmdarima` auto_arima exploration if you want automated order selection.
- Keep large/private datasets under `data/raw/` (ignored); commit small samples only.
