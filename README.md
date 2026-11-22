# AI-Powered Demand Forecasting & Inventory Optimisation (Python + R)

An end-to-end retail forecasting system: it forecasts product demand, compares statistical and machine-learning models
with leak-free time-based evaluation, and turns the forecasts into **inventory decisions** (safety stock, reorder point,
order quantity, stockout risk, holding cost). Served through a REST API and an interactive dashboard.

> **Data note:** results in this repo come from a built-in **synthetic M5-style dataset** (known ground truth: price
> elasticities, seasonality, injected anomalies) so the project runs offline in ~1 minute. Switch to the real
> [M5 dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) with `data.source: m5` (see `data/README.md`);
> numbers will then differ.

## What it covers

| Module | Implementation |
|---|---|
| Demand forecasting | Daily forecasts for 28 days per SKU-store, weekly roll-ups |
| Seasonality | **R**: MSTL (weekly + yearly), seasonal strength, weekday/month indices, stationarity tests |
| Price analysis | Poisson GLM elasticity (Python) + log-linear fixed-effects model (R); price/promo what-if |
| Inventory optimisation | Safety stock, reorder point, order-up-to level, order qty (pack/MOQ), stockout probability, EOQ, holding cost |
| Stockout prediction | `P(demand over lead time > inventory position)` -> Low / Medium / High |
| Anomaly detection | Robust weekday-aware z-score; events/promos are not flagged |
| Model comparison | 10 models, MAE / RMSE / WAPE / MASE, Friedman + Holm-adjusted Wilcoxon tests (R) |
| MLOps | MLflow tracking + model registry, drift detection (PSI + rolling WAPE), conditional retraining |
| Serving | FastAPI, Streamlit dashboard, PostgreSQL, Docker Compose |

**Models:** moving average, seasonal naive, seasonal average, Holt-Winters (statsmodels), **R** ETS / auto.arima /
ARIMAX with Fourier terms (dynamic harmonic regression with price & promo), Random Forest, LightGBM, XGBoost, optional LSTM.

## Results (synthetic data, 60 SKU-stores, 28-day test window)

| Model | MAE | RMSE | WAPE | MASE | WAPE (weekly) |
|---|---|---|---|---|---|
| **LightGBM** | **4.03** | **6.42** | **37.7%** | **0.74** | **17.6%** |
| XGBoost | 4.13 | 6.48 | 38.6% | 0.76 | 18.1% |
| R ARIMAX + Fourier | 4.33 | 6.80 | 40.4% | 0.78 | 19.2% |
| Random Forest | 4.35 | 6.63 | 40.6% | 0.80 | 20.7% |
| Holt-Winters | 4.59 | 6.84 | 42.9% | 0.83 | 22.6% |
| R ETS | 4.64 | 6.94 | 43.4% | 0.84 | 23.6% |
| Moving average (28d) | 4.66 | 6.95 | 43.6% | 0.85 | 20.9% |
| R auto.arima | 4.72 | 7.04 | 44.1% | 0.85 | 21.6% |
| Seasonal average (4w) | 4.81 | 7.27 | 44.9% | 0.87 | 20.9% |
| Seasonal naive | 7.00 | 10.66 | 65.4% | 1.26 | 30.8% |

LightGBM is best, and R's Friedman test + Holm-adjusted Wilcoxon tests find it significantly better than every other model
(p<0.01). Daily WAPE is high because per-SKU daily demand is a noisy count (Poisson), and the test window includes
Christmas; aggregating to weekly halves the error. Full tables: [`docs/model_comparison_report.md`](docs/model_comparison_report.md).

Other findings (see report): price elasticities recovered within confidence intervals of the simulator's true values;
anomaly detector precision ~71% / recall ~29% (spikes 50%, zero-drops ~0% - drops on low-volume series are indistinguishable
from noise); inventory back-test shows lowest *forecast error* does not automatically mean lowest *inventory cost* in one
28-day window (bias and calibration matter) - treat those differences as indicative.

## Quick start

```bash
pip install -r requirements.txt && pip install -e .
make data train          # Python only (~70 s): data, models, evaluation, inventory recs, artefacts
# full pipeline incl. R (R >= 4.2; `make install-r` once; R forecasts take ~8 min on 1 core, use R_CORES=n):
make install-r && make all
make test                # 25 tests
make api                 # http://localhost:8000/docs
make dashboard           # http://localhost:8501
```

`make all` = data -> R EDA -> R forecasts -> Python models (merges R forecasts) -> R significance tests -> report.
Docker: `docker compose up --build` (API, dashboard, MLflow, Postgres); `docker compose --profile jobs run --rm pipeline`.

## Project layout

```
configs/config.yaml        all settings (data source, split, models, inventory policy, monitoring)
src/demand_forecasting/    data/, features.py, models/, evaluation.py, inventory.py, anomaly.py, price.py,
                           monitoring.py, serving.py, db.py, pipeline.py, report.py
r/                         01_eda_seasonality.R  02_forecast_models.R  03_model_significance.R
api/main.py                FastAPI: /forecast /recommend /whatif /recommendations /model-info
dashboard/app.py           Streamlit (tabs, light theme)
tests/                     metrics, leakage, inventory maths, anomaly/drift, pipeline, API, DB
docs/                      model_comparison_report.md + sample results and R plots
```

## Method notes

- **No leakage:** strictly chronological train / validation / test. Lag and rolling features use only past data (unit-tested).
  Trees are early-stopped on the last 28 days *of the training period*. Test-fold models are refit on train+validation.
- **Multi-step:** tree models forecast recursively (predictions become future lags). Planned price, promotion and calendar
  events are treated as known inputs.
- **Inventory:** continuous-review (s, S). `SS = z * sigma_d * sqrt(L)`, `ROP = demand_L + SS`,
  `S = demand_(L+R) + z * sigma_d * sqrt(L+R)`; order `S - position` when position <= ROP, rounded up to pack size.
  `sigma_d` is the daily forecast-error std from the validation window.
- **Stock levels are simulated** (`data/processed/inventory_snapshot.csv`) because M5 has no inventory data; plug in an ERP/WMS feed.
- **Drift & retraining:** `python -m demand_forecasting.pipeline retrain` computes PSI on inputs, rolling WAPE vs validation
  and model age; it retrains only when a threshold is breached (`--force` to override). Schedule via cron / `retrain.yml`.
- **LSTM** is optional and off by default (`models.lstm.enabled`, `pip install -r requirements-lstm.txt`). It was not run
  in the environment this repo was built in, so treat it as untested.

## API example

```bash
curl -X POST localhost:8000/recommend -H 'content-type: application/json' \
  -d '{"series_id":"ITEM_001_CA_1","current_stock":20}'
```
returns predicted weekly demand, safety stock, reorder point, order quantity, stockout risk and holding cost.

## Limitations

Synthetic data results are illustrative; Favorita is not implemented; Docker files and the GitHub workflows were written but
not executed in the build environment; the LSTM path is untested; R forecasts use fixed (not tuned) Fourier orders.
