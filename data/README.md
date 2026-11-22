# Data

By default the project generates a **synthetic M5-style dataset** (`data.source: synthetic`) with known ground truth, so
everything runs offline in about a minute.

## Using the real M5 dataset
1. Download from https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
2. Place `calendar.csv`, `sell_prices.csv` and `sales_train_evaluation.csv` in `data/raw/`
3. Set `data.source: m5` in `configs/config.yaml` (tune `m5.n_series`), then `make all`

The loader keeps the top-N series by recent volume that were on sale for the whole window (balanced panel). M5 has no
promotion flag (promotions are derived from price drops) and no inventory/lead-time data (a simulated stock snapshot and
assumed lead times are used - replace with your ERP feed).

Corporacion Favorita is not implemented; add a loader producing the same long format as `data/m5_loader.py`.
