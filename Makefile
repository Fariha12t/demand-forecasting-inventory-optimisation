.PHONY: install install-r data r-eda r-forecast train r-significance all test api dashboard docker-up docker-train

install:
	pip install -r requirements.txt && pip install -e .

install-r:
	Rscript r/00_install_packages.R

data:
	python -m demand_forecasting.pipeline prepare-data

r-eda:
	Rscript r/01_eda_seasonality.R

r-forecast:
	Rscript r/02_forecast_models.R

train:
	python -m demand_forecasting.pipeline train

r-significance:
	Rscript r/03_model_significance.R

# Full reproducible pipeline: data -> R EDA -> R forecasts -> Python models -> R significance tests -> report
all: data r-eda r-forecast train r-significance
	python -m demand_forecasting.pipeline report

test:
	pytest -q

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

docker-up:
	docker compose up --build

docker-train:
	docker compose --profile jobs run --rm pipeline
