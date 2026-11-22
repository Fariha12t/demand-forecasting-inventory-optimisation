"""PostgreSQL (or any SQLAlchemy URL) persistence for sales, forecasts and recommendations."""
from __future__ import annotations

import logging
import os

import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)
INDEXES = {"sales": ["id", "date"], "forecasts": ["id", "date"], "recommendations": ["id"]}


def get_engine(url: str | None = None):
    url = url or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("Set DATABASE_URL, e.g. postgresql+psycopg2://forecast:forecast@localhost:5432/forecast")
    return create_engine(url)


def write_table(df: pd.DataFrame, name: str, engine) -> None:
    df.to_sql(name, engine, if_exists="replace", index=False, chunksize=5000, method="multi")
    cols = INDEXES.get(name)
    if cols:
        with engine.begin() as con:
            con.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{name}_{'_'.join(cols)} ON {name} ({', '.join(cols)})"))


def read_table(name: str, engine) -> pd.DataFrame:
    return pd.read_sql_table(name, engine)


def sync_outputs(tables: dict[str, pd.DataFrame], url: str | None = None) -> None:
    engine = get_engine(url)
    for name, df in tables.items():
        write_table(df, name, engine)
        log.info("DB: wrote %d rows to %s", len(df), name)
