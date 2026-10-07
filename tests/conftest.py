"""Shared fixtures for the NAVTAC test suite (temporary SQLite, no Docker)."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import etl_pipeline as etl  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
TABLES = [
    "repositories",
    "commits",
    "dependencies",
    "dependency_states",
    "dependency_changes",
    "pull_requests",
    "migrations",
    "program_elements",
    "evidence",
    "candidates",
]


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    """Fresh SQLite engine; DATABASE_URL is set so callers need no Docker."""
    db = tmp_path / "test.db"
    url = f"sqlite:///{db}"
    monkeypatch.setenv("DATABASE_URL", url)
    etl.DB_URL = url
    eng = create_engine(url)
    yield eng
    eng.dispose()


@pytest.fixture()
def dfs(engine):
    """Load raw CSVs into SQLite and return them as extracted DataFrames."""
    for table in TABLES:
        df = pd.read_csv(RAW_DIR / f"{table}.csv")
        df.to_sql(table, con=engine, if_exists="replace", index=False)
    return etl.extract(engine)
