"""NAVTAC ETL: raw CSVs -> SQL -> pandas (clean, join, features) -> SQL + Parquet.

Usage:
  python src/etl_pipeline.py init     # create tables + load raw CSVs into the database
  python src/etl_pipeline.py run      # extract -> transform -> load -> analysis
Set DATABASE_URL, e.g. postgresql+psycopg2://navtac:navtac@localhost:5432/navtac

The transform stages (cleaning, joins, feature engineering, evaluation)
live in the reusable ``clean`` and ``features`` modules -- the same
implementations the test suite exercises.  This CLI wires extraction,
transformation and loading together.
"""
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

import clean as cleaning
import features as feat

DB_URL = os.getenv("DATABASE_URL", "sqlite:///navtac_test.db")

RAW, PARQUET = Path("data/raw"), Path("data/parquet")
TABLES = ["repositories", "dependencies", "commits", "dependency_states", "dependency_changes",
          "pull_requests", "migrations", "program_elements", "evidence", "candidates"]


# ------------------------------------------------------------------ INIT (CSV -> raw tables)
def init_db(engine):
    """Create tables to match the CSV data and load raw CSVs into the database.

    Uses the column names from the CSV files (e.g. repo_id, dependency_id,
    commit_id, state_id, change_id, pr_id, migration_id, element_id,
    evidence_id, candidate_id) instead of the simplified schema.sql.
    """
    con = engine.connect()
    try:
        for t in TABLES:
            # Drop table if exists (CASCADE to remove FK dependencies)
            con.execute(text(f"DROP TABLE IF EXISTS {t}"))
        con.commit()

        # Load data
        for t in TABLES:
            df = pd.read_csv(RAW / f"{t}.csv")
            # Parse date columns
            DATE_COLS = {
                "repositories": ["created_at"],
                "commits": ["committed_at"],
                "dependency_changes": ["changed_at"],
                "pull_requests": ["created_at", "merged_at"],
                "migrations": ["window_start", "window_end"],
                "candidates": ["prediction_point"],
            }
            for col in DATE_COLS.get(t, []):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            # Keep the CSV's native column names (repo_id,
            # dependency_id, state_id, change_id, ...): extract() and
            # the transform stages expect those names.
            # Create table from CSV columns
            cols = []
            for col in df.columns:
                dtype = _infer_sql_type(df[col])
                cols.append(f"    {col} {dtype}")
            # Add PRIMARY KEY
            pk_col = df.columns[0]
            cols.append(f"    PRIMARY KEY ({pk_col})")
            # Join columns with commas (comma between each, none after last)
            ddl = "CREATE TABLE " + t + " (" + ", ".join(cols) + ")"
            con.execute(text(ddl))
            # FK constraints: SQLite ALTER TABLE ADD FOREIGN KEY is not supported,
            # and they are not needed for parquet or analysis. Skipped.
            df.to_sql(t, engine, if_exists="append", index=False)
            print(f"loaded raw.{t:20s} {len(df):4d} rows")
        con.commit()
    finally:
        con.close()


def _infer_sql_type(series: pd.Series) -> str:
    """Infer a PostgreSQL column type from a pandas Series."""
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "NUMERIC(10, 3)"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP WITHOUT TIME ZONE"
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    return "TEXT"


def extract(engine) -> dict[str, pd.DataFrame]:
    return {t: pd.read_sql_table(t, engine) for t in TABLES}


# ------------------------------------------------------------------ LOAD
def load(engine, marts: dict[str, pd.DataFrame]):
    """Write each mart DataFrame to PostgreSQL and Parquet.

    Args:
        engine: SQLAlchemy engine connected to the database.
        marts: Dict mapping mart name to DataFrame.

    Writes:
        PostgreSQL: mart_<name> (if_exists='replace')
        Parquet: data/parquet/<name>.parquet (pyarrow)
    """
    PARQUET.mkdir(parents=True, exist_ok=True)
    for name, df in marts.items():
        df.to_sql(f"mart_{name}", engine, if_exists="replace", index=False)
        df.to_parquet(PARQUET / f"{name}.parquet", index=False, engine="pyarrow")
        print(f"loaded mart_{name:22s} {len(df):4d} rows  + {name}.parquet")


def main(cmd):
    engine = create_engine(DB_URL)
    if cmd == "init":
        return init_db(engine)
    raw = extract(engine)
    tables, log = cleaning.clean(raw)
    print("cleaning log:", log)
    changes = feat.build_change_mart(tables)
    mig_stats = feat.build_migration_stats(tables, changes)
    cand = feat.build_candidate_features(tables)
    metrics = feat.evaluate(cand)
    load(engine, {"dependency_changes_enriched": changes, "migration_stats": mig_stats,
                  "candidate_features": cand, "experiment_metrics": metrics})
    print("\n=== FINAL ANALYSIS ===")
    preview = mig_stats[["migration_id", "full_name", "package_name", "n_elements", "n_files",
                         "mean_strength", "duration_days"]].head(10)
    print(preview.to_string(index=False))
    print(f"\n(showing 10 of {len(mig_stats)} migrations)")
    print()
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run")
