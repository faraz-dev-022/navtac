#!/usr/bin/env python3
"""Verify the extract() and profile_tables() functions.

Connects to the database using DATABASE_URL, extracts all 10 tables,
and prints a per-table profile including shape, dtypes, null counts,
and duplicate PK values.

Flags data-quality problems found during profiling.
"""

import os
import sys

# Add the project root to the path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine
from extract import extract, profile_tables


def get_engine():
    """Create a database engine using the local Unix socket (no password needed).

    Returns:
        SQLAlchemy engine connected via Unix socket.
    """
    # Default: use local PostgreSQL via Unix socket
    # (Works without password when PostgreSQL is running locally)
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg2://farraz@localhost:5432/retail_project"
    )
    # For Unix socket connections, pass host via connect_args
    engine = create_engine(
        database_url,
        connect_args={"host": "/var/run/postgresql"}
    )
    return engine


def main():
    engine = get_engine()

    print(f"Connecting to: {engine.url}")
    try:
        with engine.connect():
            pass
    except Exception as exc:
        print(f"Failed to connect: {exc}")
        print("\nMake sure PostgreSQL is running on this machine.")
        print("Note: This script uses the local Unix socket (no password needed).")
        engine.dispose()
        return 1

    # Extract all tables
    print("\nExtracting all 10 tables...")
    dfs = extract(engine)

    # Profile
    profile_tables(dfs)

    # Quick sanity check on null counts
    print("\n" + "=" * 80)
    print("DATA-QUALITY FLAGS")
    print("=" * 80)

    issues = 0
    for table_name, df in dfs.items():
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                pct = count / len(df) * 100
                print(f"  [{table_name}] {col}: {count} NULLs ({pct:.1f}%)")
                issues += 1

        # Check for duplicate PKs
        pk_col = None
        for c in df.columns:
            if c in ("id", "repo_id", "commit_sha", "element_id",
                     "evidence_id", "migration_id", "state_id",
                     "dependency_id", "change_id", "pr_id"):
                pk_col = c
                break
        if pk_col:
            dupes = df.duplicated(subset=[pk_col]).sum()
            if dupes > 0:
                print(f"  [{table_name}] DUPLICATE PRIMARY KEY ({pk_col}): {dupes} rows")
                issues += 1

        # Check for mixed date formats in observed_at
        if table_name == "dependency_states" and "observed_at" in df.columns:
            for val in df["observed_at"].dropna().unique():
                # Check if the value looks like a date but has inconsistent format
                s = str(val)
                if ('/' in s) or ('T' in s) or ('Z' in s) or ('+' in s) or (' ' in s and ':' in s and len(s) > 10):
                    # Found a date format that differs from standard ISO
                    print(f"  [{table_name}] MIXED DATE FORMAT: '{s}'")
                    issues += 1
                    break

    if issues == 0:
        print("  No data-quality issues detected.")

    engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
