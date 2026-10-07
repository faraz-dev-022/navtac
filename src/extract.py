#!/usr/bin/env python3
"""Extract all 10 tables from the database and profile them.

Functions:
    extract(engine) -> dict[str, DataFrame]
    profile_tables(dfs) -> None
"""

import pandas as pd


TABLES = [
    "repositories",
    "commits",
    "dependencies",
    "dependency_changes",
    "dependency_states",
    "pull_requests",
    "migrations",
    "program_elements",
    "evidence",
    "candidates",
]


def extract(engine) -> dict[str, pd.DataFrame]:
    """Read all 10 tables from the database into memory.

    Args:
        engine: SQLAlchemy engine connected to the database.

    Returns:
        Dictionary mapping table name to DataFrame.
    """
    # PostgreSQL stores tables in the navtac schema; SQLite has no schemas.
    schema = None if engine.dialect.name == "sqlite" else "navtac"
    dfs = {}
    for table in TABLES:
        dfs[table] = pd.read_sql_table(table, con=engine, schema=schema)
    return dfs


def profile_tables(dfs: dict[str, pd.DataFrame]) -> None:
    """Print per-table profile: shape, dtypes, null counts, duplicate PK values.

    Args:
        dfs: Dictionary mapping table name to DataFrame.
    """
    print("=" * 80)
    print("NAV TAC DATA PROFILE")
    print("=" * 80)
    for table_name, df in dfs.items():
        print(f"\n--- {table_name} ---")
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

        # Data types
        print("Dtypes:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")

        # Null counts
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        print(f"Total nulls: {total_nulls}")
        if total_nulls > 0:
            for col, count in null_counts.items():
                if count > 0:
                    pct = count / len(df) * 100
                    print(f"  {col}: {count} ({pct:.1f}%)")

        # Duplicate rows on primary key
        # Find the primary key column (usually the column ending with _id or id)
        pk_candidates = [c for c in df.columns if c in ("id", "repo_id", "commit_sha", "element_id", "evidence_id", "migration_id", "state_id", "dependency_id", "change_id", "pr_id")]
        if pk_candidates:
            pk_col = pk_candidates[0]
            dup_count = df.duplicated(subset=[pk_col]).sum()
            if dup_count > 0:
                print(f"DUPLICATE PRIMARY KEY ({pk_col}): {dup_count} rows")
        else:
            # Fall back to any column that looks like an ID
            id_cols = [c for c in df.columns if df[c].dtype in ["int64", "int32"] and c != "id" and "id" in c.lower()]
            if id_cols:
                pk_col = id_cols[0]
                dup_count = df.duplicated(subset=[pk_col]).sum()
                if dup_count > 0:
                    print(f"DUPLICATE PK CANDIDATE ({pk_col}): {dup_count} rows")

    print("\n" + "=" * 80)
    print("END PROFILE")
    print("=" * 80)
