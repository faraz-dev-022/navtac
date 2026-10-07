#!/usr/bin/env python3
"""Verify the clean() pipeline.

Extracts the data, cleans it, prints the log, and asserts no NaT in dates
and no duplicate natural keys.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine
from extract import extract
from clean import clean, parse_mixed_dates, verify


def get_engine():
    """Create a database engine using the local Unix socket (no password needed)."""
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg2://farraz@localhost:5432/retail_project"
    )
    return create_engine(
        database_url,
        connect_args={"host": "/var/run/postgresql"}
    )


def main():
    engine = get_engine()
    print(f"Connecting to: {engine.url}")

    # Extract raw tables
    print("\nExtracting raw tables...")
    raw_tables = extract(engine)
    print(f"  Extracted {len(raw_tables)} tables")

    # Clean
    print("\nCleaning tables...")
    cleaned, log = clean(raw_tables)
    print(f"  Cleaned {len(cleaned)} tables")

    # Print the log
    print("\n" + "=" * 80)
    print("CLEANING LOG")
    print("=" * 80)
    if log:
        for rule, count in sorted(log.items()):
            print(f"  {rule}: {count}")
    else:
        print("  (no rules fired)")

    # Verification
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    verify(cleaned, log)

    # Additional assertions
    print("\n" + "=" * 80)
    print("ASSERTIONS")
    print("=" * 80)

    failures = []

    # 1) No NaT in dates
    for name, df in cleaned.items():
        if "observed_at" in df.columns:
            nat_count = df["observed_at"].isna().sum()
            if nat_count > 0:
                failures.append(f"{name}: {nat_count} NaT in observed_at")

    # 2) No duplicate natural keys
    for name, df in cleaned.items():
        if name == "dependency_states":
            dupes = df.duplicated(subset=["repo_id", "dependency_id", "commit_sha", "version_spec"]).sum()
            if dupes > 0:
                failures.append(f"{name}: {dupes} duplicate natural keys")
        elif name == "program_elements":
            dupes = df.duplicated(subset=["migration_id", "file_path", "element_name", "api_symbol"]).sum()
            if dupes > 0:
                failures.append(f"{name}: {dupes} duplicate natural keys")

    # 3) No NULL in resolved_version (should have been recovered)
    if "dependency_states" in cleaned:
        ds = cleaned["dependency_states"]
        null_rv = ds["resolved_version"].isna().sum()
        if null_rv > 0:
            failures.append(f"dependency_states: {null_rv} NULL resolved_version")

    # 4) Language stripped/lowercase
    if "repositories" in cleaned:
        langs = cleaned["repositories"]["language"]
        bad = langs[~langs.apply(lambda x: isinstance(x, str) and x == x.lower() and x == x.strip())]
        if len(bad) > 0:
            failures.append(f"repositories: {len(bad)} languages not stripped/lowercased")

    # 5) observed_label is bool
    if "candidates" in cleaned:
        vals = cleaned["candidates"]["observed_label"]
        non_bool = vals[~vals.apply(lambda x: isinstance(x, bool))]
        if len(non_bool) > 0:
            failures.append(f"candidates: {len(non_bool)} non-bool observed_label")

    if failures:
        print("  FAILURES:")
        for f in failures:
            print(f"    - {f}")
        engine.dispose()
        return 1
    else:
        print("  All assertions passed: PASS")
        engine.dispose()
        return 0


if __name__ == "__main__":
    sys.exit(main())
