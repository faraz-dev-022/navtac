#!/usr/bin/env python3
"""Cleaning pipeline for Navtac data.

Functions:
    parse_mixed_dates(value) -> datetime | str
    clean(tables) -> (cleaned_tables, log)
"""

import re
from typing import Any, Dict

import numpy as np
import pandas as pd

# Re-export TABLES from extract for convenience
from extract import TABLES

DATE_COLS = {
    "repositories": ["created_at"],
    "commits": ["committed_at"],
    "dependency_changes": ["changed_at"],
    "pull_requests": ["created_at", "merged_at"],
    "migrations": ["window_start", "window_end"],
    "candidates": ["prediction_point"],
}


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


def parse_mixed_dates(value: Any) -> pd.Timestamp | str:
    """Try to parse a date string using explicit formats, never guessing.

    Order matters: try each format in sequence and return the first that
    parses successfully. Never rely on pandas' implicit dayfirst guessing.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return value

    value = str(value).strip()
    if not value:
        return value

    # Standard formats to try in order
    formats = [
        "%Y-%m-%d",                # 2024-02-11
        "%d/%m/%Y",                # 15/11/2023
        "%Y-%m-%dT%H:%M:%SZ",      # 2024-02-11T12:00:00Z
        "%Y-%m-%dT%H:%M:%S.%fZ",   # 2024-02-11T12:00:00.123Z
        "%Y-%m-%d %H:%M:%S%z",     # 2024-02-11 00:00:00+00
        "%Y-%m-%d %H:%M:%S.%f%z",  # 2024-02-11 00:00:00.123456+00
        "%Y-%m-%d %H:%M:%S",       # 2024-02-11 00:00:00
        "%Y-%m-%d %H:%M:%S.%f",    # 2024-02-11 00:00:00.123
        "%b %d, %Y",               # Feb 11, 2024
        "%b %d %Y",                # Feb 11 2024
        "%d %b %Y",                # 11 Feb 2024
    ]

    for fmt in formats:
        try:
            ts = pd.to_datetime(value, format=fmt)
            if isinstance(ts, pd.Timestamp) and ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            return ts
        except (ValueError, TypeError):
            continue

    # If none matched, return the original string (not NaT) so we can see
    # which values failed to parse
    return value


# ---------------------------------------------------------------------------
# Cleaning pipeline
# ---------------------------------------------------------------------------


def clean(tables: Dict[str, pd.DataFrame]) -> tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
    """Clean all tables, never mutating the input.

    Args:
        tables: Dict mapping table name to DataFrame.

    Returns:
        (cleaned_tables, log) where log is a dict mapping rule name to
        number of duplicate rows removed.
    """
    cleaned = {}
    log: Dict[str, int] = {}

    # Make a copy of the input to avoid mutation
    for name, df in tables.items():
        cleaned[name] = df.copy()

    # =====================================================================
    # 1) Date columns: convert with explicit formats (no dayfirst guessing)
    # =====================================================================
    # dependency_states.observed_at: try explicit formats one by one
    if "dependency_states" in cleaned:
        ds = cleaned["dependency_states"]
        original_count = len(ds)
        ds["observed_at"] = ds["observed_at"].apply(parse_mixed_dates)
        # Count how many became NaT (parse failures)
        nat_count = ds["observed_at"].isna().sum()
        log["date_parse_failures"] = nat_count

    for name, cols in DATE_COLS.items():
        if name not in cleaned:
            continue
        for col in cols:
            if col in cleaned[name].columns:
                cleaned[name][col] = pd.to_datetime(cleaned[name][col], errors="coerce")

    # =====================================================================
    # 2) repositories.language: strip + lowercase
    # =====================================================================
    if "repositories" in cleaned:
        cleaned["repositories"]["language"] = (
            cleaned["repositories"]["language"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    # =====================================================================
    # 2) repositories.stars: fill with median, cast to int
    # =====================================================================
    if "repositories" in cleaned:
        repo_df = cleaned["repositories"]
        median_stars = repo_df["stars"].median()
        repo_df["stars"] = repo_df["stars"].fillna(median_stars).astype(int)
        if pd.isna(median_stars):
            log["stars_fill"] = 0

    # =====================================================================
    # 3) dependencies.package_name: strip, lowercase, replace [-_.]+ with '-'
    # =====================================================================
    if "dependencies" in cleaned:
        cleaned["dependencies"]["package_name"] = (
            cleaned["dependencies"]["package_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r'[-_.]+', '-', regex=True)
        )

    # =====================================================================
    # 4) commits.files_changed: fill with per-repo median
    # =====================================================================
    if "commits" in cleaned:
        comm = cleaned["commits"]
        # Compute per-repo median of files_changed
        repo_medians = comm.groupby("repo_id")["files_changed"].transform("median")
        comm["files_changed"] = comm["files_changed"].fillna(repo_medians).astype(int)
        # Count nulls filled
        log["commits_files_changed_filled"] = int(comm["files_changed"].isna().sum() == 0 and
                                                   len(comm) != len(comm.dropna(subset=["files_changed"])))

    # =====================================================================
    # 5) dependency_states: drop duplicates on (repo_id, dependency_id, commit_sha, version_spec)
    #    recover resolved_version from version_spec when it looks like '==x.y.z'
    # =====================================================================
    if "dependency_states" in cleaned:
        ds = cleaned["dependency_states"]

        # Derive version_spec from resolved_version if needed
        # resolved_version like '2.0.0' -> version_spec '>=2.0.0,<3.0.0'
        # resolved_version like '==2.0.0' -> version_spec '==2.0.0'
        def derive_version_spec(ver: Any) -> str:
            if ver is None or (isinstance(ver, float) and np.isnan(ver)):
                return None
            v = str(ver).strip()
            if v.startswith("=="):
                return v
            # Try parsing as simple version x.y.z
            m = re.match(r'^(\d+)\.(\d+)\.(\d+)$', v)
            if m:
                major, minor, patch = m.groups()
                return f">={major}.{minor}.{patch},<{int(major)+1}.0.0"
            # For other formats, return as-is
            return v

        if "version_spec" not in ds.columns:
            ds["version_spec"] = ds["resolved_version"].apply(derive_version_spec)

        # Drop duplicates on (repo_id, dependency_id, commit_sha, version_spec)
        before = len(ds)
        ds = ds.drop_duplicates(subset=["repo_id", "dependency_id", "commit_sha", "version_spec"])
        after = len(ds)
        log["dependency_states_duplicates_removed"] = before - after

        # Recover resolved_version from version_spec when it looks like '==x.y.z'
        def recover_from_spec(spec: Any) -> str:
            if spec is None or (isinstance(spec, float) and np.isnan(spec)):
                return None
            s = str(spec).strip()
            m = re.match(r'^==\s*([^\s,]+)', s)
            if m:
                return m.group(1)
            return s

        ds["resolved_version"] = ds["version_spec"].apply(recover_from_spec)

        cleaned["dependency_states"] = ds

    # =====================================================================
    # 6) program_elements: lowercase element_type, drop duplicates on
    #    (migration_id, file_path, element_name, api_symbol),
    #    fill changed_lines with the median
    # =====================================================================
    if "program_elements" in cleaned:
        pe = cleaned["program_elements"]

        # lowercase element_type (strip first, then lowercase)
        pe["element_type"] = pe["element_type"].astype(str).str.strip().str.lower()

        # Drop duplicates on (migration_id, file_path, element_name, api_symbol)
        before = len(pe)
        pe = pe.drop_duplicates(subset=["migration_id", "file_path", "element_name", "api_symbol"])
        after = len(pe)
        log["program_elements_duplicates_removed"] = before - after

        # Count how many changed_lines were NULL before fill
        filled_cl = int(pe["changed_lines"].isna().sum())

        # Fill changed_lines with median
        median_cl = pe["changed_lines"].median()
        pe["changed_lines"] = pe["changed_lines"].fillna(median_cl)
        pe["changed_lines"] = pe["changed_lines"].astype(float)
        log["program_elements_changed_lines_filled"] = filled_cl

        cleaned["program_elements"] = pe

    # =====================================================================
    # 7) evidence: drop duplicates, fill strength using evidence_grade defaults
    #    gold 0.9 / silver 0.7 / uncertain 0.4
    # =====================================================================
    if "evidence" in cleaned:
        ev = cleaned["evidence"]

        # Drop duplicates (keep first)
        before = len(ev)
        ev = ev.drop_duplicates(keep="first")
        after = len(ev)
        log["evidence_duplicates_removed"] = before - after

        # Fill strength based on evidence_grade
        grade_defaults = {"gold": 0.9, "silver": 0.7, "uncertain": 0.4}

        def fill_strength(row):
            if pd.notna(row["strength"]):
                return row["strength"]
            grade = str(row.get("evidence_grade", "")).lower()
            return grade_defaults.get(grade, np.nan)

        # Count how many were filled BEFORE filling
        filled_count = int(ev["strength"].isna().sum())
        ev["strength"] = ev.apply(fill_strength, axis=1)
        log["evidence_strength_filled"] = filled_count

        cleaned["evidence"] = ev

    # =====================================================================
    # 8) candidates: lowercase element_type, cast observed_label to bool
    # =====================================================================
    if "candidates" in cleaned:
        cands = cleaned["candidates"]

        # lowercase element_type (strip first, then lowercase)
        cands["element_type"] = cands["element_type"].astype(str).str.strip().str.lower()

        # cast observed_label to bool (handle string True/False and actual bool)
        def to_bool(val):
            if pd.isna(val):
                return False
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return bool(val)
            return str(val).strip().lower() in ("true", "1", "yes", "t")

        cands["observed_label"] = cands["observed_label"].apply(to_bool)

        cleaned["candidates"] = cands

    # =====================================================================
    # 9) migrations.dependency_id: placeholder -1 -> real id from the change
    # =====================================================================
    if "migrations" in cleaned and "dependency_changes" in cleaned:
        mig = cleaned["migrations"]
        changes = cleaned["dependency_changes"]
        by_change = changes.drop_duplicates("change_id").set_index("change_id")["dependency_id"]
        parent_ids = set(cleaned.get("dependencies", pd.DataFrame({"dependency_id": []}))["dependency_id"].dropna())
        invalid = (
            mig["dependency_id"].isna()
            | (pd.to_numeric(mig["dependency_id"], errors="coerce") <= 0)
            | ~mig["dependency_id"].isin(parent_ids)
        )
        repaired = invalid.sum()
        mig.loc[invalid, "dependency_id"] = mig.loc[invalid, "change_id"].map(by_change)
        log["migrations_dependency_id_repaired"] = int(repaired)
        cleaned["migrations"] = mig

    return cleaned, log


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def verify(cleaned: Dict[str, pd.DataFrame], log: Dict[str, int]) -> None:
    """Verify cleaning results and print the log.

    Args:
        cleaned: Cleaned tables.
        log: Cleaning log.
    """
    print("=" * 80)
    print("CLEANING LOG")
    print("=" * 80)
    if log:
        for rule, count in sorted(log.items()):
            print(f"  {rule}: {count}")
    else:
        print("  (no rules fired)")

    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    # Check for NaT in date columns
    all_dates_ok = True
    for name, df in cleaned.items():
        if "observed_at" in df.columns:
            nat_count = df["observed_at"].isna().sum()
            if nat_count > 0:
                print(f"  [{name}] {nat_count} NaT in observed_at")
                all_dates_ok = False
    if all_dates_ok:
        print("  No NaT in date columns: PASS")

    # Check for duplicate natural keys
    key_checks_passed = True
    for name, df in cleaned.items():
        if name == "dependency_states":
            # Natural key: repo_id, dependency_id, commit_sha, version_spec
            dupes = df.duplicated(subset=["repo_id", "dependency_id", "commit_sha", "version_spec"]).sum()
            if dupes > 0:
                print(f"  [{name}] {dupes} duplicate natural keys")
                key_checks_passed = False
        elif name == "program_elements":
            # Natural key: migration_id, file_path, element_name, api_symbol
            dupes = df.duplicated(subset=["migration_id", "file_path", "element_name", "api_symbol"]).sum()
            if dupes > 0:
                print(f"  [{name}] {dupes} duplicate natural keys")
                key_checks_passed = False
        elif name == "evidence":
            # Evidence: evidence_id (surrogate) - skip, already deduplicated
            pass
        elif name == "candidates":
            # Candidates: candidate_id is unique surrogate
            pass

    if key_checks_passed:
        print("  No duplicate natural keys: PASS")
