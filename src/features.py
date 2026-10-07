#!/usr/bin/env python3
"""Feature engineering functions for Navtac data.

Functions:
    build_change_mart(t) -> DataFrame
    build_migration_stats(t, changes) -> DataFrame
    build_candidate_features(t) -> DataFrame
"""

import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from packaging import version

from extract import TABLES


def build_change_mart(t: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join dependency_changes + dependencies + repositories.

    Parse from/to versions with packaging.version into major/minor/patch columns.
    Add is_major_bump.

    Args:
        t: Dict mapping table name to DataFrame (extracted from the database).

    Returns:
        DataFrame with change, dependency, and repository info, plus parsed
        version components and is_major_bump.
    """
    changes = t["dependency_changes"].copy()
    deps = t["dependencies"][["dependency_id", "package_name", "version"]].copy()
    repos = t["repositories"][["repo_id", "full_name", "role"]].copy()

    # Join dependency_changes with dependencies
    df = changes.merge(deps, on="dependency_id", how="left")

    # Join with repositories
    df = df.merge(repos, on="repo_id", how="left")

    # Include from_version/to_version from migrations for reference
    migrations = t["migrations"][["change_id", "repo_id", "from_version", "to_version"]].copy()
    df = df.merge(migrations, on=["change_id", "repo_id"], how="left", suffixes=("","_mig"))

    # The dependency_changes table already has version component columns
    # (from_major, from_minor, from_patch, to_major, to_minor, to_patch) as floats.
    # Cast them to integers where they are whole numbers (version components).
    for col in ["from_major", "from_minor", "from_patch", "to_major", "to_minor", "to_patch"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: int(v) if pd.notna(v) and float(v).is_integer() else v
            )

    # is_major_bump: use the is_major_bump column from the table; if missing, derive from major version comparison
    if "is_major_bump" in df.columns:
        df["is_major_bump"] = df["is_major_bump"].fillna(
            df["from_major"].combine(df["to_major"], lambda a, b: a != b)
        )
    else:
        df["is_major_bump"] = df["from_major"].combine(df["to_major"], lambda a, b: a != b)

    return df


def build_migration_stats(t: Dict[str, pd.DataFrame], changes: pd.DataFrame) -> pd.DataFrame:
    """Join migrations + changes + pull_requests; aggregate metrics.

    Args:
        t: Dict mapping table name to DataFrame.
        changes: The change mart DataFrame (from build_change_mart),
                 or the dependency_changes DataFrame.

    Returns:
        DataFrame with per-migration stats.
    """
    if "change_id" in changes.columns and "change_type" in changes.columns:
        change_df = changes
    else:
        change_df = t["dependency_changes"].copy()

    migrations = t["migrations"].copy()
    pulls = t["pull_requests"][["pr_id", "repo_id", "state", "merged_at"]].copy()
    elements = t["program_elements"].copy()
    evidence = t["evidence"].copy()

    df = migrations.copy()

    df = df.merge(pulls, on=["pr_id", "repo_id"], how="left")

    if "dependency_id" in change_df.columns:
        df = df.merge(change_df[["change_id", "repo_id", "dependency_id"]], on=["change_id", "repo_id"], how="left")

    df["window_start"] = pd.to_datetime(df["window_start"], errors="coerce")
    df["window_end"] = pd.to_datetime(df["window_end"], errors="coerce")
    if "merged_at" in df.columns:
        df["merged_at"] = pd.to_datetime(df["merged_at"], errors="coerce")
    else:
        df["merged_at"] = df["window_start"]
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    else:
        df["created_at"] = df["window_start"]

    df["duration_days"] = (df["window_end"] - df["window_start"]).dt.total_seconds() / 86400.0
    df["days_to_merge"] = (df["merged_at"] - df["window_start"]).dt.total_seconds() / 86400.0

    pe_valid = elements.dropna(subset=["migration_id"])
    pe_agg = (
        pe_valid.groupby("migration_id")
        .agg(
            n_elements=("element_id", "count"),
            n_files=("n_files", "sum"),
            total_changed_lines=("changed_lines", "sum"),
            n_distinct_symbols=("element_name", "nunique"),
        )
        .reset_index()
    )

    ev_valid = evidence.dropna(subset=["migration_id"])
    ev_agg = (
        ev_valid.groupby("migration_id")
        .agg(
            n_evidence=("evidence_id", "count"),
            mean_strength=("strength", "mean"),
        )
        .reset_index()
    )

    df = df.merge(pe_agg, on="migration_id", how="left", suffixes=("", "_pe"))
    df = df.merge(ev_agg, on="migration_id", how="left", suffixes=("", "_ev"))

    # The migrations table may already carry precomputed per-migration stats
    # (denormalised raw data).  Prefer those values and fall back to the
    # aggregates computed above instead of clobbering them with NaN.
    for col in ["n_elements", "n_files", "total_changed_lines", "n_distinct_symbols"]:
        alt = f"{col}_pe"
        if alt in df.columns:
            df[col] = df[col].where(df[col].notna(), df[alt])
            df = df.drop(columns=[alt])
    for col in ["n_evidence", "mean_strength"]:
        alt = f"{col}_ev"
        if alt in df.columns:
            df[col] = df[col].where(df[col].notna(), df[alt])
            df = df.drop(columns=[alt])

    for col in ["n_elements", "n_files", "total_changed_lines", "n_distinct_symbols", "n_evidence", "mean_strength"]:
        if col not in df.columns:
            df[col] = np.nan
    df["n_elements"] = df["n_elements"].fillna(0).astype(int)
    df["n_files"] = df["n_files"].fillna(0).astype(float)
    df["total_changed_lines"] = df["total_changed_lines"].fillna(0).astype(float)
    df["n_distinct_symbols"] = df["n_distinct_symbols"].fillna(0).astype(int)
    df["n_evidence"] = df["n_evidence"].fillna(0).astype(int)
    df["mean_strength"] = df["mean_strength"].fillna(0.0)

    df["elements_per_file"] = df["n_elements"] / df["n_files"].replace(0, np.nan)
    df["elements_per_file"] = df["elements_per_file"].fillna(0)

    cols = [
        "migration_id", "repo_id", "dependency_id", "change_id", "pr_id", "full_name",
        "package_name",
        "from_version", "to_version", "change_type", "is_major_bump",
        "window_start", "window_end", "merged_at", "duration_days", "days_to_merge",
        "evidence_grade", "n_elements", "n_files", "total_changed_lines",
        "n_distinct_symbols", "n_evidence", "mean_strength", "elements_per_file",
    ]
    available_cols = [c for c in cols if c in df.columns]
    return df[available_cols]


def build_candidate_features(t: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build candidate features with time-cutoff and historical signals.

    Args:
        t: Dict mapping table name to DataFrame.

    Returns:
        DataFrame with candidate features including hist_migrations,
        hist_repos, hist_freq, hist_score, score_static_only,
        score_static_plus_hist, rank_static_only, rank_static_plus_hist.
    """
    candidates = t["candidates"].copy()
    migrations = t["migrations"].copy()
    elements = t["program_elements"].copy()
    evidence = t["evidence"].copy()

    candidates["prediction_point"] = pd.to_datetime(
        candidates["prediction_point"], errors="coerce"
    )
    migrations["window_end"] = pd.to_datetime(migrations["window_end"], errors="coerce")

    # Evidence strength per migration (mean)
    ev_with_mig = evidence.dropna(subset=["migration_id"]).copy()
    ev_strength = (
        ev_with_mig.groupby("migration_id")["strength"].mean().reset_index().rename(
            columns={"strength": "mean_evidence_strength"}
        )
    )
    migrations_with_ev = migrations.merge(ev_strength, on="migration_id", how="left")

    # Element counts per migration
    pe_agg = (
        elements.dropna(subset=["migration_id"])
        .groupby("migration_id")
        .agg(
            n_elements=("element_id", "count"),
            n_files=("n_files", "sum"),
            total_changed_lines=("changed_lines", "sum"),
            n_distinct_symbols=("element_name", "nunique"),
        )
        .reset_index()
    )

    # Build pe_with_mig for api_symbol-based features
    pe_with_mig = elements.dropna(subset=["migration_id", "api_symbol"]).copy()
    pe_with_mig = pe_with_mig.merge(
        migrations[["migration_id", "window_end", "repo_id"]],
        on="migration_id", how="left"
    )

    # Filter static candidates
    static_candidates = candidates[candidates["generated_by"] == "static"].copy()
    static_candidates["prediction_point"] = pd.to_datetime(
        static_candidates["prediction_point"], errors="coerce"
    )

    # ------------------------------------------------------------------
    # Every historical feature value depends only on the candidate's
    # (prediction_point, api_symbol): which migrations are eligible
    # (window_end < prediction_point) and, per api_symbol, how many
    # of their element rows mention the symbol, how many distinct
    # repos they span, and the mean evidence strength of those
    # migrations.  Compute those aggregates once per distinct
    # prediction point instead of rescanning the migrations and
    # elements tables for every candidate.
    # ------------------------------------------------------------------
    mig_strength_map = migrations_with_ev.set_index("migration_id")[
        "mean_evidence_strength"
    ]

    point_stats = {}
    for pp in candidates["prediction_point"].dropna().unique():
        eligible = migrations[migrations["window_end"] < pp]
        n_eligible = len(eligible)
        if n_eligible == 0:
            point_stats[pp] = (0, None, None)
            continue
        eligible_elements = pe_with_mig[
            pe_with_mig["migration_id"].isin(eligible["migration_id"])
        ]
        grp = eligible_elements.groupby("api_symbol")
        sym_counts = grp.size()
        sym_repos = grp["repo_id"].nunique()
        # Mean evidence strength per symbol over its *unique*
        # migrations (a migration appears once per element row).
        uniq_pairs = eligible_elements[["api_symbol", "migration_id"]].drop_duplicates()
        uniq_pairs["mean_evidence_strength"] = uniq_pairs["migration_id"].map(
            mig_strength_map
        )
        sym_strength = uniq_pairs.groupby("api_symbol")[
            "mean_evidence_strength"
        ].mean()
        point_stats[pp] = (n_eligible, sym_counts, (sym_repos, sym_strength))

    def _hist_signals(pp, api_symbol):
        """(n_eligible, hist_migrations, hist_repos, hist_freq, mean_strength)."""
        n_eligible, sym_counts, sym_aggs = point_stats.get(pp, (0, None, None))
        if sym_counts is None or pd.isna(api_symbol):
            return n_eligible, 0, 0, 0.0, 0.0
        count = sym_counts.get(api_symbol)
        if count is None or pd.isna(count):
            return n_eligible, 0, 0, 0.0, 0.0
        sym_repos, sym_strength = sym_aggs
        mean_strength = sym_strength.get(api_symbol)
        mean_strength = float(mean_strength) if pd.notna(mean_strength) else 0.0
        hist_migrations = int(count)
        hist_repos = int(sym_repos.get(api_symbol, 0))
        hist_freq = hist_migrations / n_eligible
        return n_eligible, hist_migrations, hist_repos, hist_freq, mean_strength

    feature_rows = []

    for _, cand in static_candidates.iterrows():
        pred_point = cand["prediction_point"]
        if pd.isna(pred_point):
            continue

        n_eligible, hist_migrations, hist_repos, hist_freq, mean_strength = \
            _hist_signals(pred_point, cand.get("api_symbol"))

        static_score = cand.get("static_score")
        if pd.isna(static_score):
            static_score = 0.0

        if n_eligible == 0:
            # No eligible migrations: the historical arm contributes nothing.
            hist_score = 0.0
            score_static_plus_hist = 0.0
        else:
            hist_score = hist_freq * mean_strength
            score_static_plus_hist = 0.6 * static_score + 0.4 * hist_score
        score_static_only = static_score

        feature_rows.append({
            "candidate_id": cand["candidate_id"],
            "generated_by": cand["generated_by"],
            "static_score": static_score,
            "hist_migrations": hist_migrations,
            "hist_repos": hist_repos,
            "hist_freq": hist_freq,
            "hist_score": hist_score,
            "score_static_only": score_static_only,
            "score_static_plus_hist": score_static_plus_hist,
            "api_symbol": cand.get("api_symbol"),
        })

    # Build the feature DataFrame for static candidates
    features_static = pd.DataFrame(feature_rows)

    # Historical expansion candidates
    hist_candidates = candidates[candidates["generated_by"] == "historical_expansion"].copy()
    hist_candidates["prediction_point"] = pd.to_datetime(
        hist_candidates["prediction_point"], errors="coerce"
    )

    hist_feature_rows = []
    for _, cand in hist_candidates.iterrows():
        pred_point = cand["prediction_point"]
        if pd.isna(pred_point):
            continue

        n_eligible, hist_migrations, hist_repos, hist_freq, mean_strength = \
            _hist_signals(pred_point, cand.get("api_symbol"))

        if n_eligible == 0:
            hist_score = 0.0
            score_static_plus_hist = 0.0
        else:
            hist_score = hist_freq * mean_strength
            score_static_plus_hist = 0.4 * hist_score

        hist_feature_rows.append({
            "candidate_id": cand["candidate_id"],
            "generated_by": cand["generated_by"],
            "static_score": 0.0,
            "hist_migrations": hist_migrations,
            "hist_repos": hist_repos,
            "hist_freq": hist_freq,
            "hist_score": hist_score,
            "score_static_only": np.nan,        # hist rows have no static-only arm
            "score_static_plus_hist": score_static_plus_hist,
            "api_symbol": cand.get("api_symbol"),
        })

    features_hist = pd.DataFrame(hist_feature_rows)

    # Combine static and historical features
    if len(features_static) > 0 and len(features_hist) > 0:
        features = pd.concat([features_static, features_hist], ignore_index=True)
    elif len(features_static) > 0:
        features = features_static
    else:
        features = features_hist

    # Left-join onto candidates
    # Rename feature columns to avoid collision with base columns that already
    # exist in the candidates table (e.g. score_static_only, rank_static_only)
    features_renamed = features.rename(columns={
        "static_score": "feat_static_score",
        "hist_migrations": "feat_hist_migrations",
        "hist_repos": "feat_hist_repos",
        "hist_freq": "feat_hist_freq",
        "hist_score": "feat_hist_score",
        "score_static_only": "feat_score_static_only",
        "score_static_plus_hist": "feat_score_static_plus_hist",
        "rank_static_only": "feat_rank_static_only",
        "rank_static_plus_hist": "feat_rank_static_plus_hist",
        "api_symbol": "feat_api_symbol",
    })

    merged = candidates.merge(
        features_renamed,
        on=["candidate_id", "generated_by"],
        how="left",
    )

    # Select the expected output columns (base + feature columns)
    # Use the renamed feature columns
    expected_feature_cols = [
        "feat_hist_migrations", "feat_hist_repos", "feat_hist_freq", "feat_hist_score",
        "feat_score_static_only", "feat_score_static_plus_hist",
        "feat_rank_static_only", "feat_rank_static_plus_hist",
    ]
    available_feature = [c for c in expected_feature_cols if c in merged.columns]

    # Rename back to the original feature names
    for new_col, old_col in [
        ("hist_migrations", "feat_hist_migrations"),
        ("hist_repos", "feat_hist_repos"),
        ("hist_freq", "feat_hist_freq"),
        ("hist_score", "feat_hist_score"),
        ("score_static_only", "feat_score_static_only"),
        ("score_static_plus_hist", "feat_score_static_plus_hist"),
        ("rank_static_only", "feat_rank_static_only"),
        ("rank_static_plus_hist", "feat_rank_static_plus_hist"),
    ]:
        if old_col in merged.columns:
            merged[new_col] = merged[old_col]

    # Rank columns (use the renamed columns after the rename back)
    static_mask = merged["generated_by"] == "static"
    if static_mask.any():
        rank_static = (
            merged.loc[static_mask, "score_static_only"]
            .rank(ascending=False, method="min")
            .astype("Int64")
        )
        merged.loc[static_mask, "rank_static_only"] = rank_static

    rank_hist = (
        merged["score_static_plus_hist"]
        .rank(ascending=False, method="min")
        .astype("Int64")
    )
    merged["rank_static_plus_hist"] = rank_hist

    base_cols = ["candidate_id", "generated_by", "static_score", "evidence_id",
                 "observed_label", "prediction_point", "api_symbol",
                 "n_eligible_migrations"]
    available_base = [c for c in base_cols if c in merged.columns]

    # Use the original feature column names (not the _feat_ prefixed ones)
    original_feature_cols = [
        "hist_migrations", "hist_repos", "hist_freq", "hist_score",
        "score_static_only", "score_static_plus_hist",
        "rank_static_only", "rank_static_plus_hist",
    ]
    available_feature = [c for c in original_feature_cols if c in merged.columns]

    return merged[available_base + available_feature]


def evaluate(cand: pd.DataFrame) -> pd.DataFrame:
    """Compare static-only vs static-plus-hist experiment arms.

    Metrics (plain language):

    * **candidates**: the number of candidates kept for a given arm
      (only rows where that arm's score is not null).

    * **candidate_recall**: the fraction of all true positives (observed_label=True)
      that appear in the pool for this arm. It answers: "of all the real positives
      we were trying to find, what fraction did this arm's ranking bring to the top?"

    * **recall@K**: of the top K ranked candidates for an arm, how many of the
      total positives overall appear there. It answers: "if I only look at the
      first K candidates, how many of the real positive cases do I catch?"

    * **precision@K**: of the top K ranked candidates for an arm, what fraction
      are actually true positives. It answers: "of the first K candidates I
      look at, how many are genuinely useful?"

    * **MRR** (Mean Reciprocal Rank): for each arm, find the rank (position)
      of the first true positive in the sorted pool, then compute 1 / rank.
      It answers: "how early in the ranking does the first positive show up?"

    Args:
        cand: the output of build_candidate_features — a DataFrame with columns
              score_static_only (arm 'static_only'), score_static_plus_hist (arm
              'static_plus_hist'), rank_static_only, rank_static_plus_hist, and
              observed_label (boolean: True = positive).

    Returns:
        A DataFrame with one row per arm ('static_only', 'static_plus_hist') and
        columns: arm, candidates, candidate_recall, recall@5, recall@10,
        recall@15, precision@5, precision@10, precision@15, mrr.
    """
    # Columns that hold the score and rank for each arm
    score_cols = {
        "static_only": "score_static_only",
        "static_plus_hist": "score_static_plus_hist",
    }
    rank_cols = {
        "static_only": "rank_static_only",
        "static_plus_hist": "rank_static_plus_hist",
    }

    rows = []
    for arm in ("static_only", "static_plus_hist"):
        score_col = score_cols[arm]
        rank_col = rank_cols[arm]

        # Keep only candidates that have a non-null score for this arm
        pool = cand[cand[score_col].notna()].copy()
        # Sort by rank (ascending = rank 1 = best)
        pool = pool.sort_values(rank_col)

        total_pos = int(pool["observed_label"].sum())
        total_all_pos = int(cand["observed_label"].sum())
        if total_all_pos == 0:
            total_all_pos = 1

        row = {
            "arm": arm,
            "candidates": len(pool),
            "candidate_recall": (pool["observed_label"].sum() / total_all_pos)
            if total_all_pos > 0
            else 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "recall@15": 0.0,
            "precision@5": 0.0,
            "precision@10": 0.0,
            "precision@15": 0.0,
            "mrr": 0.0,
        }

        # Compute recall@K and precision@K for each K
        for k in (5, 10, 15):
            top_k = pool.head(k)
            if len(top_k) > 0:
                row[f"recall@{k}"] = top_k["observed_label"].sum() / total_all_pos
                row[f"precision@{k}"] = top_k["observed_label"].mean()

        # Compute MRR: reciprocal rank of the first positive
        pos_indices = pool.index[pool["observed_label"]].tolist()
        if pos_indices:
            first_pos_rank = pool.loc[pos_indices[0], rank_col]
            row["mrr"] = 1.0 / first_pos_rank

        rows.append(row)

    return pd.DataFrame(rows).round(4)


# Re-export for convenience
__all__ = [
    "build_change_mart",
    "build_migration_stats",
    "build_candidate_features",
    "evaluate",
]

