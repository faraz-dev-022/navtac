"""Tests for the NAVTAC pipeline.

All tests run against a temporary SQLite database via DATABASE_URL so no
PostgreSQL/Docker is required.
"""

import ast
import inspect
from pathlib import Path

import pandas as pd

import clean as cl
import etl_pipeline as etl
import features as feat
import visualize as viz

ROOT = Path(__file__).resolve().parent.parent
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

NATURAL_KEYS = {
    "dependency_states": ["repo_id", "dependency_id", "commit_sha", "version_spec"],
    "program_elements": ["migration_id", "file_path", "element_name", "api_symbol"],
}

# Child column -> (parent table, parent key). CSV / SQLite use natural id names.
FK_MAP = {
    "dependencies": {"repo_id": ("repositories", "repo_id")},
    "commits": {"repo_id": ("repositories", "repo_id")},
    "dependency_states": {
        "repo_id": ("repositories", "repo_id"),
        "dependency_id": ("dependencies", "dependency_id"),
    },
    "dependency_changes": {
        "repo_id": ("repositories", "repo_id"),
        "dependency_id": ("dependencies", "dependency_id"),
        "from_state_id": ("dependency_states", "state_id"),
        "to_state_id": ("dependency_states", "state_id"),
    },
    "pull_requests": {"repo_id": ("repositories", "repo_id")},
    "migrations": {
        "repo_id": ("repositories", "repo_id"),
        "dependency_id": ("dependencies", "dependency_id"),
        "change_id": ("dependency_changes", "change_id"),
        "pr_id": ("pull_requests", "pr_id"),
    },
    "program_elements": {"migration_id": ("migrations", "migration_id")},
    "evidence": {"migration_id": ("migrations", "migration_id")},
    "candidates": {
        "target_repo_id": ("repositories", "repo_id"),
        "target_change_id": ("dependency_changes", "change_id"),
    },
}

MART_NAMES = [
    "dependency_changes_enriched",
    "migration_stats",
    "candidate_features",
    "experiment_metrics",
]


def _run_pipeline(engine):
    raw = etl.extract(engine)
    tables, log = cl.clean(raw)
    changes = feat.build_change_mart(tables)
    mig_stats = feat.build_migration_stats(tables, changes)
    cand = feat.build_candidate_features(tables)
    metrics = feat.evaluate(cand)
    marts = {
        "dependency_changes_enriched": changes,
        "migration_stats": mig_stats,
        "candidate_features": cand,
        "experiment_metrics": metrics,
    }
    etl.load(engine, marts)
    return tables, marts, log


def test_raw_row_counts_equal_csv_rows(dfs):
    """(1) Raw table row counts equal the number of rows in the CSV files."""
    expected = {t: len(pd.read_csv(RAW_DIR / f"{t}.csv")) for t in TABLES}
    actual = {t: len(dfs[t]) for t in TABLES}
    assert actual == expected


def test_clean_no_duplicate_natural_keys_and_no_nat(dfs):
    """(2) After clean() there are no duplicate natural keys and no NaT."""
    cleaned, _ = cl.clean(dfs)

    for table, key in NATURAL_KEYS.items():
        dupes = cleaned[table].duplicated(subset=key).sum()
        assert dupes == 0, f"{table} still has {dupes} duplicate natural keys"

    date_cols = dict(cl.DATE_COLS)
    date_cols.setdefault("dependency_states", [])
    if "observed_at" not in date_cols["dependency_states"]:
        date_cols["dependency_states"] = date_cols["dependency_states"] + ["observed_at"]

    for table, cols in date_cols.items():
        for col in cols:
            series = cleaned[table][col]
            nat_count = 0
            if pd.api.types.is_datetime64_any_dtype(series):
                nat_count = int(series.isna().sum())
            else:
                for v in series:
                    if v is pd.NaT or (isinstance(v, pd.Timestamp) and pd.isna(v)):
                        nat_count += 1
            assert nat_count == 0, f"NaT in {table}.{col} ({nat_count})"


def test_pipeline_idempotent(dfs, engine, tmp_path, monkeypatch):
    """(3) Running the pipeline twice yields identical mart row counts."""
    monkeypatch.setattr(etl, "PARQUET", tmp_path / "parquet")
    _run_pipeline(engine)
    counts1 = {
        name: len(pd.read_sql_table(f"mart_{name}", con=engine))
        for name in MART_NAMES
    }
    _run_pipeline(engine)
    counts2 = {
        name: len(pd.read_sql_table(f"mart_{name}", con=engine))
        for name in MART_NAMES
    }
    assert counts2 == counts1, f"first run: {counts1}\nsecond run: {counts2}"


def test_no_leakage_candidate_feature_migration_window(dfs):
    """(4) No candidate feature uses a migration with window_end >= prediction_point."""
    cleaned, _ = cl.clean(dfs)
    features = feat.build_candidate_features(cleaned).dropna(subset=["candidate_id"])

    mig = cleaned["migrations"].copy()
    mig["window_end"] = pd.to_datetime(mig["window_end"], errors="coerce")
    pe_with_mig = (
        cleaned["program_elements"]
        .dropna(subset=["migration_id", "api_symbol"])
        .merge(mig[["migration_id", "window_end"]], on="migration_id", how="left")
    )

    cands = cleaned["candidates"].copy()
    cands["prediction_point"] = pd.to_datetime(cands["prediction_point"], errors="coerce")
    assert (mig["window_end"] >= cands["prediction_point"].min()).any(), (
        "fixture has no post-cutoff migration to test leakage against"
    )

    for _, cand in cands.iterrows():
        cid, pp, api = cand["candidate_id"], cand["prediction_point"], cand["api_symbol"]
        if pd.isna(cid) or pd.isna(pp):
            continue
        feat_rows = features[features["candidate_id"] == cid]
        if feat_rows.empty:
            continue
        hist = feat_rows.iloc[0]["hist_migrations"]
        if pd.isna(hist):
            continue
        if pd.isna(api):
            past = pe_with_mig.iloc[0:0]
            future = pe_with_mig.iloc[0:0]
        else:
            past = pe_with_mig[
                (pe_with_mig["window_end"] < pp) & (pe_with_mig["api_symbol"] == api)
            ]
            future = pe_with_mig[
                (pe_with_mig["window_end"] >= pp) & (pe_with_mig["api_symbol"] == api)
            ]
        assert hist == len(past), (
            f"candidate {cid} hist_migrations={hist} "
            f"but eligible element rows={len(past)}"
        )
        if len(future) > 0:
            assert hist != len(past) + len(future), (
                f"candidate {cid} used {len(future)} migrations with "
                f"window_end >= prediction_point {pp}"
            )
        assert not (past["window_end"] >= pp).any(), (
            f"candidate {cid} uses a migration with window_end >= {pp}"
        )


def test_parquet_read_back_row_count(dfs, engine, tmp_path, monkeypatch):
    """(5) Parquet files read back with the same row count as the in-memory marts."""
    parquet_dir = tmp_path / "parquet"
    monkeypatch.setattr(etl, "PARQUET", parquet_dir)
    _, marts, _ = _run_pipeline(engine)
    assert parquet_dir.is_dir()
    for name, df in marts.items():
        path = parquet_dir / f"{name}.parquet"
        assert path.exists(), f"missing {path}"
        back = pd.read_parquet(path)
        assert len(back) == len(df), f"row count mismatch for {name}"


def test_all_foreign_keys_resolve(dfs):
    """(6) Every non-null foreign key resolves to an existing parent row."""
    cleaned, _ = cl.clean(dfs)
    for child, parents in FK_MAP.items():
        for col, (parent, parent_col) in parents.items():
            parent_ids = set(cleaned[parent][parent_col].dropna())
            child_vals = cleaned[child][col].dropna()
            unresolved = child_vals[~child_vals.isin(parent_ids)]
            assert unresolved.empty, (
                f"{child}.{col} has {len(unresolved)} rows referencing "
                f"non-existent {parent}.{parent_col}: {sorted(unresolved.unique())[:10]}"
            )


def test_png_count_equals_fig_functions(dfs, engine, tmp_path, monkeypatch):
    """(7) The number of PNGs equals the number of fig_* functions called in main()."""
    # Count fig_* calls inside visualize.main() from its own source.
    tree = ast.parse(inspect.getsource(viz.main))
    fig_calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("fig_")
    ]
    assert len(fig_calls) == len(set(fig_calls)), "main() calls a fig_* function twice"

    # Build the marts into a temp parquet dir (same as test 5) and render.
    parquet_dir = tmp_path / "parquet"
    fig_dir = tmp_path / "figures"
    monkeypatch.setattr(etl, "PARQUET", parquet_dir)
    _run_pipeline(engine)
    monkeypatch.setattr(viz, "PARQUET", parquet_dir)
    monkeypatch.setattr(viz, "FIG", fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)   # production gets this at import time
    viz.main()

    pngs = sorted(fig_dir.glob("*.png"))
    assert len(pngs) == len(fig_calls), (
        f"main() calls {len(fig_calls)} fig_* functions but {len(pngs)} PNGs were "
        f"written: {[p.name for p in pngs]}"
    )
    names = [p.stem for p in pngs]
    assert len(names) == len(set(names)), f"duplicate figure names: {names}"
