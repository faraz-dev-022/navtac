# Navtac

ETL and feature pipeline for software-dependency **migration** analysis: extract
ten raw tables, clean injected data-quality issues, build marts (change stats,
migration stats, time-safe candidate features, experiment metrics), and load
them to SQL plus Parquet.

The CSV under `data/raw/` is **synthetic** (see Limitations).

## Setup

Python 3.10+ recommended. Create and activate a virtual environment, then
install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tests do **not** need Docker. They set `DATABASE_URL` to a temporary SQLite
file.

Optional PostgreSQL (local compose):

```bash
docker compose up -d
export DATABASE_URL=postgresql+psycopg2://navtac:navtac@localhost:5432/navtac
```

Default URL if unset: `sqlite:///navtac_test.db`.

## Run commands

```bash
# Load CSVs into SQL (PostgreSQL or SQLite, depending on DATABASE_URL)
python src/etl_pipeline.py init

# Extract → clean → features → load marts + Parquet
python src/etl_pipeline.py run

# Test suite (temporary SQLite; no Docker)
pytest -q
```

Mart tables are written as `mart_<name>`. Parquet files go to `data/parquet/`
(or a temp directory during tests).

## Architecture

```
data/raw/*.csv
        |  init (optional)
        v
   SQL raw tables          DATABASE_URL
   (SQLite or Postgres)    sqlite: no Docker
        |  extract()
        v
   pandas DataFrames
        |  clean()          types, text, nulls, dupes, FKs
        v
   cleaned tables
        |  features
        |   • build_change_mart
        |   • build_migration_stats
        |   • build_candidate_features   (window_end < prediction_point)
        |   • evaluate                   (static vs static+hist)
        v
   mart DataFrames
        |  load()
        +----> SQL  mart_*
        +----> Parquet  data/parquet/<name>.parquet
```

Source layout:

- `src/extract.py` — read ten tables
- `src/clean.py` — data-quality rules
- `src/features.py` — joins, stats, leakage-safe candidate features
- `src/etl_pipeline.py` — CLI (`init` / `run`) and Parquet/SQL load
- `src/generate_data.py` — synthetic generator (seed 42)
- `tests/test_pipeline.py` — row counts, clean invariants, idempotence,
  leakage, Parquet round-trip, foreign keys

## Data-quality issues handled

Dirt is injected in generation and repaired in `clean()`:

- **Mixed date formats** on `dependency_states.observed_at` (`YYYY-MM-DD`,
  `dd/MM/YYYY`, ISO timestamps, `Mon DD, YYYY`, offsets). Parsed with explicit
  formats only (no `dayfirst` guessing). Other timestamp columns coerced to
  datetime.
- **Text noise**: `repositories.language` strip + lowercase;
  `dependencies.package_name` strip, lowercase, collapse `[-_.]+` to `-`;
  `program_elements` / `candidates` `element_type` strip + lowercase.
- **Missing values**: `stars` and `changed_lines` median fill;
  `commits.files_changed` per-repo median; `resolved_version` recovered from
  `==x.y.z` specs; `evidence.strength` from grade defaults (gold 0.9 / silver
  0.7 / uncertain 0.4); `static_score` treated as 0 when null in scoring.
- **Duplicate natural keys**: `dependency_states` on
  `(repo_id, dependency_id, commit_sha, version_spec)`; `program_elements` on
  `(migration_id, file_path, element_name, api_symbol)`; `evidence` row
  duplicates.
- **Placeholder FKs**: `migrations.dependency_id = -1` filled from the linked
  `dependency_changes` row.

## Limitations

The data is **synthetic**. Repositories, commits, PRs, migrations, evidence, and
candidate labels are generated (`src/generate_data.py`, seed 42) with known
dirt, not scraped from real GitHub history. Metrics and feature scores are
illustrative; they are not production migration-risk measurements.
