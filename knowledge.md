# NAVTAC mini ETL - project context
Goal: PostgreSQL (10 tables) -> pandas (clean, join, features) -> PostgreSQL marts + Parquet -> analysis.
Stack: Python 3.11+, pandas, SQLAlchemy 2, psycopg2, pyarrow, pytest, Docker (Postgres 16 only).
Layout: sql/schema.sql, src/generate_data.py, src/etl_pipeline.py, data/raw (CSV), data/parquet, tests/.
Rules: DATABASE_URL env var only (no hardcoded secrets); every stage idempotent; never mutate raw tables;
no future leakage (history used for a candidate must have window_end < prediction_point);
ranking scores are review priorities, NOT probabilities; small functions, type hints, docstrings.
Run: docker compose up -d ; python src/generate_data.py ; python src/etl_pipeline.py init ; python src/etl_pipeline.py run

You are my senior data engineer pair-programmer. We are building a mini ETL project called NAVTAC
that
mimics a historical dependency-migration data model.
GOAL: PostgreSQL (10 related tables) -> pandas (clean, join, feature creation) -> load results back
to
PostgreSQL as mart_* tables AND export Parquet files -> final analysis.
STACK: Python 3.11+, pandas, SQLAlchemy 2, psycopg2-binary, pyarrow, pytest, Docker (Postgres 16
only).
LAYOUT: sql/schema.sql, src/generate_data.py, src/etl_pipeline.py, data/raw (CSV), data/parquet,
tests/.
TABLES: repositories, dependencies, dependency_states, dependency_changes, commits, pull_requests,
migrations, program_elements, evidence, candidates.
CHAIN: Repository -> Dependency -> State -> Change -> Migration -> Program Element -> Evidence ->
Candidate -> Review Score.
RULES: read DATABASE_URL from the environment (never hardcode passwords); every stage must be
idempotent;
never modify raw tables, write results to mart_* tables; no future leakage (history used for a
candidate
must have window_end < prediction_point); review scores are rankings, not probabilities; use small
functions with type hints and docstrings; after each task, run the verification command and fix
errors.
Acknowledge in 3 lines, then wait for my next prompt. Do not write code yet.
