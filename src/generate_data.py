#!/usr/bin/env python3
"""Generate synthetic migration data for Navtac.

Creates 10 CSV files in data/raw/ with a fixed seed (42) using pandas and numpy.
The data is internally consistent: repositories -> dependencies -> commits ->
dependency_states -> dependency_changes -> migrations -> program_elements ->
evidence -> candidates, plus pull_requests linking repos to migrations.

Scale: every table has more than 100k rows. There are N_REPOS (120,000)
repositories; each repository gets 2 commits, 2 dependency states,
1 dependency change, 1 pull request, 1 migration and 2 program elements;
each program element gets 1 evidence row; plus 120,000 candidates.

INJECTED DIRT:
  - Mixed date formats in dependency_states.observed_at
  - Language casing/whitespace variations
  - NULLs in stars, files_changed, resolved_version, changed_lines, strength,
    static_score
  - Duplicate rows with new surrogate IDs in dependency_states,
    program_elements, evidence
  - Messy element_type casing in program_elements and candidates
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fixed seed
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Separate RNG for injected dirt (deterministic separate stream)
DIRT_SEED = 1234
rng_dirt = np.random.default_rng(DIRT_SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Scale: every table ends up with > 100k rows
# ---------------------------------------------------------------------------
N_REPOS = 120_000              # repositories (also changes, PRs, migrations)
COMMITS_PER_REPO = 2           # 240,000 commits
STATES_PER_REPO = 2            # 240,000 dependency states (+ duplicates)
ELEMENTS_PER_MIGRATION = 2     # 240,000 program elements (+ duplicates)
EVIDENCE_PER_ELEMENT = 1       # 240,000 evidence rows (+ duplicates)
N_CANDIDATES = 120_000         # candidates
PREDICTION_POINT = pd.Timestamp("2024-07-01 10:00:00")

# ---------------------------------------------------------------------------
# Constant definitions
# ---------------------------------------------------------------------------
REPO_NAMES = {
    "historical": [
        "acme/billing-api",
        "acme/user-service",
        "devtools/data-validator",
        "webkit/http-client",
        "orbit/analytics-core",
    ],
    "target": "acme/target-orders-app",
}

ROLE_OLD = "historical"
ROLE_NEW = "target"

# Anchor commit per repo: commit_sha where dependency version changes.
# Repositories 1-5 are historical, 6 is the prediction target.
# Repos 3 and 6 have windows that end AFTER the prediction point
# (2024-07-01) so the leakage guard has something to exclude.
ANCHOR_COMMITS = {
    1: ("58970d4508025694fe4c3c429a43bb88aa2ccdbd", "1.10.14", "2.0.0", pd.Timestamp("2024-02-20")),
    2: ("e191148c490435112b0077c349bab4cae020b706", "1.9.2", "2.1.1", pd.Timestamp("2023-08-24")),
    3: ("5f58d7ee87ad6e90d924d59f7db74c18dd11b362", "1.10.2", "2.4.2", pd.Timestamp("2024-09-10")),
    4: ("fcc960446c28da106cbc7b9bfff84ff9f2c3d326", "2.25.1", "2.31.0", pd.Timestamp("2023-06-01")),
    5: ("c5192baadd71525534b0323735bf772881cc420a", "1.4.46", "2.0.0", pd.Timestamp("2023-03-15")),
    6: ("2e8cfd8f955d8b4852efa699617f9ebad01da9d8", "1.10.14", "2.0.0", pd.Timestamp("2024-07-01")),
}

# Generated repos (id >= 7) get anchor dates spread over this range so that
# window_end (= anchor + 21 days) stays before the prediction point.
GEN_ANCHOR_START = pd.Timestamp("2023-01-01")
GEN_ANCHOR_END = pd.Timestamp("2024-05-15")

CHANGE_TYPES = ["version_upgrade", "specifier_change"]

# Evidence grades for migrations
EVIDENCE_GRADES = ["gold", "silver", "uncertain"]
EVIDENCE_GRADE_MESSY = ["gold", "silver", "uncertain", "Gold", "SILVER", "  uncertain  "]

# Element types for program_elements
ELEMENT_TYPES = ["class", "function", "method", "test", "module", "variable", "import"]

# api_symbols for program_elements
API_SYMBOLS = ["validator", "dict", "Config", "parse_obj", "root_validator",
               "parse_item", "build_invoice", "BaseModel", "ItemOut",
               "DateTime", "typing", "logging", "Optional", "Field", "uuid",
               "serializers", "session", "Order", "User", "validate_email",
               "to_payload", "load_config", "Client", "send", "from_row",
               "OrderSchema", "Payload", "test_settings", "test_client_timeout",
               "test_order_json", "test_order_json", "validate_email"]

# Element names for program_elements
ELEMENT_NAMES = [
    "serialize_user", "parse_item", "build_invoice", "validate_email",
    "Settings", "ItemOut", "Payload", "Connection", "Session",
    "get_db", "Order", "User", "OrderSchema", "UserSchema",
    "test_settings", "test_client_timeout", "test_order_json",
    "load_config", "to_payload", "from_row",
]

# Messy element_type casing to inject
ELEMENT_TYPE_MESSY = ["Class", "Function", "METHOD", "  test  ", "Module", "Variable", "IMPORT", "  Class  "]

# File paths for program_elements / evidence / candidates
FILE_PATHS = [
    "app/api.py", "app/core/config.py", "app/schemas/order.py",
    "app/utils/serializers.py", "app/services/billing.py",
    "app/models/user.py", "app/db/session.py", "tests/test_models.py",
    "tests/test_config.py", "app/utils/__init__.py",
]
FILE_PATH_WEIGHTS = [0.2, 0.2, 0.15, 0.1, 0.1, 0.08, 0.07, 0.05, 0.03, 0.02]

# Commit messages / authors
COMMIT_MESSAGES = [
    "fix bug", "add tests", "refactor", "chore: update deps",
    "improve performance", "docs: update", "fix typo",
    "add config", "remove unused", "optimize query",
]
AUTHORS = ["ana.silva", "bilal.khan", "carlos.m", "dana.w", "elon.m"]
AUTHOR_INITIALS = ["a", "b", "c", "d", "e"]

# Mixed date formats for dependency_states.observed_at (all parseable by
# clean.parse_mixed_dates with explicit formats).
OBSERVED_AT_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
    "%b %d, %Y",
    "%Y-%m-%d %H:%M:%S+00",
    "%Y-%m-%dT%H:%M:%S.%fZ",
)

PR_TITLES = [
    "Upgrade pydantic to 2.x", "Migrate to v2 API",
    "Update dependencies", "Fix type hints",
    "Refactor config loading", "Remove deprecated methods",
]
PR_STATES = ["merged", "merged", "merged", "closed", "open", "MERGED", "  merged  "]

# Evidence source / raw_data variants (messy)
SOURCE_VARIANTS = [
    "github.com/acme/billing-api/commit/",
    "https://github.com/acme/billing-api/commit/",
    "git@github.com:acme/billing-api.git",
    "gh:acme/billing-api@",
]
RAW_DATA_VARIANTS = ['{"score": 0.5}', "N/A", "{}", "[1,2,3]", "legacy format"]


# ---------------------------------------------------------------------------
# Per-repo attribute vectors (repos 1-6 keep the fixed anchor story)
# ---------------------------------------------------------------------------
def _repo_names(n_repos: int) -> list[str]:
    """Repository full names: 5 historical + 1 target + generated ones."""
    names = [REPO_NAMES["historical"][i - 1] if i <= 5 else REPO_NAMES["target"]
             for i in range(1, 7)]
    names += [f"acme/team-{i % 40:02d}/svc-{i:06d}" for i in range(7, n_repos + 1)]
    return names


def _anchor_shas(n_repos: int) -> list[str]:
    """Anchor commit SHAs per repo (fixed for repos 1-6, random after)."""
    shas = [ANCHOR_COMMITS[i][0] for i in range(1, 7)]
    if n_repos > 6:
        ints = rng.integers(0, 16**15, size=n_repos - 6)
        shas += [f"{v:040x}" for v in ints]
    return shas


def _anchor_dates(n_repos: int) -> pd.DatetimeIndex:
    """Anchor commit dates per repo (fixed for repos 1-6)."""
    dates = pd.DatetimeIndex([ANCHOR_COMMITS[i][3] for i in range(1, 7)])
    if n_repos > 6:
        span = int((GEN_ANCHOR_END - GEN_ANCHOR_START).days)
        extra = GEN_ANCHOR_START + pd.to_timedelta(
            rng.integers(0, span, size=n_repos - 6), unit="D"
        )
        dates = dates.append(extra)
    return dates


def _versions(n_repos: int) -> tuple[list[str], list[str]]:
    """(from_version, to_version) per repo: fixed for repos 1-6."""
    from_v = [ANCHOR_COMMITS[i][1] for i in range(1, 7)]
    to_v = [ANCHOR_COMMITS[i][2] for i in range(1, 7)]
    if n_repos > 6:
        n = n_repos - 6
        from_v += [f"1.{a}.{b}" for a, b in
                   zip(rng.integers(0, 30, size=n), rng.integers(0, 20, size=n))]
        to_v += [f"2.{a}.{b}" for a, b in
                 zip(rng.integers(0, 30, size=n), rng.integers(0, 20, size=n))]
    return from_v, to_v


def _packages(n_repos: int) -> np.ndarray:
    """Package names per repo (pydantic for the six story repos)."""
    packages = rng.choice(
        ["pydantic", "pydantic", "pydantic", "requests", "sqlalchemy", "fastapi"],
        size=n_repos,
    )
    packages[:6] = "pydantic"
    return packages


def _random_shas(count: int) -> list[str]:
    """Random 40-char commit SHAs."""
    ints = rng.integers(0, 16**15, size=count)
    return [f"{v:040x}" for v in ints]


def _format_dates(dates: pd.DatetimeIndex, formats: tuple[str, ...]) -> np.ndarray:
    """Format each date with a randomly chosen explicit format (dirty)."""
    out = np.empty(len(dates), dtype=object)
    choice = rng.integers(0, len(formats), size=len(dates))
    for i, fmt in enumerate(formats):
        mask = choice == i
        out[mask] = dates[mask].strftime(fmt)
    return out


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------
def generate_repositories(n_repos: int = N_REPOS) -> pd.DataFrame:
    """n_repos repositories: 5 historical + 1 target + generated ones."""
    names = _repo_names(n_repos)
    roles = np.array([ROLE_NEW if i == 6 else ROLE_OLD for i in range(1, n_repos + 1)],
                     dtype=object)
    languages = rng.choice(["Python", "python", "PYTHON", "  Python  "], size=n_repos)
    stars = rng.integers(100, 5000, size=n_repos).astype(float)
    stars[rng.random(n_repos) < 0.05] = np.nan          # injected NULLs
    created = pd.Timestamp("2020-01-01") + pd.to_timedelta(
        rng.integers(0, 1000, size=n_repos), unit="D"
    )
    df = pd.DataFrame({
        "repo_id": np.arange(1, n_repos + 1),
        "full_name": names,
        "clone_url": [f"https://github.com/{n}.git" for n in names],
        "default_branch": "main",
        "language": languages,
        "stars": stars,
        "created_at": created,
        "role": roles,
    })
    # Target repo keeps its fixed profile
    target = df["repo_id"] == 6
    df.loc[target, "language"] = "Python"
    df.loc[target, "stars"] = 450.0
    df.loc[target, "created_at"] = pd.Timestamp("2021-06-01")
    return df


def generate_commits(n_repos: int = N_REPOS,
                     per_repo: int = COMMITS_PER_REPO) -> pd.DataFrame:
    """per_repo commits per repository (chronological); the first is the anchor."""
    total = n_repos * per_repo
    anchor_shas = np.array(_anchor_shas(n_repos), dtype=object)
    anchor_dates = _anchor_dates(n_repos)

    repo_ids = np.repeat(np.arange(1, n_repos + 1), per_repo)
    is_anchor = np.zeros(total, dtype=bool)
    is_anchor[::per_repo] = True

    # Anchor rows reuse the repo's anchor SHA; the rest get random SHAs.
    shas = np.empty(total, dtype=object)
    shas[is_anchor] = anchor_shas
    shas[~is_anchor] = _random_shas(int((~is_anchor).sum()))

    # Anchor rows sit exactly on the anchor date; others within +/- 180 days.
    base = np.repeat(anchor_dates.to_numpy(), per_repo)
    offsets = rng.integers(-180, 181, size=total)
    offsets[is_anchor] = 0
    committed_at = pd.to_datetime(base) + pd.to_timedelta(offsets, unit="D")

    # Anchor commit messages describe the version bump.
    from_v, to_v = _versions(n_repos)
    messages = np.empty(total, dtype=object)
    messages[is_anchor] = [f"Upgrade {f} -> {t}" for f, t in zip(from_v, to_v)]
    messages[~is_anchor] = rng.choice(COMMIT_MESSAGES, size=int((~is_anchor).sum()))

    # Injected dirt: at most one NULL files_changed per repo (per-repo median
    # fill in clean() always has a value to take).
    files_changed = rng.integers(1, 20, size=total)
    null_repos = rng.choice(n_repos, size=int(n_repos * 0.08), replace=False)
    for r in null_repos:
        files_changed[r * per_repo + rng.integers(0, per_repo)] = -1
    files_changed = np.where(files_changed < 0, np.nan, files_changed).astype(float)

    return pd.DataFrame({
        "commit_id": np.arange(1, total + 1),
        "repo_id": repo_ids,
        "commit_sha": shas,
        "author_name": rng.choice(AUTHORS, size=total),
        "author_email": [f"{i}@example.com" for i in rng.choice(AUTHOR_INITIALS, size=total)],
        "message": messages,
        "files_changed": files_changed,
        "additions": rng.integers(1, 100, size=total),
        "deletions": rng.integers(0, 50, size=total),
        "committed_at": committed_at,
    })


def generate_dependencies(n_repos: int = N_REPOS) -> pd.DataFrame:
    """One dependency per repository (dependency_id == repo_id)."""
    from_v, _ = _versions(n_repos)
    return pd.DataFrame({
        "dependency_id": np.arange(1, n_repos + 1),
        "repo_id": np.arange(1, n_repos + 1),
        "package_name": _packages(n_repos),
        "version": from_v,
    })


def generate_dependency_states(n_repos: int = N_REPOS,
                             per_repo: int = STATES_PER_REPO) -> pd.DataFrame:
    """
    per_repo states per repository (old + new version on the anchor commit).
    Mixed date formats in observed_at, some NULLs, duplicate rows with new
    surrogate IDs (duplicates share the natural key so clean() removes them).
    """
    total = n_repos * per_repo
    anchor_shas = np.array(_anchor_shas(n_repos), dtype=object)
    anchor_dates = _anchor_dates(n_repos)
    from_v, to_v = _versions(n_repos)

    repo_ids = np.repeat(np.arange(1, n_repos + 1), per_repo)
    is_new = np.zeros(total, dtype=bool)
    is_new[1::per_repo] = True

    # Per-row versions: each repo contributes per_repo rows (old + new).
    resolved = np.where(
        is_new,
        np.repeat(np.array(to_v, dtype=object), per_repo),
        np.repeat(np.array(from_v, dtype=object), per_repo),
    )

    # Old state: before the anchor. New state: around the anchor.
    base = np.repeat(anchor_dates.to_numpy(), per_repo)
    offsets = rng.integers(-30, 0, size=total)
    offsets[is_new] = rng.integers(-10, 11, size=int(is_new.sum()))
    observed_at = _format_dates(pd.to_datetime(base) + pd.to_timedelta(offsets, unit="D"),
                                OBSERVED_AT_FORMATS)

    files_changed = rng.integers(1, 20, size=total).astype(float)
    changed_lines = rng.integers(1, 500, size=total).astype(float)
    strength = np.round(rng.uniform(0.1, 1.0, size=total), 4)
    for col in (files_changed, changed_lines, strength):
        col[rng.random(total) < 0.10] = np.nan           # injected NULLs

    rows = pd.DataFrame({
        "state_id": np.arange(1, total + 1),
        "repo_id": repo_ids,
        "commit_sha": np.repeat(anchor_shas, per_repo),
        "observed_at": observed_at,
        "resolved_version": resolved,
        "files_changed": files_changed,
        "changed_lines": changed_lines,
        "strength": strength,
        "dependency_id": repo_ids,
    })

    # Duplicate rows with NEW surrogate IDs: same natural key as the new
    # state, NULL metrics. clean() drops them (keep="first" keeps the
    # original, so state_id 2*(repo-1)+2 stays resolvable).
    n_dup_repos = int(n_repos * 0.08)
    dup_repos = rng.choice(np.arange(1, n_repos + 1), size=n_dup_repos, replace=False)
    dup_rows = []
    next_id = total + 1
    for r in dup_repos:
        anchor = anchor_dates[r - 1]
        for _ in range(rng.integers(1, 3)):
            dup_rows.append({
                "state_id": next_id,
                "repo_id": r,
                "commit_sha": anchor_shas[r - 1],
                "observed_at": (anchor + pd.Timedelta(days=int(rng.integers(-10, 11)))).strftime(
                    random.choice(OBSERVED_AT_FORMATS)),
                "resolved_version": to_v[r - 1],
                "files_changed": None,
                "changed_lines": None,
                "strength": None,
                "dependency_id": r,
            })
            next_id += 1
    return pd.concat([rows, pd.DataFrame(dup_rows)], ignore_index=True)


def generate_pull_requests(n_repos: int = N_REPOS) -> pd.DataFrame:
    """One merged PR per repository; pr_id == repo_id."""
    anchor_dates = _anchor_dates(n_repos)
    # Migration windows: anchor - 7d .. anchor + 21d (target repo extends
    # past the prediction point).
    window_start = np.where(
        np.arange(n_repos) == 5,
        pd.Timestamp("2024-06-01").to_numpy(),
        (anchor_dates - pd.Timedelta(days=7)).to_numpy(),
    )
    window_end = np.where(
        np.arange(n_repos) == 5,
        pd.Timestamp("2024-08-15").to_numpy(),
        (anchor_dates + pd.Timedelta(days=21)).to_numpy(),
    )
    merged_at = (pd.to_datetime(window_start)
                 + pd.to_timedelta(rng.integers(1, 22, size=n_repos), unit="D")
                 + pd.to_timedelta(rng.integers(0, 24, size=n_repos), unit="h"))
    created_at = anchor_dates - pd.to_timedelta(rng.integers(1, 30, size=n_repos), unit="D")
    return pd.DataFrame({
        "pr_id": np.arange(1, n_repos + 1),
        "repo_id": np.arange(1, n_repos + 1),
        "pr_number": rng.integers(100, 100000, size=n_repos),
        "title": rng.choice(PR_TITLES, size=n_repos),
        "state": rng.choice(PR_STATES, size=n_repos),
        "created_at": created_at,
        "updated_at": merged_at,
        "merged_at": merged_at,
    })


def generate_migrations(n_repos: int = N_REPOS) -> pd.DataFrame:
    """
    One migration per repository (migration_id == repo_id).
    Repos 3 and 6 end after the prediction point (leakage guard targets).
    """
    anchor_shas = np.array(_anchor_shas(n_repos), dtype=object)
    anchor_dates = _anchor_dates(n_repos)
    from_v, to_v = _versions(n_repos)
    names = _repo_names(n_repos)
    is_target = np.arange(n_repos) == 5

    window_start = np.where(is_target, pd.Timestamp("2024-06-01").to_numpy(),
                            (anchor_dates - pd.Timedelta(days=7)).to_numpy())
    window_end = np.where(is_target, pd.Timestamp("2024-08-15").to_numpy(),
                          (anchor_dates + pd.Timedelta(days=21)).to_numpy())

    grade_idx = np.arange(n_repos) % 3
    grades = np.array(EVIDENCE_GRADES, dtype=object)[grade_idx]
    # Messy grade casing on ~10% of rows (gold -> Gold, etc.).
    messy_grades = [g for g in EVIDENCE_GRADE_MESSY if g not in EVIDENCE_GRADES]
    messy_mask = rng.random(n_repos) < 0.10
    grades[messy_mask] = rng.choice(messy_grades, size=int(messy_mask.sum()))

    n_elements = rng.integers(1, 4, size=n_repos)
    n_files = rng.integers(1, 3, size=n_repos)
    return pd.DataFrame({
        "migration_id": np.arange(1, n_repos + 1),
        "repo_id": np.arange(1, n_repos + 1),
        "dependency_id": -1,                       # repaired in clean()
        "change_id": np.arange(1, n_repos + 1),
        "pr_id": np.arange(1, n_repos + 1),
        "anchor_commit_sha": anchor_shas,
        "window_start": pd.to_datetime(window_start),
        "window_end": pd.to_datetime(window_end),
        "evidence_grade": grades,
        "package_name": _packages(n_repos),
        "from_version": from_v,
        "to_version": to_v,
        "is_major_bump": True,
        "full_name": names,
        "title": [f"Upgrade {f} -> {t}" for f, t in zip(from_v, to_v)],
        "n_elements": n_elements,
        "n_files": n_files,
        "total_changed_lines": rng.integers(20, 200, size=n_repos),
        "n_distinct_symbols": rng.integers(1, 3, size=n_repos),
        "n_evidence": rng.integers(1, 3, size=n_repos),
        "mean_strength": np.round(rng.uniform(0.5, 0.9, size=n_repos), 3),
        "duration_days": rng.integers(10, 25, size=n_repos),
        "days_to_merge": rng.integers(10, 25, size=n_repos),
        "elements_per_file": np.round(n_elements / n_files, 2),
    })


def generate_dependency_changes(n_repos: int = N_REPOS) -> pd.DataFrame:
    """One dependency change per repository, linking the two states."""
    anchor_shas = np.array(_anchor_shas(n_repos), dtype=object)
    anchor_dates = _anchor_dates(n_repos)
    from_v, to_v = _versions(n_repos)
    names = _repo_names(n_repos)
    packages = _packages(n_repos)

    def major(v: str) -> int:
        return int(v.split(".")[0])

    from_major = np.array([major(v) for v in from_v], dtype=float)
    from_minor = np.array([int(v.split(".")[1]) for v in from_v], dtype=float)
    from_patch = np.array([int(v.split(".")[2]) for v in from_v], dtype=float)
    to_major = np.array([major(v) for v in to_v], dtype=float)
    to_minor = np.array([int(v.split(".")[1]) for v in to_v], dtype=float)
    to_patch = np.array([int(v.split(".")[2]) for v in to_v], dtype=float)
    is_major = (from_major != to_major)

    return pd.DataFrame({
        "change_id": np.arange(1, n_repos + 1),
        "repo_id": np.arange(1, n_repos + 1),
        "dependency_id": np.arange(1, n_repos + 1),
        "from_state_id": 2 * np.arange(n_repos) + 1,
        "to_state_id": 2 * np.arange(n_repos) + 2,
        "change_type": np.where(is_major, "version_upgrade", "specifier_change"),
        "introduced_by": rng.choice(AUTHORS[:3], size=n_repos),
        "commit_sha": anchor_shas,
        "changed_at": anchor_dates,
        "package_name": packages,
        "full_name": names,
        "role": np.array([ROLE_NEW if i == 6 else ROLE_OLD
                          for i in range(1, n_repos + 1)], dtype=object),
        "from_major": from_major,
        "from_minor": from_minor,
        "from_patch": from_patch,
        "to_major": to_major,
        "to_minor": to_minor,
        "to_patch": to_patch,
        "is_major_bump": is_major,
    })


def generate_program_elements(n_repos: int = N_REPOS,
                            per_migration: int = ELEMENTS_PER_MIGRATION) -> pd.DataFrame:
    """
    per_migration program elements per migration with weighted api_symbols.
    Messy element_type casing, NULLs, and duplicate rows (new element_ids)
    sharing the natural key (migration_id, file_path, element_name, api_symbol).
    """
    total = n_repos * per_migration
    migration_id = np.repeat(np.arange(1, n_repos + 1), per_migration)

    file_paths = rng.choice(FILE_PATHS, size=total, p=FILE_PATH_WEIGHTS)
    element_types = rng.choice(ELEMENT_TYPE_MESSY, size=total)
    element_names = rng.choice(ELEMENT_NAMES, size=total)
    api_symbols = rng.choice(API_SYMBOLS, size=total, p=_symbol_weights())

    # Injected NULLs
    file_paths[rng.random(total) < 0.10] = None
    element_names[rng.random(total) < 0.05] = None

    changed_lines = rng.integers(0, 100, size=total).astype(float)
    strength = np.round(rng.uniform(0.1, 1.0, size=total), 4)
    changed_lines[rng.random(total) < 0.08] = np.nan
    strength[rng.random(total) < 0.12] = np.nan

    rows = pd.DataFrame({
        "element_id": np.arange(1, total + 1),
        "migration_id": migration_id,
        "file_path": file_paths,
        "element_type": element_types,
        "element_name": element_names,
        "api_symbol": api_symbols,
        "n_files": rng.integers(1, 10, size=total),
        "changed_lines": changed_lines,
        "strength": strength,
    })

    # Duplicate rows with NEW surrogate IDs sharing the natural key.
    n_dup = int(n_repos * 0.06)
    dup_migs = rng.choice(np.arange(1, n_repos + 1), size=n_dup, replace=False)
    dup_rows = []
    next_id = total + 1
    for m in dup_migs:
        pool = rows[rows["migration_id"] == m]
        for _ in range(rng.integers(1, 3)):
            src = pool.iloc[int(rng.integers(0, len(pool)))]
            dup_rows.append({
                "element_id": next_id,
                "migration_id": m,
                "file_path": src["file_path"],
                "element_type": src["element_type"],
                "element_name": src["element_name"],
                "api_symbol": src["api_symbol"],
                "n_files": src["n_files"],
                "changed_lines": None,
                "strength": None,
            })
            next_id += 1
    return pd.concat([rows, pd.DataFrame(dup_rows)], ignore_index=True)


def _symbol_weights() -> np.ndarray:
    """Weighted frequencies for api_symbols (common ones appear more often)."""
    common_symbols = {"validator": 10, "dict": 8, "Config": 12, "parse_obj": 6,
                      "root_validator": 8, "BaseModel": 6, "parse_item": 5,
                      "build_invoice": 4, "validate_email": 5, "DateTime": 4,
                      "typing": 3, "logging": 3, "Optional": 5, "Field": 2,
                      "uuid": 3, "Order": 3, "User": 4, "Client": 2,
                      "Session": 3, "OrderSchema": 2, "Payload": 2,
                      "test_settings": 2, "test_client_timeout": 3,
                      "test_order_json": 3, "load_config": 3, "to_payload": 3,
                      "from_row": 2, "user": 2, "config": 2, "serializers": 2}
    weights = np.array([common_symbols.get(sym, 2) for sym in API_SYMBOLS], dtype=float)
    return weights / weights.sum()


def generate_evidence(n_repos: int = N_REPOS,
                      elements_per_migration: int = ELEMENTS_PER_MIGRATION,
                      per_element: int = EVIDENCE_PER_ELEMENT) -> pd.DataFrame:
    """
    per_element evidence rows per program element, on the same migration.
    Messy sources, NULLs, and duplicate rows with new evidence_ids.
    """
    n_elements = n_repos * elements_per_migration
    total = n_elements * per_element

    migration_id = np.repeat(np.arange(1, n_repos + 1),
                             elements_per_migration * per_element)
    anchor_shas = np.array(_anchor_shas(n_repos), dtype=object)
    source_commit = np.repeat(anchor_shas, elements_per_migration * per_element)

    file_paths = rng.choice(FILE_PATHS, size=total)
    sources = rng.choice(SOURCE_VARIANTS, size=total).astype(object)
    with_sha = rng.random(total) < 0.5
    sources[with_sha] = [s + c for s, c in
                         zip(sources[with_sha], source_commit[with_sha])]
    sources[rng.random(total) < 0.15] = None          # injected NULLs

    grades = rng.choice(EVIDENCE_GRADE_MESSY, size=total)
    strength = np.round(rng.uniform(0.1, 1.0, size=total), 4)
    strength[rng.random(total) < 0.12] = np.nan

    rows = pd.DataFrame({
        "evidence_id": np.arange(1, total + 1),
        "migration_id": migration_id,
        "file_path": file_paths,
        "source": sources,
        "raw_data": rng.choice(RAW_DATA_VARIANTS + [None], size=total),
        "source_commit_sha": source_commit,
        "collected_at": pd.Timestamp("2024-01-01") + pd.to_timedelta(
            rng.integers(0, 200, size=total), unit="D"),
        "strength": strength,
        "evidence_grade": grades,
    })

    # Duplicate rows with NEW surrogate IDs (source set to None).
    dup_rows = []
    for _ in range(int(total * 0.03)):
        src = rows.iloc[int(rng.integers(0, len(rows)))]
        dup_rows.append({**src.to_dict(), "evidence_id": len(rows) + len(dup_rows) + 1,
                         "source": None})
    return pd.concat([rows, pd.DataFrame(dup_rows)], ignore_index=True)


def generate_candidates(n_candidates: int = N_CANDIDATES) -> pd.DataFrame:
    """
    Static + historical_expansion candidates for the target repo (6).
    Static rows carry a static_score (higher for observed positives);
    historical_expansion rows have static_score NULL and an evidence_id.
    """
    n = n_candidates
    is_hist = rng.random(n) < 0.30
    observed = rng.random(n) < 0.30

    static_score = np.where(observed, rng.uniform(0.5, 0.9, size=n),
                            rng.uniform(0.1, 0.5, size=n))
    static_score[rng.random(n) < 0.05] = np.nan       # injected NULLs
    static_score[is_hist] = np.nan                    # hist rows have no static score

    n_evidence = int(N_REPOS * ELEMENTS_PER_MIGRATION * EVIDENCE_PER_ELEMENT * 1.03)
    evidence_id = np.full(n, np.nan)
    evidence_id[is_hist] = rng.integers(1, n_evidence, size=int(is_hist.sum()))

    api_symbols = rng.choice(API_SYMBOLS, size=n, p=_symbol_weights())
    element_types = rng.choice(["function", "class", "method", "test"],
                               size=n).astype(object)
    messy_mask = rng.random(n) < 0.20
    element_types[messy_mask] = rng.choice(ELEMENT_TYPE_MESSY,
                                           size=int(messy_mask.sum()))

    return pd.DataFrame({
        "candidate_id": np.arange(1, n + 1),
        "target_repo_id": 6,
        "target_change_id": 6,
        "prediction_point": PREDICTION_POINT,
        "file_path": rng.choice(FILE_PATHS[:7], size=n),
        "element_type": element_types,
        "element_name": rng.choice(ELEMENT_NAMES, size=n),
        "api_symbol": api_symbols,
        "generated_by": np.where(is_hist, "historical_expansion", "static"),
        "static_score": static_score,
        "evidence_id": evidence_id,
        "observed_label": observed,
        "hist_migrations": rng.integers(0, 3, size=n),
        "hist_repos": rng.integers(0, 3, size=n),
        "hist_mean_strength": np.round(rng.uniform(0.3, 0.8, size=n), 3),
        "hist_freq": np.round(rng.uniform(0.1, 1.0, size=n), 3),
        "hist_score": np.round(rng.uniform(0.3, 0.8, size=n), 3),
        "n_eligible_migrations": N_REPOS - 2,   # repos 3 and 6 end after the cutoff
        "score_static_only": np.round(rng.uniform(0.3, 0.8, size=n), 3),
        "score_static_plus_hist": np.round(rng.uniform(0.3, 0.8, size=n), 3),
        "rank_static_only": rng.integers(1, n, size=n),
        "rank_static_plus_hist": rng.integers(1, n, size=n),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Generate all 10 raw CSV tables and print row counts."""
    print(f"Generating Navtac synthetic data (seed={SEED}, {N_REPOS} repos)...")

    writers = [
        ("repositories", lambda: generate_repositories()),
        ("commits", lambda: generate_commits()),
        ("dependencies", lambda: generate_dependencies()),
        ("dependency_changes", lambda: generate_dependency_changes()),
        ("dependency_states", lambda: generate_dependency_states()),
        ("pull_requests", lambda: generate_pull_requests()),
        ("migrations", lambda: generate_migrations()),
        ("program_elements", lambda: generate_program_elements()),
        ("evidence", lambda: generate_evidence()),
        ("candidates", lambda: generate_candidates()),
    ]
    for name, gen in writers:
        print(f"  {name}...")
        gen().to_csv(RAW_DIR / f"{name}.csv", index=False)

    print("\nData generation complete. Row counts:")
    for f in sorted(RAW_DIR.glob("*.csv")):
        print(f"  {f.name}: {sum(1 for _ in f.open()) - 1} rows")
    print(f"\nAll CSVs written to {RAW_DIR}")


if __name__ == "__main__":
    main()
