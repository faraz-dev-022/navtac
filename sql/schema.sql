-- Navtac schema.sql
-- Re-runnable migration script: drop tables in reverse FK order first,
-- then create tables in dependency order.

-- Drop in reverse FK order
DROP TABLE IF EXISTS candidates;
DROP TABLE IF EXISTS evidence;
DROP TABLE IF EXISTS program_elements;
DROP TABLE IF EXISTS migrations;
DROP TABLE IF EXISTS dependency_changes;
DROP TABLE IF EXISTS dependency_states;
DROP TABLE IF EXISTS pull_requests;
DROP TABLE IF EXISTS commits;
DROP TABLE IF EXISTS dependencies;
DROP TABLE IF EXISTS repositories;

-- ============================================================
-- repositories
-- ============================================================
CREATE TABLE repositories (
    id          SERIAL PRIMARY KEY,
    full_name   TEXT    NOT NULL,
    role        TEXT    NOT NULL,
    created_at  TIMESTAMP WITHOUT TIME ZONE,
    updated_at  TIMESTAMP WITHOUT TIME ZONE,
    CONSTRAINT chk_repositories_role
        CHECK (role IN ('historical', 'target'))
);

CREATE UNIQUE INDEX idx_repositories_full_name
    ON repositories (full_name);

-- ============================================================
-- dependencies
-- ============================================================
CREATE TABLE dependencies (
    id              SERIAL PRIMARY KEY,
    repo_id         INTEGER NOT NULL,
    package_name    TEXT    NOT NULL,
    version         TEXT,
    CONSTRAINT fk_dependencies_repo
        FOREIGN KEY (repo_id) REFERENCES repositories (id)
        ON DELETE CASCADE,
    CONSTRAINT uq_dependencies_package_name
        UNIQUE (package_name),
    CONSTRAINT uq_dependencies_repo_package
        UNIQUE (repo_id, package_name)
);

CREATE INDEX idx_dependencies_repo_id
    ON dependencies (repo_id);

-- ============================================================
-- commits
-- ============================================================
CREATE TABLE commits (
    id            SERIAL PRIMARY KEY,
    repo_id       INTEGER NOT NULL,
    commit_sha    CHAR(40)    NOT NULL,
    author_name   TEXT,
    author_email  TEXT,
    message       TEXT,
    authored_at   TIMESTAMP WITHOUT TIME ZONE,
    CONSTRAINT fk_commits_repo
        FOREIGN KEY (repo_id) REFERENCES repositories (id)
        ON DELETE CASCADE
);

CREATE INDEX idx_commits_repo_id
    ON commits (repo_id);

-- ============================================================
-- dependency_states
-- ============================================================
CREATE TABLE dependency_states (
    id             SERIAL PRIMARY KEY,
    commit_id      INTEGER NOT NULL,
    dependency_id  INTEGER NOT NULL,
    observed_at    TEXT NOT NULL,
    version        TEXT,
    CONSTRAINT fk_dependency_states_commit
        FOREIGN KEY (commit_id) REFERENCES commits (id)
        ON DELETE CASCADE,
    CONSTRAINT fk_dependency_states_dependency
        FOREIGN KEY (dependency_id) REFERENCES dependencies (id)
        ON DELETE CASCADE
);

CREATE INDEX idx_dependency_states_commit_id
    ON dependency_states (commit_id);

CREATE INDEX idx_dependency_states_dependency_id
    ON dependency_states (dependency_id);

-- ============================================================
-- dependency_changes
-- ============================================================
CREATE TABLE dependency_changes (
    id              SERIAL PRIMARY KEY,
    repo_id         INTEGER NOT NULL,
    change_type     TEXT    NOT NULL,
    introduced_by   TEXT,
    CHANGE_ORDER    INTEGER,
    CONSTRAINT fk_dependency_changes_repo
        FOREIGN KEY (repo_id) REFERENCES repositories (id)
        ON DELETE CASCADE,
    CONSTRAINT chk_dependency_changes_change_type
        CHECK (change_type IN ('major', 'minor', 'patch', 'replacement', 'unknown'))
);

CREATE INDEX idx_dependency_changes_repo_id
    ON dependency_changes (repo_id);

-- ============================================================
-- pull_requests
-- ============================================================
CREATE TABLE pull_requests (
    id          SERIAL PRIMARY KEY,
    repo_id     INTEGER NOT NULL,
    pr_number   INTEGER NOT NULL,
    title       TEXT,
    state       TEXT    NOT NULL,
    created_at  TIMESTAMP WITHOUT TIME ZONE,
    updated_at  TIMESTAMP WITHOUT TIME ZONE,
    CONSTRAINT fk_pull_requests_repo
        FOREIGN KEY (repo_id) REFERENCES repositories (id)
        ON DELETE CASCADE,
    CONSTRAINT chk_pull_requests_state
        CHECK (state IN ('open', 'closed', 'merged', 'pending'))
);

CREATE UNIQUE INDEX idx_pull_requests_repo_pr_number
    ON pull_requests (repo_id, pr_number);

CREATE INDEX idx_pull_requests_repo_id
    ON pull_requests (repo_id);

-- ============================================================
-- migrations
-- ============================================================
CREATE TABLE migrations (
    id                SERIAL PRIMARY KEY,
    repo_id           INTEGER NOT NULL,
    dependency_id     INTEGER NOT NULL,
    change_id         INTEGER NOT NULL,
    migration_date    TIMESTAMP WITHOUT TIME ZONE,
    evidence_grade    TEXT    NOT NULL,
    CONSTRAINT fk_migrations_repo
        FOREIGN KEY (repo_id) REFERENCES repositories (id)
        ON DELETE CASCADE,
    CONSTRAINT fk_migrations_dependency
        FOREIGN KEY (dependency_id) REFERENCES dependencies (id)
        ON DELETE CASCADE,
    CONSTRAINT fk_migrations_change
        FOREIGN KEY (change_id) REFERENCES dependency_changes (id)
        ON DELETE CASCADE,
    CONSTRAINT chk_migrations_evidence_grade
        CHECK (evidence_grade IN ('A', 'B', 'C', 'D', 'F'))
);

CREATE INDEX idx_migrations_repo_id
    ON migrations (repo_id);

CREATE INDEX idx_migrations_dependency_id
    ON migrations (dependency_id);

CREATE INDEX idx_migrations_change_id
    ON migrations (change_id);

-- ============================================================
-- program_elements
-- ============================================================
CREATE TABLE program_elements (
    id              SERIAL PRIMARY KEY,
    migration_id    INTEGER NOT NULL,
    element_type    TEXT    NOT NULL,
    name            TEXT    NOT NULL,
    version         TEXT,
    location        TEXT,
    CONSTRAINT fk_program_elements_migration
        FOREIGN KEY (migration_id) REFERENCES migrations (id)
        ON DELETE CASCADE
);

CREATE INDEX idx_program_elements_migration_id
    ON program_elements (migration_id);

-- ============================================================
-- evidence
-- ============================================================
CREATE TABLE evidence (
    id              SERIAL PRIMARY KEY,
    migration_id    INTEGER NOT NULL,
    source          TEXT    NOT NULL,
    raw_data        TEXT,
    collected_at    TIMESTAMP WITHOUT TIME ZONE,
    CONSTRAINT fk_evidence_migration
        FOREIGN KEY (migration_id) REFERENCES migrations (id)
        ON DELETE CASCADE
);

CREATE INDEX idx_evidence_migration_id
    ON evidence (migration_id);

-- ============================================================
-- candidates
-- ============================================================
CREATE TABLE candidates (
    id                  SERIAL PRIMARY KEY,
    evidence_id         INTEGER NOT NULL,
    score               NUMERIC(10, 6),
    confidence          NUMERIC(10, 6),
    generated_by        TEXT    NOT NULL,
    CONSTRAINT fk_candidates_evidence
        FOREIGN KEY (evidence_id) REFERENCES evidence (id)
        ON DELETE CASCADE,
    CONSTRAINT chk_candidates_generated_by
        CHECK (generated_by IN ('static_analysis', 'dynamic_analysis', 'ml_model', 'manual_review'))
);

CREATE INDEX idx_candidates_evidence_id
    ON candidates (evidence_id);

