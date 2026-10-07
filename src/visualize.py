#!/usr/bin/env python3
"""NAVTAC ETL visualisation.

Reads the three finished ETL mart tables from data/parquet and renders one
matplotlib figure per chart into reports/figures/.  Each title states the
question or finding that the chart answers.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# --------------------------------------------------------------- colour palette
# Master colour-blind-safe palette:
#   blue    #4C72B0 = static only
#   orange  #DD8452 = static + historical
#   teal    #2A9D8F = positive / eligible
#   grey    #9AA5B1 = other
#   red     #C0392B = problem / excluded
BLUE = "#4C72B0"
ORANGE = "#DD8452"
TEAL = "#2A9D8F"
GREY = "#9AA5B1"
RED = "#C0392B"

# ------------------------------------------------------------- file layout
PARQUET = Path("data/parquet")
RAW = Path("data/raw")
FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------- global style
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "axes.titleweight": "bold",
        "font.size": 10,
    }
)

FONTSIZE = 10  # every title, label and tick is at or above 8 pt

# The marts hold 100k+ rows; a per-migration bar chart cannot draw
# every row, so migration-indexed figures render a deterministic,
# evenly spaced sample (extremes and post-cutoff rows always kept).
MAX_SAMPLE_ROWS = 16
MAX_SAMPLE_VALUES = 2000


def _sample_rows(d: pd.DataFrame, n: int = MAX_SAMPLE_ROWS,
                 keep_index=()) -> pd.DataFrame:
    """Deterministic evenly spaced sample of at most n rows.

    The caller sorts d first: the first and last rows (the extremes)
    are always kept, as are any rows whose index appears in keep_index
    (e.g. migrations whose window reaches past the prediction point).
    """
    if len(d) <= n:
        return d
    idx = set(np.linspace(0, len(d) - 1, n).astype(int))
    idx.update(keep_index)
    return d.loc[sorted(idx)].reset_index(drop=True)


def _sample_values(s: pd.Series, n: int = MAX_SAMPLE_VALUES) -> np.ndarray:
    """Deterministic evenly spaced sample of a Series' values."""
    if len(s) <= n:
        return s.to_numpy()
    return s.iloc[np.linspace(0, len(s) - 1, n).astype(int)].to_numpy()


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three finished ETL mart tables from data/parquet.

    Returns:
        (migration_stats, candidate_features, experiment_metrics) DataFrames.
    """
    migration_stats = pd.read_parquet(PARQUET / "migration_stats.parquet")
    candidate_features = pd.read_parquet(PARQUET / "candidate_features.parquet")
    experiment_metrics = pd.read_parquet(PARQUET / "experiment_metrics.parquet")
    return migration_stats, candidate_features, experiment_metrics


def save(fig: plt.Figure, name: str, tight: bool = True) -> None:
    """Fixed save routine: tight_layout, dpi 150, bbox_inches tight, close.

    Pass tight=False for figures that manage their own layout (the dashboard
    tight_layouts with a rect to clear its suptitle; a second plain
    tight_layout() would override that rect).
    """
    if tight:
        fig.tight_layout()
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ======================================================================
# Chart functions
# ======================================================================


def fig_migration_lifecycle() -> None:
    """Do uncertain-evidence migrations take longer to close than gold/silver?"""
    migration_stats, candidate_features, _ = load()
    m = migration_stats
    order = ["gold", "silver", "uncertain"]
    means = [float(m.loc[m["evidence_grade"] == g, "duration_days"].mean()) for g in order]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(order, means, color=[TEAL, GREY, RED], width=0.55)
    ax.set_title(
        "Uncertain-grade migrations take longer to close than gold/silver"
        f"  (mean {means[2]:.1f} d vs {means[0]:.1f} d gold)"
    )
    ax.set_ylabel("Mean duration to merge (days)")
    ax.set_xlabel("Evidence grade")
    ax.set_ylim(0, max(means) * 1.25 if means else 1.0)
    ax.tick_params(labelsize=FONTSIZE)
    save(fig, "07_migration_lifecycle")


def fig_eligible_migration_breakdown() -> None:
    """How many eligible historical migrations support each candidate?"""
    _, candidate_features, _ = load()
    counts = candidate_features["hist_migrations"].value_counts().sort_index()
    labels = [str(int(i)) if not np.isnan(i) else "NA" for i in counts.index]
    vals = counts.values.astype(float)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(labels, vals, color=TEAL, width=0.55)
    ax.set_title("Eligible historical migrations per candidate (static arm)")
    ax.set_xlabel("Eligible historical migrations")
    ax.set_ylabel("Number of candidates")
    ax.tick_params(labelsize=FONTSIZE)
    save(fig, "08_eligible_migration_breakdown")


def fig_candidate_score_arm() -> None:
    """Does adding the historical signal shift candidate scores?"""
    _, candidate_features, _ = load()
    static_arm = candidate_features.loc[
        candidate_features["generated_by"] == "static", "score_static_only"
    ].dropna()
    plus_hist_arm = candidate_features.loc[
        candidate_features["generated_by"] == "static", "score_static_plus_hist"
    ].dropna()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    data = [static_arm.values, plus_hist_arm.values]
    bp = ax.boxplot(data, patch_artist=True, widths=0.45)
    ax.set_xticklabels(["Static only", "Static + historical"], fontsize=FONTSIZE)
    for patch, color in zip(bp["boxes"], [BLUE, ORANGE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title("Score distribution by candidate arm")
    ax.set_ylabel("Review priority score")
    ax.set_xlabel("Experiment arm")
    ax.tick_params(labelsize=FONTSIZE)
    save(fig, "09_candidate_score_arm")


def fig_rank_gap() -> None:
    """Where does the historical-augmented score lift/lower a candidate?"""
    _, candidate_features, _ = load()
    gap = (
        candidate_features["score_static_plus_hist"]
        - candidate_features["score_static_only"]
    ).dropna()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if gap.empty:
        ax.text(0.5, 0.5, "No paired candidates", ha="center", va="center", fontsize=FONTSIZE)
        ax.tick_params(labelsize=FONTSIZE)
        save(fig, "10_rank_gap")
        return
    positive = _sample_values(gap[gap > 0])
    negative = _sample_values(gap[gap < 0])
    n_pos = int(positive.size)
    n_neg = int(negative.size)
    if n_pos:
        ax.bar(np.arange(n_pos), positive, color=TEAL, width=0.6, label="score increases")
    if n_neg:
        ax.bar(np.arange(n_neg) + n_pos + 1, negative, color=RED, width=0.6, label="score decreases")
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_title("Historical-augmented score change for candidates")
    ax.set_xlabel("Candidate (ordered)")
    ax.set_ylabel("Change in score")
    ax.tick_params(labelsize=FONTSIZE)
    ax.legend(frameon=False, fontsize=FONTSIZE)
    save(fig, "10_rank_gap")


def fig_dependency_chain() -> None:
    """Which years saw the most dependency-change events tracked?"""
    # The dependency_changes mart is not part of load()'s 3-DataFrame return, so
    # read it directly from data/parquet (per the single-read-source rule).
    enriched = pd.read_parquet(PARQUET / "dependency_changes_enriched.parquet")
    yearly = (
        pd.to_datetime(enriched["changed_at"], errors="coerce")
        .dt.to_period("Y")
        .value_counts()
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([str(p) for p in yearly.index], yearly.values.astype(float), color=BLUE, width=0.6)
    ax.set_title("Dependency change events recorded per year")
    ax.set_xlabel("Year of change")
    ax.set_ylabel("Change events")
    ax.tick_params(labelsize=FONTSIZE)
    save(fig, "11_dependency_chain")


def fig_experiment_metrics() -> None:
    """Does static+historical beat static-only at precision and recall@5?"""
    _, _, experiment_metrics = load()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    arms = experiment_metrics["arm"].tolist()
    precision = experiment_metrics["precision@5"].tolist()
    recall = experiment_metrics["recall@5"].tolist()
    x = np.arange(len(arms))
    ax.bar(x - 0.18, precision, width=0.35, color=BLUE, label="Precision@5")
    ax.bar(x + 0.18, recall, width=0.35, color=ORANGE, label="Recall@5")
    ax.set_title("Static-only vs static+historical arms at precision@5 and recall@5")
    ax.set_ylabel("Fraction (recall denominator: all true positives)")
    ax.set_xlabel("Experiment arm")
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.legend(frameon=False, fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    save(fig, "12_experiment_metrics")


def fig_migration_size(m: pd.DataFrame) -> None:
    """How large is each migration's observed surface (elements and files)?"""
    d = _sample_rows(m.sort_values("n_elements", ascending=True).reset_index(drop=True))
    labels = (d["full_name"] + "\n" + d["package_name"]).tolist()
    y = np.arange(len(d))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8, max(4.2, 0.95 * len(d) + 1.6)))
    b1 = ax.barh(y - h / 2, d["n_elements"], height=h, color=BLUE, label="n_elements")
    b2 = ax.barh(y + h / 2, d["n_files"], height=h, color=ORANGE, label="n_files")
    vmax = float(pd.concat([d["n_elements"], d["n_files"]], ignore_index=True).max())
    ax.set_xlim(0, vmax * 1.15 if vmax > 0 else 1.0)   # headroom so end-of-bar values are not cut off
    ax.bar_label(b1, labels=[str(int(v)) for v in d["n_elements"]], padding=3, fontsize=FONTSIZE)
    ax.bar_label(b2, labels=[f"{v:g}" for v in d["n_files"]], padding=3, fontsize=FONTSIZE)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FONTSIZE)
    ax.set_title("Size of each migration's observed surface")
    ax.set_xlabel("count")
    ax.set_ylabel("Migration (repository / package)")
    ax.legend(frameon=False, fontsize=FONTSIZE, loc="lower right")
    ax.tick_params(labelsize=FONTSIZE)
    save(fig, "01_migration_size")


def fig_timeline(m: pd.DataFrame, c: pd.DataFrame) -> None:
    """When does each migration open and close relative to the prediction point?"""
    d = m.sort_values("window_start", ascending=True).reset_index(drop=True)
    pp = pd.to_datetime(c["prediction_point"].iloc[0])
    pp_num = mdates.date2num(pp)

    # Keep every migration whose window reaches the cutoff (red) plus an
    # evenly spaced sample of the rest, so the leakage cases stay visible.
    post_cutoff = d.index[(d["window_end"].notna()) & (d["window_end"] >= pp)]
    d = _sample_rows(d, keep_index=post_cutoff)

    colours = []
    for _, row in d.iterrows():
        if pd.notna(row["window_end"]) and row["window_end"] >= pp:
            colours.append(RED)          # leakage: window reaches the cutoff
        elif str(row["package_name"]) == "pydantic":
            colours.append(TEAL)         # pydantic, closes before the cutoff
        else:
            colours.append(GREY)         # another package

    y = np.arange(len(d))
    left = mdates.date2num(d["window_start"])
    width = mdates.date2num(d["window_end"]) - left
    fig, ax = plt.subplots(figsize=(8, max(4.2, 0.75 * len(d) + 1.8)))
    ax.barh(y, width, left=left, height=0.55, color=colours)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r.full_name} ({r.package_name})" for r in d.itertuples()], fontsize=FONTSIZE)
    ax.set_ylim(-0.7, len(d) + 0.3)
    ax.axvline(pp_num, color="#666666", linestyle="--", linewidth=1.2)
    ax.text(pp_num - 2, len(d) - 0.25, "prediction point", ha="right", va="top",
            fontsize=FONTSIZE, color="#666666")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.set_title("Migration windows relative to the candidates' prediction point")
    ax.set_xlabel("time")
    ax.set_ylabel("Migration")
    ax.legend(
        handles=[Patch(color=TEAL, label="pydantic, closes before cutoff"),
                 Patch(color=GREY, label="other package"),
                 Patch(color=RED, label="closes at/after prediction point (leakage)")],
        frameon=False, fontsize=FONTSIZE, loc="lower right")
    ax.tick_params(labelsize=FONTSIZE)
    save(fig, "02_timeline_cutoff")


def fig_symbol_freq(c: pd.DataFrame) -> None:
    """Which APIs were historically migrated, and how often relative to N?"""
    d = (
        c[c["api_symbol"].notna()]
        .groupby("api_symbol")
        .first()[["hist_migrations", "hist_freq"]]
        .query("hist_migrations > 0")
        .sort_values(["hist_freq", "hist_migrations"], ascending=True)
        .reset_index()
    )
    n = int(pd.to_numeric(c["n_eligible_migrations"], errors="coerce").dropna().iloc[0])

    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, max(4.2, 0.55 * len(d) + 1.8)))
    bars = ax.barh(y, d["hist_freq"], height=0.6, color=TEAL)
    ax.set_xlim(0, 1.05)
    # A hist_freq > 1.0 (e.g. 1.25) exceeds the capped x axis; annotation_clip
    # would suppress that bar's label entirely, so disable it and keep the
    # label visible just past the axis edge instead.
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in d["hist_freq"]],
                 padding=3, fontsize=FONTSIZE, annotation_clip=False)
    for t in ax.texts:
        t.set_clip_on(False)
    ax.set_yticks(y)
    ax.set_yticklabels(d["api_symbol"], fontsize=FONTSIZE)
    ax.set_title("Which APIs were historically migrated?")
    ax.set_xlabel(f"share of the {n} eligible pydantic migrations that changed this API")
    ax.set_ylabel("API symbol")
    ax.tick_params(labelsize=FONTSIZE)
    save(fig, "03_symbol_frequency")


def fig_score_distribution(c: pd.DataFrame) -> None:
    """Does the static score separate observed from unobserved candidates?"""
    s = (
        c.loc[c["generated_by"] == "static", ["static_score", "observed_label"]]
        .dropna(subset=["static_score"])
    )
    edges = np.linspace(0.0, 1.0, 11)   # 11 bin edges from 0 to 1 -> 10 bins
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(s.loc[~s["observed_label"].astype(bool), "static_score"],
            bins=edges, alpha=0.75, color=GREY, label="not in observed surface")
    ax.hist(s.loc[s["observed_label"].astype(bool), "static_score"],
            bins=edges, alpha=0.75, color=TEAL, label="in observed surface")
    ax.set_title("Static score overlaps heavily: it separates the classes only weakly")
    ax.set_xlabel("static score")
    ax.set_ylabel("candidates")
    ax.legend(frameon=False, fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    save(fig, "04_static_score_distribution")


def fig_recall_curve(c: pd.DataFrame) -> None:
    """How much of the observed surface has a developer found after K reviews?"""
    total_pos = int(c["observed_label"].astype(bool).sum())
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for arm, colour, label in (("static_only", BLUE, "static only"),
                               ("static_plus_hist", ORANGE, "static + historical")):
        pool = c.loc[c[f"score_{arm}"].notna()].sort_values(f"rank_{arm}")
        recall = pool["observed_label"].astype(bool).cumsum() / total_pos
        ax.step(np.arange(1, len(recall) + 1), recall, where="post", lw=2,
                color=colour, label=label)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("K (items a developer reviews, top of queue first)")
    ax.set_ylabel(f"Recall@K (of {total_pos} observed locations)")
    ax.set_title(
        f"Static + historical surfaces all {total_pos} observed locations; "
        f"static-only caps at 12/{total_pos}"
    )
    ax.legend(frameon=False, fontsize=FONTSIZE, loc="lower right")
    ax.tick_params(labelsize=FONTSIZE)
    save(fig, "05_recall_at_k_curve")


def fig_metrics(e: pd.DataFrame) -> None:
    """How do the two arms compare on recall and precision at fixed budgets?"""
    metrics = ["candidate_recall", "recall@5", "recall@10", "recall@15",
               "precision@10", "mrr"]
    labels = ["candidate\nrecall", "recall@5", "recall@10", "recall@15",
              "precision@10", "mrr"]
    x = np.arange(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, (arm, colour, label) in enumerate(
            (("static_only", BLUE, "static only"),
             ("static_plus_hist", ORANGE, "static + historical"))):
        row = e.loc[e["arm"] == arm].iloc[0]
        vals = row[metrics].astype(float)
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w, color=colour, label=label)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=2, fontsize=FONTSIZE)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONTSIZE)
    ax.set_title("Experiment metrics: static-only vs static + historical")
    ax.set_ylabel("score")
    # Single-row legend so the upper-left slot clears the tall first bars.
    ax.legend(frameon=False, fontsize=FONTSIZE, loc="upper left", ncols=2)
    ax.tick_params(labelsize=FONTSIZE)
    save(fig, "06_metrics_comparison")


def fig_dashboard(m: pd.DataFrame, c: pd.DataFrame, e: pd.DataFrame) -> None:
    """One-page 2x3 summary of key charts (mirrors 01/02/03/04/05/06)."""
    small = 8  # tick/label size inside the small panels (floor of the 8 pt rule)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))

    # (0, 0) migration size --------------------------------------------
    ax = axes[0, 0]
    d = _sample_rows(m.sort_values("n_elements", ascending=True).reset_index(drop=True))
    y = np.arange(len(d))
    h = 0.38
    b1 = ax.barh(y - h / 2, d["n_elements"], height=h, color=BLUE, label="n_elements")
    b2 = ax.barh(y + h / 2, d["n_files"], height=h, color=ORANGE, label="n_files")
    ax.bar_label(b1, labels=[str(int(v)) for v in d["n_elements"]], padding=2, fontsize=small)
    ax.bar_label(b2, labels=[f"{v:g}" for v in d["n_files"]], padding=2, fontsize=small)
    vmax = float(pd.concat([d["n_elements"], d["n_files"]], ignore_index=True).max())
    ax.set_xlim(0, vmax * 1.2 if vmax > 0 else 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(d["full_name"], fontsize=small)
    ax.set_title("Migration size", fontsize=FONTSIZE)
    ax.legend(fontsize=small, frameon=False, loc="lower right")
    ax.tick_params(labelsize=small)

    # (0, 1) timeline with cutoff (red = excluded) ----------------------
    ax = axes[0, 1]
    d = m.sort_values("window_start", ascending=True).reset_index(drop=True)
    pp = pd.to_datetime(c["prediction_point"].iloc[0])
    pp_num = mdates.date2num(pp)
    # Keep every post-cutoff migration (red) plus an even sample.
    post_cutoff = d.index[(d["window_end"].notna()) & (d["window_end"] >= pp)]
    d = _sample_rows(d, keep_index=post_cutoff)
    colours = [
        RED if (pd.notna(r["window_end"]) and r["window_end"] >= pp)
        else (TEAL if str(r["package_name"]) == "pydantic" else GREY)
        for _, r in d.iterrows()
    ]
    left = mdates.date2num(d["window_start"])
    width = mdates.date2num(d["window_end"]) - left
    ax.barh(np.arange(len(d)), width, left=left, height=0.55, color=colours)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["full_name"], fontsize=small)
    ax.set_ylim(-0.7, len(d) + 0.3)
    ax.axvline(pp_num, color="#666666", linestyle="--", linewidth=1.0)
    ax.text(pp_num - 2, len(d) - 0.2, "prediction point", ha="right", va="top",
            fontsize=small, color="#666666")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.set_title("Timeline vs cutoff (red = excluded)", fontsize=FONTSIZE)
    ax.legend(
        handles=[Patch(color=TEAL, label="pydantic, closes before cutoff"),
                 Patch(color=GREY, label="other package"),
                 Patch(color=RED, label="excluded (leakage)")],
        frameon=False, fontsize=small, loc="lower right")
    ax.tick_params(labelsize=small)

    # (0, 2) historical API frequency -----------------------------------
    ax = axes[0, 2]
    d = (
        c[c["api_symbol"].notna()]
        .groupby("api_symbol")
        .first()[["hist_migrations", "hist_freq"]]
        .query("hist_migrations > 0")
        .sort_values("hist_freq", ascending=True)
        .reset_index()
    )
    bars = ax.barh(np.arange(len(d)), d["hist_freq"], height=0.6, color=TEAL)
    ax.set_xlim(0, 1.05)
    # hist_freq > 1.0 can pass the axis edge; keep its % label visible (as 03).
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in d["hist_freq"]],
                 padding=2, fontsize=small, annotation_clip=False)
    for t in ax.texts:
        t.set_clip_on(False)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["api_symbol"], fontsize=small)
    ax.set_title("Historical API frequency", fontsize=FONTSIZE)
    ax.set_xlabel("share of eligible migrations", fontsize=small)
    ax.tick_params(labelsize=small)

    # (1, 0) static score by label --------------------------------------
    ax = axes[1, 0]
    s = (
        c.loc[c["generated_by"] == "static", ["static_score", "observed_label"]]
        .dropna(subset=["static_score"])
    )
    edges = np.linspace(0.0, 1.0, 11)
    ax.hist(s.loc[~s["observed_label"].astype(bool), "static_score"],
            bins=edges, alpha=0.75, color=GREY, label="not in observed surface")
    ax.hist(s.loc[s["observed_label"].astype(bool), "static_score"],
            bins=edges, alpha=0.75, color=TEAL, label="in observed surface")
    ax.set_title("Static score by label", fontsize=FONTSIZE)
    ax.set_xlabel("static score", fontsize=small)
    ax.set_ylabel("candidates", fontsize=small)
    ax.legend(fontsize=small, frameon=False)
    ax.tick_params(labelsize=small)

    # (1, 1) Recall@K curve ----------------------------------------------
    ax = axes[1, 1]
    total_pos = int(c["observed_label"].astype(bool).sum())
    for arm, colour, label in (("static_only", BLUE, "static only"),
                               ("static_plus_hist", ORANGE, "static + historical")):
        pool = c.loc[c[f"score_{arm}"].notna()].sort_values(f"rank_{arm}")
        recall = pool["observed_label"].astype(bool).cumsum() / total_pos
        ax.step(np.arange(1, len(recall) + 1), recall, where="post", lw=2,
                color=colour, label=label)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("K (items reviewed)", fontsize=small)
    ax.set_ylabel(f"Recall@K (of {total_pos})", fontsize=small)
    ax.set_title("Recall@K curve", fontsize=FONTSIZE)
    ax.legend(fontsize=small, frameon=False, loc="lower right")
    ax.tick_params(labelsize=small)

    # (1, 2) key metrics: candidate recall, recall@10, MRR ---------------
    ax = axes[1, 2]
    metrics = ["candidate_recall", "recall@10", "mrr"]
    x = np.arange(len(metrics))
    w = 0.38
    for i, (arm, colour, label) in enumerate(
            (("static_only", BLUE, "static only"),
             ("static_plus_hist", ORANGE, "static + historical"))):
        vals = e.loc[e["arm"] == arm].iloc[0][metrics].astype(float)
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w, color=colour, label=label)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=2, fontsize=small)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(["candidate\nrecall", "recall@10", "mrr"], fontsize=small)
    ax.set_title("Key metrics", fontsize=FONTSIZE)
    ax.legend(fontsize=small, frameon=False, loc="upper left", ncols=2)
    ax.tick_params(labelsize=small)

    fig.suptitle("NAVTAC ETL dashboard (synthetic data)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))   # leave the top strip for the suptitle
    save(fig, "13_dashboard", tight=False)


def main() -> None:
    """Render every chart into reports/figures/."""
    m, c, e = load()
    fig_migration_lifecycle()
    fig_eligible_migration_breakdown()
    fig_candidate_score_arm()
    fig_rank_gap()
    fig_dependency_chain()
    fig_experiment_metrics()
    fig_migration_size(m)
    fig_timeline(m, c)
    fig_symbol_freq(c)
    fig_score_distribution(c)
    fig_recall_curve(c)
    fig_metrics(e)
    fig_dashboard(m, c, e)


if __name__ == "__main__":
    migration_stats, candidate_features, experiment_metrics = load()

    print(
        "migration_stats:      ", migration_stats.shape,
        f"({migration_stats['evidence_grade'].nunique()} evidence grades, "
        f"{migration_stats['repo_id'].nunique()} repos)",
    )
    print(
        "candidate_features:   ", candidate_features.shape,
        f"({candidate_features['generated_by'].nunique()} arms, "
        f"avg {(candidate_features[['hist_migrations','hist_repos','hist_freq','hist_score']].mean(axis=1)).mean():.3f} mean hist score)",
    )
    print(
        "experiment_metrics:   ", experiment_metrics.shape,
        "arms:", experiment_metrics['arm'].tolist(),
    )
    main()
