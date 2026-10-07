# NAVTAC ETL — figure index

Every PNG in this directory, with the `src/visualize.py` function that creates
it and the `main()` call order (order matches the numbering).

| # | File | Created by | Question the chart answers |
|---|------|------------|----------------------------|
| 01 | `01_migration_size.png` | `fig_migration_size(m)` | How large is each migration's observed surface (elements and files)? |
| 02 | `02_timeline_cutoff.png` | `fig_timeline(m, c)` | When does each migration open and close relative to the prediction point? Red bar = closes at/after the cutoff (leakage). |
| 03 | `03_symbol_frequency.png` | `fig_symbol_freq(c)` | Which APIs were historically migrated, and how often relative to the eligible migrations? |
| 04 | `04_static_score_distribution.png` | `fig_score_distribution(c)` | Does the static score separate observed from unobserved candidates? |
| 05 | `05_recall_at_k_curve.png` | `fig_recall_curve(c)` | How much of the observed surface has a developer found after K reviews? |
| 06 | `06_metrics_comparison.png` | `fig_metrics(e)` | How do the two arms compare on candidate recall, recall@K, precision@10 and MRR? |
| 07 | `07_migration_lifecycle.png` | `fig_migration_lifecycle()` | Do uncertain-evidence migrations take longer to close than gold/silver? |
| 08 | `08_eligible_migration_breakdown.png` | `fig_eligible_migration_breakdown()` | How many eligible historical migrations support each candidate? |
| 09 | `09_candidate_score_arm.png` | `fig_candidate_score_arm()` | Does adding the historical signal shift candidate scores? |
| 10 | `10_rank_gap.png` | `fig_rank_gap()` | Where does the historical-augmented score lift or lower a candidate? |
| 11 | `11_dependency_chain.png` | `fig_dependency_chain()` | Which years saw the most dependency-change events tracked? |
| 12 | `12_experiment_metrics.png` | `fig_experiment_metrics()` | Does static+historical beat static-only at precision@5 and recall@5? |
| 13 | `13_dashboard.png` | `fig_dashboard(m, c, e)` | One-page 2x3 summary mirroring charts 01, 02, 03, 04, 05 and 06. |

Notes:

- 13 charts total, numbered `01`–`13` with no gaps; numbering matches the call
  order in `main()`.
- The dashboard's 2x3 panels mirror charts 01, 02, 03, 04, 05 and 06; chart 07
  (migration lifecycle) is not on the dashboard.
- Regenerate all of them with `python src/visualize.py` (requires the four
  Parquet marts in `data/parquet/`).
