Evaluation reference files
==========================

Frozen rolling-origin summaries for the dashboard Model evaluation tab
(GET /api/evaluation). These are not live training output.

- rolling_comparison_summary_expanded.json — 200-well expanded run (final demo).
- rolling_comparison_summary.json — 60-well real run.
- Results_Current.txt — human-readable 60-well metrics.

If full experiment folders exist locally under experiments/, prefer those
artifacts for fold-level numbers. Do not invent per-fold values from this
summary.
