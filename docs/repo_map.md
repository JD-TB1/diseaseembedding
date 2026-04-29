# Repo Map

## Maintained Areas

These are the directories new contributors should treat as the working repository.

- `data/`
  - source hierarchy TSV
- `references/poincare-embeddings/`
  - copied upstream reference implementation
- `experiments/poincare_only/`
  - frozen pure-Poincare baseline
- `experiments/poincare_hypstructure/`
  - active hybrid method and tuning framework
- `tools/`
  - reusable utilities
- `visualizations/`
  - curated hierarchy views only
- `docs/`
  - handoff and reproducibility documentation

## Active Code Paths

Use these entrypoints first:

- `experiments/poincare_only/scripts/run_disease90_pipeline.py`
- `experiments/poincare_hypstructure/scripts/run_disease90_pipeline.py`
- `experiments/poincare_hypstructure/scripts/run_radius_tuning_campaign.py`
- `tools/visualize_disease_tree.py`

Poster-oriented maintained generators:

- `experiments/poincare_hypstructure/scripts/render_poster_panels.py`
- `experiments/poincare_hypstructure/scripts/render_rl_roc_panel.py`

## Curated Committed Outputs

Committed outputs are deliberately narrow.

Keep in version control:

- summaries and evaluation JSON/Markdown files that explain current status
- a small set of representative plots used in docs
- curated tree views
- train configuration JSON files that document canonical hyperparameters

Do not keep in version control:

- large checkpoints
- exported embedding tables
- large HTML tree exports
- full tuning run directories
- poster export bundles
- logs and Matplotlib cache directories

## Archive

`archive/legacy_root_workspace/` is kept only for provenance.

What remains there:

- the legacy scripts
- a README explaining what the old layout used to be

What was intentionally removed from the archive:

- logs
- metadata copies already preserved elsewhere
- generated results
- duplicated plots
- checkpoints

If you need historical output numbers, use the maintained experiment summaries instead of the archive.

## Files You Should Normally Not Edit

- `references/poincare-embeddings/`
  - treat as a vendored reference unless there is a deliberate upstream-facing reason to patch it
- `archive/`
  - provenance only
- committed result summaries in `results/`
  - regenerate them by rerunning scripts instead of hand-editing them

## Typical Contributor Flow

1. Read the docs.
2. Rebuild the reference extensions if needed.
3. Reproduce the direct baseline.
4. Reproduce the active hybrid run.
5. Continue tuning from the staged tuning scripts.
6. Only promote new outputs into version control if they belong in the curated canonical set.
