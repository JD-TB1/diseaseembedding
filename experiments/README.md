# Experiments

This directory is the main working area of the repository.

Each experiment track owns its own:

- `scripts/`
- `metadata/`
- `results/`

Generated logs, checkpoints, and bulk run artifacts are intentionally not treated as part of the committed interface.

## Tracks

### `poincare_only/`

Role:

- frozen baseline
- pure Poincare objective only
- relation-set comparison across `closure`, `direct`, and `hybrid`

Use this track when you need:

- the cleanest reference against the original objective
- the committed direct-edge baseline
- relation-construction comparisons without added regularizers

### `poincare_hypstructure/`

Role:

- active method-development track
- original Poincare edge loss plus CPCC and radial ordering
- structural evaluation and radius-separation tuning

Use this track when you need:

- the current working method
- radius-aware evaluation
- the staged hyperparameter tuning campaign
- poster-specific panel/ROC generators

## How To Work

1. Read `../docs/current_stage.md`.
2. Reproduce the direct baseline in `poincare_only/`.
3. Move to `poincare_hypstructure/` for any current development.
4. Ignore `archive/` unless you need historical script provenance.

## Output Policy

Committed outputs here are curated orientation artifacts:

- summaries
- metrics JSON files
- a small set of representative plots
- train configuration snapshots

Per-run checkpoints, logs, HTML exports, poster bundles, and large tuning run directories should remain local and ignored unless they are deliberately promoted into the canonical set.
