# Experiments

This directory is the primary working area of the repository.

Each subdirectory is an experiment track with its own:

- `scripts/`
- `results/`
- `metadata/`
- `logs/`

This separation keeps baselines, hybrid objectives, and tuning runs isolated from each other.

## Experiment Tracks

### `poincare_only/`

Purpose:

- pure Poincare embedding baseline
- comparison of relation-set constructions (`closure`, `direct`, `hybrid`)
- stable reference point for all later modifications

Use this track when you want to answer:

- How well does the original Poincare objective work on the disease-90 subtree?
- What changes when the relation graph uses direct edges instead of closure?

### `poincare_hypstructure/`

Purpose:

- hybrid objective that combines:
  - original Poincare relation loss
  - HypStructure-inspired CPCC regularization
  - radial ordering regularization
- radius-separation tuning and structural evaluation

Use this track when you want to answer:

- Can we improve depth-by-radius ordering without losing branch structure?
- Which hyperparameter settings best separate depth shells?

## Recommended Workflow

1. Start with `poincare_only/` to understand the baseline.
2. Move to `poincare_hypstructure/` for the hybrid objective and tuning.
3. Use the archived root workspace only if you need historical provenance.
