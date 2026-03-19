# Legacy Root Workspace

This directory preserves the original top-level disease embedding workspace that existed before the repository was reorganized around experiment-specific directories.

## What It Contains

- `scripts/`
  - Earlier disease-90 pipeline scripts from the pre-experiment-layout phase.
- `results/`
  - Outputs generated from the older top-level workflow.
- `metadata/`
  - Disease-90 metadata used by that workflow.
- `logs/`
  - Corresponding training and evaluation logs.

## Why It Is Kept

This workspace is kept so earlier intermediate work is not lost after the repository cleanup.

## Recommended Practice

Do not start new runs from this directory unless you specifically need to reproduce the older layout.

For maintained pipelines, use:

- `../../experiments/poincare_only/`
- `../../experiments/poincare_hypstructure/`
