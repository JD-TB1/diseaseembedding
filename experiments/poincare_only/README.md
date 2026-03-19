# Pure Poincare Baseline

This experiment track preserves the disease-90 baseline built around the original Poincare embedding objective.

## Goal

Establish a clean comparison point before adding any HypStructure-inspired regularization.

This track is used to compare how the original Poincare loss behaves under different relation graphs:

- `closure`
- `direct`
- `hybrid`

## Directory Layout

- `scripts/`
  - Disease-90 pipeline scripts for:
    - relation-file construction
    - training
    - export
    - visualization
    - evaluation
- `results/disease90/`
  - Produced artifacts for the disease-90 subtree.
- `metadata/`
  - Node tables and dataset summaries for the subtree.
- `logs/`
  - Training and evaluation logs for the baseline runs.

## Main Entry Point

- `scripts/run_disease90_pipeline.py`

This script orchestrates:

1. subtree extraction
2. relation-file generation
3. training through the copied Poincare reference
4. embedding export
5. visualization
6. evaluation

## Key Results

- `results/disease90/eval_metrics.json`
  - Closure-edge pure-Poincare baseline.
- `results/disease90/direct_eval_metrics.json`
  - Direct-edge pure-Poincare baseline.
- `results/disease90/hybrid_eval_metrics.json`
  - Mixed-edge pure-Poincare baseline.
- `results/disease90/comparison_summary.md`
  - Side-by-side comparison across the three relation constructions.

## Interpretation

In this repository, the direct-edge Poincare baseline became the main comparison point for later hybrid-loss work, because it improved radial hierarchy ordering relative to closure while preserving strong branch structure.

## When To Use This Directory

Use `poincare_only/` when you want:

- the cleanest baseline against the original objective
- relation-set comparisons without extra losses
- a frozen reference for ablations against the hybrid model
