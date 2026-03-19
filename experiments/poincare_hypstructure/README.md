# Poincare + HypStructure Hybrid

This experiment track extends the original Poincare embedding pipeline with additional structural losses motivated by the HypStructure project.

## Goal

Improve hierarchical level separation in the Poincare ball, especially radius ordering by depth, while keeping the branch clustering learned by the original Poincare relation loss.

## Objective Components

This track keeps the original Poincare relation-reconstruction term and adds:

- CPCC-style structural regularization
- radial parent-child ordering loss

The intended tradeoff is:

- keep strong branch clusters
- improve shell separation by tree depth

## Directory Layout

- `scripts/`
  - Hybrid training, evaluation, visualization, and tuning utilities.
- `results/disease90/`
  - Main hybrid run artifacts.
- `metadata/`
  - Disease-90 metadata and relation summaries used by this experiment track.
- `logs/`
  - Training and evaluation logs.
- `tuning/`
  - Radius-separation hyperparameter sweep outputs and summaries.

## Main Entry Points

- `scripts/run_disease90_pipeline.py`
  - Standard end-to-end hybrid run.
- `scripts/run_radius_tuning_campaign.py`
  - Multi-stage hyperparameter sweep focused on depth-by-radius separation.
- `scripts/rescore_disease90_run.py`
  - Offline checkpoint rescoring with radius-specific metrics.

## Evaluation Focus

In addition to reconstruction-style metrics, this track emphasizes:

- depth-radius Spearman/Pearson correlation
- mean radius by depth
- adjacent depth gaps
- parent-child radial violation rate
- leaf/internal radius ratio
- sibling cohesion
- within-branch vs across-branch separation

## Recommended Comparison Target

Compare this directory primarily against:

- `../poincare_only/results/disease90/direct_eval_metrics.json`

That pure-Poincare direct-edge baseline is the most relevant reference when judging whether the added structural losses improve radial hierarchy without destroying cluster structure.
