# Pure Poincare Baseline

This directory is the frozen baseline track.

It preserves the disease-90 experiments that use the original Poincare objective without the added CPCC or radial-ordering losses.

## What This Track Answers

- How does the original Poincare objective behave on the disease-90 subtree?
- What changes when the relation graph uses `closure`, `direct`, or `hybrid` edges?
- What is the cleanest baseline to compare against the active hybrid method?

## Recommended Baseline

The baseline that matters most for the active project is the **direct-edge** run.

Canonical committed references:

- `results/disease90/direct_eval_summary.md`
- `results/disease90/direct_eval_metrics.json`
- `results/disease90/comparison_summary.md`

The direct-edge baseline is the most relevant comparison because it preserves the original objective while avoiding closure-edge supervision that tends to distort depth/radius behavior.

## What Is Still Committed Here

- `scripts/`
  - baseline pipeline and helpers
- `metadata/`
  - disease-90 node tables and dataset summaries
- `results/disease90/*.json`
  - committed metric snapshots
- `results/disease90/*.md`
  - committed summaries
- `results/disease90/train_config*.json`
  - canonical hyperparameter settings
- `results/disease90/disease90_relations_*.csv`
  - relation-set definitions used by the baseline comparison

Large checkpoints, embedding tables, logs, and bulk plot outputs were intentionally removed from version control.

## Entry Point

Use:

- `scripts/run_disease90_pipeline.py`

Recommended command:

```bash
conda run -n reasoning python experiments/poincare_only/scripts/run_disease90_pipeline.py \
  --relation-mode direct \
  --fresh
```

## Maintenance Status

Treat this directory as read-mostly.

New work here should usually be limited to:

- reproducing the baseline
- reading metric differences across relation modes
- making fair baseline comparisons against the active hybrid track
