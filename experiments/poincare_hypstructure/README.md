# Poincare + HypStructure Hybrid

This is the active method-development track.

It combines the original Poincare relation loss with two additional structure losses:

- CPCC global hierarchy alignment
- radial parent-child ordering

The active data mode is:

- `direct`

## Goal

Learn a disease embedding that preserves:

- parent/ancestor reconstruction
- branch separation
- radial hierarchy ordering

without reverting to a large artifact dump or a closure-edge training graph.

## Objective

```text
L_total = L_edge + alpha * L_CPCC + beta * L_radial
```

See `../../docs/algorithm.md` for the full explanation.

## What Is Canonical Here

Main committed references:

- `results/disease90/eval_summary.md`
- `results/disease90/eval_metrics.json`
- `results/disease90/train_config.json`
- `results/disease90/plots/branch_separation_summary.png`
- `results/disease90/plots/depth_vs_radius.png`
- `results/disease90/plots/poincare_disk_branch_labeled_centroids.png`
- `results/disease90/plots/poincare_disk_branch_labeled_route.png`
- `results/disease90/plots/sibling_distance_summary.png`

Committed tuning orientation files:

- `tuning/radius_separation/stage0/baseline_calibration.md`
- `tuning/radius_separation/summaries/stage1_summary.md`

Generated checkpoints, logs, exported embedding tables, and full per-run tuning directories are intentionally excluded from version control.

## Entry Points

- `scripts/run_disease90_pipeline.py`
  - main end-to-end hybrid pipeline
- `scripts/train_disease90.py`
  - direct trainer if you need to run training in isolation
- `scripts/evaluate_disease90.py`
  - structural evaluation summary
- `scripts/rescore_disease90_run.py`
  - offline checkpoint rescoring for tuning
- `scripts/run_radius_tuning_campaign.py`
  - staged radius-separation sweep

Maintained figure generators:

- `scripts/render_poster_panels.py`
- `scripts/render_rl_roc_panel.py`

These scripts are maintained because they encode the current presentation logic, but the files they generate are ignored by git.

## Recommended Command

```bash
conda run -n reasoning python experiments/poincare_hypstructure/scripts/run_disease90_pipeline.py \
  --relation-mode direct \
  --fresh
```

## Comparison Target

Judge this track first against:

- `../poincare_only/results/disease90/direct_eval_metrics.json`

That direct-edge pure-Poincare baseline is the fair reference for deciding whether CPCC and radial ordering improve the embedding.
