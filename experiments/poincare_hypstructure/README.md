# Poincare + HypStructure Hybrid

This is the active method-development track.

It combines the original Poincare relation loss with optional structure losses:

- CPCC global hierarchy alignment
- radial parent-child ordering
- depth-band radial targets
- adjacent-depth quantile margins
- top-branch angular separation
- teacher-preserved branch layout
- same-depth branch contrastive margins

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
L_total = L_edge
        + alpha * L_CPCC
        + beta * L_radial
        + gamma * L_depth
        + delta * L_branch
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
- `scripts/run_geometry_tuning_campaign.py`
  - staged branch/depth geometry sweep
- `scripts/run_branch_repair_campaign.py`
  - current-hybrid repair sweep using gate-deficit selection

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

The default trainer now uses a ramped geometry schedule and initializes from
the current hybrid checkpoint when available. Fixed depth bands and the older
angular branch loss are available but opt-in; the branch-repair campaign uses
the current-hybrid checkpoint, constant geometry weighting, quantile depth
margins, teacher branch preservation, and same-depth branch contrastive loss.
The main geometry knobs are:

- `--depth-band-weight`
- `--depth-quantile-weight`
- `--depth-quantile-margin`
- `--branch-weight`
- `--branch-cos-margin`
- `--branch-teacher-weight`
- `--branch-teacher-checkpoint`
- `--branch-contrastive-weight`
- `--branch-contrastive-margin`
- `--branch-contrastive-hard-k`
- `--geometry-schedule`
- `--init-source`
- `--init-checkpoint`

If `--init-checkpoint` is omitted, the trainer attempts to initialize from the
best available current-hybrid checkpoint, then falls back to the direct
pure-Poincare checkpoint, then scratch.

## Branch Repair Campaign

Run the next gate-oriented repair campaign with:

```bash
conda run -n reasoning python experiments/poincare_hypstructure/scripts/run_branch_repair_campaign.py \
  --stages stageA stageB \
  --skip-existing
```

Stage A re-ranks current and prior geometry baselines with gate-deficit scoring.
Stage B repairs from the current hybrid checkpoint for 150 epochs. Stage C adds
stronger hard-negative branch contrastive loss if Stage B still misses the
branch-ratio gate. Stage D reruns the top five candidates for 500 epochs.

## Comparison Target

Judge this track first against:

- `../poincare_only/results/disease90/direct_eval_metrics.json`

That direct-edge pure-Poincare baseline is the fair reference for deciding whether CPCC and radial ordering improve the embedding.
