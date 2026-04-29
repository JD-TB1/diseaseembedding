# Disease Embedding Workspace

This repository is the maintained stage-1 workspace for the broader proteomic disease-prediction project.

Its purpose is to learn hyperbolic embeddings of disease labels from the ICD-10-style hierarchy in `data/datacode-19.tsv`, evaluate how well those embeddings preserve hierarchy and branch structure, and produce reproducible outputs that can be used by downstream work such as embedding-guided reinforcement-learning feature selection.

The current active method is:

- direct-edge disease graph on the disease-90 subtree
- original Poincare relation-reconstruction loss
- HypStructure-inspired CPCC regularization
- radial parent-child ordering loss

This repository does **not** contain the full RL feature-selection pipeline. It owns the embedding stage and the evaluation/tuning machinery around it.

## Start Here

Read these in order after cloning:

1. `docs/current_stage.md`
2. `docs/algorithm.md`
3. `docs/reproduce.md`
4. `docs/repo_map.md`
5. `experiments/README.md`

## What Is Canonical

The committed outputs are intentionally small. They are the orientation artifacts for new contributors, not a full experiment dump.

Canonical baseline references:

- `experiments/poincare_only/results/disease90/comparison_summary.md`
- `experiments/poincare_only/results/disease90/direct_eval_summary.md`
- `experiments/poincare_only/results/disease90/direct_eval_metrics.json`

Canonical active-method references:

- `experiments/poincare_hypstructure/results/disease90/eval_summary.md`
- `experiments/poincare_hypstructure/results/disease90/eval_metrics.json`
- `experiments/poincare_hypstructure/results/disease90/train_config.json`
- `experiments/poincare_hypstructure/results/disease90/plots/`

Canonical tuning references:

- `experiments/poincare_hypstructure/tuning/radius_separation/stage0/baseline_calibration.md`
- `experiments/poincare_hypstructure/tuning/radius_separation/summaries/stage1_summary.md`

Canonical hierarchy visualizations:

- `visualizations/disease_subtree_90_focus_paths.svg`
- `visualizations/disease_subtree_1150_leftright_by_route.svg`
- `visualizations/disease_subtree_90_radial_by_route.svg`

Large generated outputs such as checkpoints, per-run tuning artifacts, HTML exports, poster bundles, and logs are intentionally excluded from version control. Regenerate them from the scripts when needed.

## Repository Layout

- `data/`
  - Source disease hierarchy TSV used by all maintained experiment tracks.
- `references/`
  - Local copy of the Facebook `poincare-embeddings` implementation used as the low-level manifold/training backend.
- `experiments/poincare_only/`
  - Frozen pure-Poincare baseline track.
- `experiments/poincare_hypstructure/`
  - Active development track: Poincare + CPCC + radial ordering.
- `tools/`
  - General utilities, especially tree visualization from TSV.
- `visualizations/`
  - Curated reusable hierarchy views only.
- `archive/`
  - Slimmed historical provenance. Do not start new work here.
- `docs/`
  - Handoff documentation for labmates.

## Quick Commands

Build the reference extensions once after cloning:

```bash
cd references/poincare-embeddings
conda run -n reasoning python setup.py build_ext --inplace
```

Run the frozen direct-edge baseline:

```bash
conda run -n reasoning python experiments/poincare_only/scripts/run_disease90_pipeline.py \
  --relation-mode direct \
  --fresh
```

Run the active hybrid method:

```bash
conda run -n reasoning python experiments/poincare_hypstructure/scripts/run_disease90_pipeline.py \
  --relation-mode direct \
  --fresh
```

Continue the radius-separation tuning campaign:

```bash
conda run -n reasoning python experiments/poincare_hypstructure/scripts/run_radius_tuning_campaign.py \
  --stages stage1 stage2 stage3 stage4 stage5 \
  --skip-existing
```

## Contributor Guidance

- Start from `experiments/poincare_hypstructure/` unless you are explicitly working on the baseline.
- Treat `experiments/poincare_only/` as a reference point, not the main development area.
- Treat `archive/` as historical context only.
- If you generate new checkpoints or poster assets, they should stay local unless the output is promoted into the curated canonical set.
