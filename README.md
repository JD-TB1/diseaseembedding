# Disease Embedding

This repository contains a reproducible workspace for learning and analyzing hyperbolic embeddings of a disease hierarchy derived from `datacode-19.tsv`.

The project started by reconstructing the original Facebook `poincare-embeddings` training pipeline, then branched into two experiment tracks:

- `experiments/poincare_only/`
  - Pure Poincare embedding experiments on the disease-90 subtree.
  - Includes baseline runs comparing closure, direct, and hybrid relation sets.
- `experiments/poincare_hypstructure/`
  - Extended experiments that combine the original Poincare relation loss with HypStructure-inspired CPCC regularization and a radial ordering term.
  - Includes visualization, evaluation, and radius-separation tuning utilities.

## Repository Layout

- `data/`
  - Input disease hierarchy data.
- `references/`
  - Copied third-party code used as the reference implementation.
- `experiments/`
  - Primary working area for all maintained experiment pipelines and outputs.
- `archive/`
  - Preserved older workspaces and transitional files kept for provenance.
- `tools/`
  - General-purpose utilities, including TSV tree visualization.
- `visualizations/`
  - Generated whole-forest and subtree views.

## Data

The current experiments focus on the subtree rooted at `node_id=90` from:

- `data/datacode-19.tsv`

This subtree corresponds to `Chapter IX` of the ICD-style disease forest and is used to study how well hyperbolic embeddings preserve:

- hierarchical depth
- branch separation
- local subtree geometry

## Experiment Tracks

### 1. Pure Poincare Baseline

Path:

- `experiments/poincare_only/`

Purpose:

- reproduce the original Poincare-style embedding workflow on the disease tree
- compare `closure`, `direct`, and `hybrid` edge constructions
- provide a frozen baseline for later comparison

Key outputs:

- `experiments/poincare_only/results/disease90/comparison_summary.md`
- `experiments/poincare_only/results/disease90/direct_eval_metrics.json`
- `experiments/poincare_only/results/disease90/plots_direct/`

### 2. Poincare + HypStructure Hybrid

Path:

- `experiments/poincare_hypstructure/`

Purpose:

- keep the original Poincare relation-reconstruction objective
- add a CPCC structural regularizer inspired by HypStructure
- add a radial ordering loss to improve radius separation by depth

Key outputs:

- `experiments/poincare_hypstructure/results/disease90/eval_summary.md`
- `experiments/poincare_hypstructure/results/disease90/plots/`
- `experiments/poincare_hypstructure/tuning/radius_separation/`

## Reference Implementation

The copied original Facebook implementation is kept under:

- `references/poincare-embeddings/`

This directory is treated as a reference dependency for the local experiment scripts. The maintained disease-specific pipelines live in `experiments/`, not in the copied reference tree.

## Environment

The project was developed and run in the `reasoning` conda environment.

The copied Poincare reference contains its own `environment.yml` and `setup.py`, but the disease-specific experiments are orchestrated from the scripts in `experiments/`.

If the compiled Cython extensions for the copied Poincare reference are missing after cloning, rebuild them from:

- `references/poincare-embeddings/`

using:

```bash
conda run -n reasoning python setup.py build_ext --inplace
```

## Recommended Entry Points

For pure Poincare experiments:

- `experiments/poincare_only/scripts/run_disease90_pipeline.py`

For hybrid-loss experiments:

- `experiments/poincare_hypstructure/scripts/run_disease90_pipeline.py`
- `experiments/poincare_hypstructure/scripts/run_radius_tuning_campaign.py`

## Suggested Reading Order

1. `README.md`
2. `experiments/README.md`
3. `experiments/poincare_only/README.md`
4. `experiments/poincare_hypstructure/README.md`
5. `references/README.md`

## Notes for Labmates

- Treat `experiments/poincare_only/` as the baseline reference.
- Treat `experiments/poincare_hypstructure/` as the active method-development area.
- Treat `archive/` as preserved history, not the preferred place to start.
- If you rerun experiments, use the scripts inside the experiment subdirectories rather than the archived legacy workspace.
