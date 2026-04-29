# Reproduce

## Environment Assumptions

The maintained workflows were developed in the `reasoning` conda environment.

If you use a different environment, make sure it can provide:

- Python
- PyTorch
- NumPy
- Matplotlib
- the compiled Cython extensions from `references/poincare-embeddings`

## Build The Reference Extensions

Run this once after cloning, or again if the compiled reference modules are missing:

```bash
cd references/poincare-embeddings
conda run -n reasoning python setup.py build_ext --inplace
```

The experiment scripts call into this local reference tree for data loading, manifold math, and Riemannian SGD.

## Baseline: Frozen Pure Poincare

Recommended baseline command:

```bash
conda run -n reasoning python experiments/poincare_only/scripts/run_disease90_pipeline.py \
  --relation-mode direct \
  --fresh
```

What it does:

1. builds the direct-edge disease-90 relation CSV
2. trains the pure Poincare model
3. exports embeddings
4. produces plots
5. evaluates the run

Expected committed orientation outputs:

- `experiments/poincare_only/results/disease90/direct_eval_summary.md`
- `experiments/poincare_only/results/disease90/direct_eval_metrics.json`

Notes:

- the baseline pipeline defaults to `closure`, so pass `--relation-mode direct` explicitly when reproducing the comparison target used by the hybrid track
- large generated checkpoints and plots are ignored and do not need to be committed

## Active Method: Poincare + CPCC + Radial Ordering

Recommended hybrid command:

```bash
conda run -n reasoning python experiments/poincare_hypstructure/scripts/run_disease90_pipeline.py \
  --relation-mode direct \
  --fresh
```

Default active hyperparameters:

- `dim = 10`
- `epochs = 300`
- `lr = 0.1`
- `negs = 50`
- `batchsize = 64`
- `burnin = 20`
- `cpcc_weight = 0.05`
- `radial_weight = 0.01`
- `radial_margin = 0.02`
- `cpcc_min_group_size = 2`
- `depth_band_weight = 0.0`
- `depth_quantile_weight = 0.0`
- `depth_quantile_margin = 0.001`
- `branch_weight = 0.0`
- `branch_cos_margin = 0.2`
- `branch_teacher_weight = 0.0`
- `branch_contrastive_weight = 0.0`
- `branch_contrastive_margin = 0.02`
- `branch_contrastive_hard_k = 0`
- `init_source = current-hybrid`
- `geometry_schedule = ramp`
- `selection_metric = combined`

Expected committed orientation outputs:

- `experiments/poincare_hypstructure/results/disease90/eval_summary.md`
- `experiments/poincare_hypstructure/results/disease90/eval_metrics.json`
- `experiments/poincare_hypstructure/results/disease90/plots/`

## Tuning Campaign

The tuning campaign is launched from the active hybrid track.

Continue the committed staged campaign with:

```bash
conda run -n reasoning python experiments/poincare_hypstructure/scripts/run_radius_tuning_campaign.py \
  --stages stage1 stage2 stage3 stage4 stage5 \
  --skip-existing
```

Useful variants:

```bash
conda run -n reasoning python experiments/poincare_hypstructure/scripts/run_radius_tuning_campaign.py \
  --stages stage1 \
  --limit-runs 1
```

Committed tuning orientation files:

- `experiments/poincare_hypstructure/tuning/radius_separation/stage0/baseline_calibration.md`
- `experiments/poincare_hypstructure/tuning/radius_separation/summaries/stage1_summary.md`

Large per-run tuning artifacts are intentionally excluded from version control.

## Geometry Campaign

The branch/depth geometry campaign is separate from the older radius campaign:

```bash
conda run -n reasoning python experiments/poincare_hypstructure/scripts/run_geometry_tuning_campaign.py \
  --stages stage0 stage1 stage2 stage3 \
  --skip-existing
```

Stage 1 runs depth-only, branch-only, and combined ablations. Stage 2 sweeps
depth-band, branch, and radial weights. Stage 3 reruns the top candidates for
500 epochs with Poincare initialization and the ramped geometry schedule.

## Branch Repair Campaign

The next campaign is gate-oriented and starts from the current hybrid checkpoint
instead of rebuilding geometry from direct Poincare initialization:

```bash
conda run -n reasoning python experiments/poincare_hypstructure/scripts/run_branch_repair_campaign.py \
  --stages stageA stageB \
  --skip-existing
```

Stage A re-scores the current hybrid and prior Stage 1-3 geometry candidates
with `gate_deficit`. Stage B sweeps quantile depth margins, teacher branch
preservation, same-depth branch contrastive loss, and lower learning rates for
150 epochs. If branch ratio remains above `0.35`, Stage C increases
contrastive weight and enables same-depth hard-negative mining. Stage D reruns
the top five candidates for 500 epochs.

## Tree Visualizations

Reusable tree-view generation lives under `tools/`.

Focused disease-90 path view:

```bash
conda run -n reasoning python tools/visualize_disease_tree.py \
  --input-tsv data/datacode-19.tsv \
  --root-id 90 \
  --mode graphviz \
  --layout leftright \
  --color-by route \
  --auto-focus-by-route 6 \
  --collapse-omitted \
  --output visualizations/disease_subtree_90_focus_paths.svg \
  --title "Disease-90 subtree (focused paths)"
```

Detailed `I10-I15` branch view:

```bash
conda run -n reasoning python tools/visualize_disease_tree.py \
  --input-tsv data/datacode-19.tsv \
  --root-id 1150 \
  --mode graphviz \
  --layout leftright \
  --color-by route \
  --output visualizations/disease_subtree_1150_leftright_by_route.svg \
  --title "Block I10-I15"
```

## Poster Figure Generators

Poster-specific renderers are maintained scripts, but their outputs are generated locally and ignored by git.

Examples:

```bash
conda run -n reasoning python experiments/poincare_hypstructure/scripts/render_poster_panels.py
conda run -n reasoning python experiments/poincare_hypstructure/scripts/render_rl_roc_panel.py --auc 0.82
```

## Sanity Checks After A Fresh Clone

- `README.md` and all docs open without broken links
- `references/poincare-embeddings` builds successfully
- the direct baseline can run end-to-end
- the hybrid track can run end-to-end
- generated logs, checkpoints, and poster assets remain ignored by git
