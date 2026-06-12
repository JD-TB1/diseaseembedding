# Labmate Disease Embedding Handoff

Generated for review on 2026-06-12 from the committed C, G, I, and C/G/I
Poincare + CPCC + branch-repair embedding exports.

## Which File To Use

Use the code-only TSVs for downstream feature-selection training. Use the
combined C/G/I file when C, G, and I diseases must be compared in one shared
coordinate system. Do not concatenate or merge the separately trained C, G, and
I vectors for cross-group comparison.

| Use case | Primary downstream TSV | Rows |
| --- | --- | ---: |
| C-only downstream training | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_c/labmate/c_embeddings_codes_only.tsv` | 559 |
| G-only downstream training | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_g/labmate/g_embeddings_codes_only.tsv` | 396 |
| I-only downstream training | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_i/labmate/i_embeddings_codes_only.tsv` | 475 |
| Joint C/G/I downstream training | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_cgi/labmate/cgi_embeddings_codes_only.tsv` | 1430 |

## Context And Audit Files

Each bundle includes an all-node hierarchy table, a human-readable README, a
machine-readable manifest, evaluation metrics, and labeled plots.

| Dataset | All-node context TSV | Manifest | Evaluation summary | Labeled plots |
| --- | --- | --- | --- | --- |
| C | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_c/labmate/c_embeddings_all_nodes.tsv` | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_c/labmate/manifest.json` | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_c/eval_summary.md` | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_c/plots/` |
| G | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_g/labmate/g_embeddings_all_nodes.tsv` | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_g/labmate/manifest.json` | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_g/eval_summary.md` | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_g/plots/` |
| I | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_i/labmate/i_embeddings_all_nodes.tsv` | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_i/labmate/manifest.json` | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_i/eval_summary.md` | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_i/plots/` |
| C/G/I | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_cgi/labmate/cgi_embeddings_all_nodes.tsv` | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_cgi/labmate/manifest.json` | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_cgi/eval_summary.md` | `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_cgi/plots/` |

Recommended plot files to inspect first:

- `poincare_disk_branch_labeled_centroids.png`
- `poincare_disk_branch_labeled_route.png`
- `poincare_disk_depth.png`
- `depth_vs_radius.png`

## Data Structure

The downstream code-only TSVs contain one row per target ICD disease code. The
all-node TSVs contain the same target code rows plus hierarchy context nodes
such as chapter roots, block ancestors, and the synthetic `CGI_ROOT` in the
combined C/G/I export.

Columns:

| Column | Meaning |
| --- | --- |
| `node_id` | Source hierarchy node id from `data/datacode-19.tsv`, except synthetic roots such as `CGI_ROOT`. |
| `coding` | ICD code, chapter label, block label, or synthetic root code. |
| `meaning` | Source disease/category description. |
| `parent_id` | Parent node id in the exported training hierarchy. |
| `depth` | Depth in the exported hierarchy. Individual C/G/I roots start at depth 0; combined C/G/I has `CGI_ROOT` at depth 0. |
| `top_branch_id` | Top-level branch assignment used for branch geometry metrics and plots. |
| `top_branch_code` | Code/label for `top_branch_id`. |
| `selectable` | Original source selectable flag. |
| `is_target_code` | `Y` for downstream disease-code rows, `N` for context nodes. |
| `dim_1` ... `dim_10` | Ten-dimensional learned Poincare embedding coordinates. |

All embedding dimensions were validated as finite numeric values. The code-only
files contain no duplicate `coding` values and no unintended D-code targets.

## Dataset Counts

| Dataset | Target prefixes | Target-code rows | All-node rows | Direct edges | Closure edges | Top branches | Root mode |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| C | `C` | 559 | 575 | 574 | 1602 | 15 | Individual Chapter II root |
| G | `G` | 396 | 408 | 407 | 1131 | 11 | Individual Chapter VI root |
| I | `I` | 475 | 486 | 485 | 1368 | 10 | Individual Chapter IX root |
| C/G/I | `C`, `G`, `I` | 1430 | 1470 | 1469 | 5570 | 3 | Synthetic `CGI_ROOT` with Chapter II, VI, and IX branches |

## Training Recipe

All four bundles use the same two-stage recipe:

1. Train a dataset-specific current-hybrid baseline for 300 epochs.
2. Train a Stage D branch-repair run for 500 epochs, initialized from that
   dataset's own baseline checkpoint and using that baseline as the branch
   teacher.

Key repair hyperparameters:

| Parameter | Value |
| --- | ---: |
| `dim` | 10 |
| `lr` | 0.03 |
| `cpcc_weight` | 0.05 |
| `radial_weight` | 0.1 |
| `radial_margin` | 0.0005 |
| `depth_quantile_weight` | 0.1 |
| `depth_quantile_margin` | 0.0005 |
| `branch_teacher_weight` | 0.1 |
| `branch_contrastive_weight` | 0.3 |
| `branch_contrastive_margin` | 0.02 |
| `geometry_schedule` | `constant` |

Exact commands, selected checkpoint paths, and training configs are recorded in
each bundle's `manifest.json`.

## Embedding Quality Metrics

Lower is better for parent mean rank, branch ratio, sibling ratio, radial
violation rate, and gate deficit. Higher is better for reconstruction MAP,
ancestor MAP, and silhouette.

| Dataset | Selected epoch | Reconstruction MAP | Parent mean rank | Ancestor MAP | Branch within/across ratio | Sibling/non-sibling ratio | Branch silhouette | Gate deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 6 | 0.3590 | 1.3240 | 0.9501 | 0.8729 | 0.3839 | -0.0573 | 3.0745 |
| G | 230 | 0.3005 | 1.0098 | 0.9951 | 0.3903 | 0.1191 | 0.5337 | 1.2666 |
| I | 36 | 0.2927 | 1.0206 | 0.9937 | 0.4811 | 0.2467 | 0.3585 | 0.3989 |
| C/G/I | 34 | 0.3006 | 1.2342 | 0.9617 | 0.5702 | 0.2111 | 0.4226 | 0.8490 |

Depth/radius diagnostics:

| Dataset | Minimum adjacent depth mean gap | Minimum adjacent depth quantile gap | Positive quantile gaps | Parent-child radial violation rate |
| --- | ---: | ---: | ---: | ---: |
| C | 0.000490 | -0.000795 | 2 | 0.1864 |
| G | 0.003278 | -0.000739 | 2 | 0.1818 |
| I | 0.000481 | -0.000812 | 3 | 0.0990 |
| C/G/I | 0.000564 | -0.001199 | 4 | 0.1191 |

## Practical Notes

- G is the strongest individual embedding by branch geometry and sibling
  cohesion, but its radial violation rate remains nontrivial.
- I has strong parent/ancestor structure and the lowest gate deficit among the
  individual trees, but reconstruction MAP is slightly below 0.30.
- C preserves reconstruction MAP but has weak branch separation; use it for
  C-only experiments, but interpret branch-cluster geometry cautiously.
- C/G/I is the correct shared-coordinate file for mixed C/G/I downstream
  experiments. It is valid and documented, but branch separation is weaker than
  the G-only and I-only runs.
- These exports are embedding-input artifacts only. They do not by themselves
  prove downstream RL feature-selection performance; downstream validation
  should evaluate those models directly.
