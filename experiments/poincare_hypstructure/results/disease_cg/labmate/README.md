# C/G Disease Embeddings

Use `cg_embeddings_codes_only.tsv` for downstream feature-selection training.
`cg_embeddings_all_nodes.tsv` is provided only when hierarchy context or plotting needs
the synthetic root, chapter nodes, and block ancestors.

## Files

- Primary training table: `cg_embeddings_codes_only.tsv`
- Hierarchy/context table: `cg_embeddings_all_nodes.tsv`
- Machine-readable manifest: `manifest.json`

## Source Data

- Raw hierarchy: `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/data/datacode-19.tsv`
- Target code regex: `^[CG][0-9]`
- C target codes: `559`
- G target codes: `396`
- Target disease-code rows: `955`
- Training nodes including synthetic root: `984`
- Direct training edges: `983`

## Embedding Source

- Final checkpoint: `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_cg/cg_stage_d_repair.offline_best.pth`
- Recipe: C/G current-hybrid baseline followed by Stage D branch-repair geometry
  using the C/G baseline checkpoint as both initialization and branch teacher.

## Columns

`node_id`, `coding`, `meaning`, `parent_id`, `depth`, `top_branch_id`,
`top_branch_code`, `selectable`, `is_target_code`, then `dim_1` through `dim_10`.
