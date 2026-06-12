# I Disease Embeddings

Use `i_embeddings_codes_only.tsv` for downstream training limited to the I disease tree.
`i_embeddings_all_nodes.tsv` keeps the chapter root, block ancestors, and target codes for hierarchy context.

## Files

- Primary training table: `i_embeddings_codes_only.tsv`
- Hierarchy/context table: `i_embeddings_all_nodes.tsv`
- Machine-readable manifest: `manifest.json`

## Source Data

- Raw hierarchy: `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/data/datacode-19.tsv`
- Target code regex: `^[I][0-9]`
- Target prefixes: `I`
- I target codes: `475`
- Target disease-code rows: `475`
- Training nodes: `486`
- Direct training edges: `485`
- Root mode: `individual`

## Embedding Source

- Final checkpoint: `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_i/i_stage_d_repair.offline_best.pth`
- File prefix: `i`
- Recipe: current-hybrid baseline followed by Stage D branch-repair geometry
  using the same-prefix baseline checkpoint as both initialization and branch teacher.

## Columns

`node_id`, `coding`, `meaning`, `parent_id`, `depth`, `top_branch_id`,
`top_branch_code`, `selectable`, `is_target_code`, then `dim_1` through `dim_10`.
