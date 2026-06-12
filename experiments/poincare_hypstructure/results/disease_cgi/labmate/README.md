# CGI Disease Embeddings

Use `cgi_embeddings_codes_only.tsv` when downstream training needs C, G, and I disease codes in one shared coordinate system.
`cgi_embeddings_all_nodes.tsv` keeps the synthetic root, chapter nodes, block ancestors, and target codes for hierarchy context.

## Files

- Primary training table: `cgi_embeddings_codes_only.tsv`
- Hierarchy/context table: `cgi_embeddings_all_nodes.tsv`
- Machine-readable manifest: `manifest.json`

## Source Data

- Raw hierarchy: `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/data/datacode-19.tsv`
- Target code regex: `^[CGI][0-9]`
- Target prefixes: `C/G/I`
- C target codes: `559`
- G target codes: `396`
- I target codes: `475`
- Target disease-code rows: `1430`
- Training nodes: `1470`
- Direct training edges: `1469`
- Root mode: `combined`

## Embedding Source

- Final checkpoint: `/Users/jayding/Desktop/DUKE/Research/BioStats_Lynn/diseaseembedding/experiments/poincare_hypstructure/results/disease_cgi/cgi_stage_d_repair.offline_best.pth`
- File prefix: `cgi`
- Recipe: current-hybrid baseline followed by Stage D branch-repair geometry
  using the same-prefix baseline checkpoint as both initialization and branch teacher.

## Columns

`node_id`, `coding`, `meaning`, `parent_id`, `depth`, `top_branch_id`,
`top_branch_code`, `selectable`, `is_target_code`, then `dim_1` through `dim_10`.
