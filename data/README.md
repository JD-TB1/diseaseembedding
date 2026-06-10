# Data

This directory stores the source disease hierarchy used by the embedding experiments.

## Files

- `datacode-19.tsv`
  - The raw disease forest used in this project.
  - Key columns include:
    - `coding`
    - `meaning`
    - `node_id`
    - `parent_id`
    - `selectable`

## Current Use

The maintained pipelines in `experiments/` currently extract and embed:

- the disease-90 subtree rooted at `node_id=90`
- the C/G target subforest used for downstream feature-selection training

These views are converted into experiment-specific relation CSV files inside each experiment directory, for example:

- `experiments/poincare_only/results/disease90/disease90_relations_direct.csv`
- `experiments/poincare_hypstructure/results/disease90/disease90_relations_direct.csv`
- `experiments/poincare_hypstructure/results/disease_cg/cg_relations_direct.csv`

The C/G subforest uses ICD code regex `^[CG][0-9]`, then adds only required
Chapter II, Chapter VI, and block ancestors plus a synthetic `CG_ROOT` training
root. It intentionally excludes D-code neoplasm rows from the target disease
embedding table.

The raw TSV is kept unchanged here so both experiment tracks use the same source data.
