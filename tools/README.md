# Tools

This directory contains general-purpose utilities that are not tied to a single experiment track.

## Current Utilities

- `visualize_disease_tree.py`
  - Build reusable visualizations of the disease TSV as either:
    - interactive collapsible HTML
    - static Graphviz node-link diagrams

## Typical Use

Whole forest as interactive HTML:

```bash
conda run -n reasoning python tools/visualize_disease_tree.py \
  --input-tsv data/datacode-19.tsv \
  --root-id 0 \
  --mode html \
  --layout collapsible \
  --output visualizations/disease_forest_full.html \
  --title "Entire disease forest"
```

Current disease-90 subtree as a radial SVG:

```bash
conda run -n reasoning python tools/visualize_disease_tree.py \
  --input-tsv data/datacode-19.tsv \
  --root-id 90 \
  --mode graphviz \
  --layout radial \
  --color-by route \
  --output visualizations/disease_subtree_90_radial.svg \
  --title "Disease-90 subtree"
```

The `--color-by route` setting colors each node by the first branch below the selected root, which is often more useful than depth coloring when you want to inspect parent-child cluster structure.
