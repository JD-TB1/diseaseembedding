# Tools

This directory contains general-purpose utilities that are not tied to a single experiment track.

## Current Utilities

- `visualize_disease_tree.py`
  - Build reusable visualizations of the disease TSV as either:
    - interactive collapsible HTML
    - static Graphviz node-link diagrams
  - Supports focused-path views and poster-friendly label/layout controls.

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

Traditional tree view with only a few representative paths kept and all other sibling branches collapsed into ellipsis nodes:

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

Useful focus options:

- `--focus-node-id <node>` repeated several times to keep specific root-to-node paths
- `--auto-focus-by-route N` to choose up to `N` representative deepest leaf paths from distinct first-level routes
- `--collapse-omitted` to replace all non-kept sibling subtrees with dashed ellipsis nodes

Useful poster-oriented Graphviz options:

- `--compact-labels`
  - shorter labels for flatter figures
- `--hide-legend`
  - omit the route legend
- `--nodesep`
  - adjust horizontal spacing
- `--ranksep`
  - adjust vertical spacing
- `--node-font-size`
  - control label size
- `--full-label-node-id <node>`
  - keep selected nodes on full text even when `--compact-labels` is enabled
