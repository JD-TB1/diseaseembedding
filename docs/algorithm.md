# Algorithm

## Problem Setup

The maintained experiments operate on the disease-90 subtree extracted from `data/datacode-19.tsv`.

Each node is a disease label in an ICD-10-style hierarchy with:

- `node_id`
- `parent_id`
- `coding`
- `meaning`
- `depth`

The active experiment uses a **direct-edge** training graph:

- each positive edge is `child -> parent`
- no transitive closure edges are used in the active hybrid track

This is important because the project is currently optimizing for a balance between:

- local parent-child reconstruction
- branch separation
- radial depth ordering

## Embedding Model

Each disease node gets a trainable embedding vector in the Poincare ball.

The low-level model comes from the copied Facebook `poincare-embeddings` reference:

- relation model: distance-based Poincare embedding
- geometry: Poincare ball
- parameter updates: Riemannian SGD

In practice, the embedding table itself is the main trainable parameter set.

## Objective

The active loss is:

```text
L_total = L_edge + alpha * L_CPCC + beta * L_radial
```

### 1. Edge Loss

`L_edge` is the original Poincare relation-reconstruction loss.

For a source node `u`, one positive target `v+`, and sampled negatives `v1-, v2-, ...`, the model computes Poincare distances and applies cross-entropy to the negated distances.

Effect:

- makes the true parent closer than sampled wrong targets
- drives local branch structure
- is the main source of parent recovery and branch discrimination

### 2. CPCC Loss

`L_CPCC` is the global hierarchy regularizer borrowed from HypStructure.

Implementation idea:

- build hierarchy groups from tree nodes
- compute Poincare means for the embeddings of each eligible group
- measure pairwise embedding distances between those group representatives
- compare them against pairwise tree distances

The loss is:

```text
L_CPCC = 1 - corr(embedding_group_distances, tree_group_distances)
```

Effect:

- aligns global embedding geometry with the tree structure
- encourages hierarchy-consistent organization beyond local parent-child edges

### 3. Radial Ordering Loss

`L_radial` is a margin-based parent-child radius constraint.

For each direct `child, parent` pair:

```text
L_radial = mean( max(0, radius(parent) + margin - radius(child)) )
```

Effect:

- penalizes cases where a child is not farther from the origin than its parent
- improves radius-depth ordering

## Negative Sampling

The edge loss does not compare each positive edge against every possible wrong target.

Instead, each minibatch example contains:

- one source node
- one true related target
- `K` sampled negatives

Active defaults in the hybrid track:

- `negs = 50`
- `dampening = 0.75`

Meaning:

- for each true child-parent pair, sample 50 wrong targets
- train the model to rank the true parent above those negatives

Why it matters:

- this makes training efficient
- this is also where much of the branch discrimination pressure comes from
- tuning `negs` and `dampening` changes how hard the ranking problem is

## Optimization

Training uses minibatch backpropagation, but the parameter update is not plain Euclidean SGD.

The update rule is:

- compute gradients of `L_total`
- convert them into the Riemannian gradient under the Poincare metric
- take a stochastic gradient step in the manifold
- project/normalize back into the Poincare ball

The active trainer uses:

- optimizer: `RiemannianSGD`
- default learning rate: `0.1`
- default epochs: `300`
- burn-in: `20` epochs

During burn-in, the effective learning rate is scaled down to `0.01 * lr`.

## Checkpoint Selection

There are two checkpoint-selection modes in the repository.

### Main hybrid training run

The trainer can select by one of:

- `combined`
- `reconstruction_map`
- `depth_spearman`
- `negative_loss`

The current default for the main hybrid pipeline is:

- `selection_metric = combined`

which means:

```text
combined = reconstruction_map + depth_spearman
```

### Tuning campaign

The tuning campaign does **not** trust the trainer-selected `.best` checkpoint as authoritative.

Instead, it rescans saved checkpoints offline and ranks them using radius-aware criteria, including:

- monotonic mean radius by depth
- minimum adjacent depth gap
- depth-radius Spearman
- parent-child radial violation rate
- leaf mean radius
- reconstruction MAP

## Evaluation

The active evaluator reports:

- reconstruction mean rank
- reconstruction MAP
- depth-radius Spearman and Pearson
- parent mean rank
- ancestor MAP
- mean radius by depth
- adjacent depth gaps
- parent-child radial violation rate
- leaf/internal radius ratio
- sibling vs same-depth non-sibling distances
- within-branch vs across-branch distances

Interpretation rule of thumb:

- lower parent mean rank is better
- higher MAP is better
- higher depth-radius correlation is better
- lower sibling distance than same-depth non-sibling distance is good
- lower within-branch distance than across-branch distance is good
