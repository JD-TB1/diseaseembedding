# Current Stage

## Repository Scope

This repository currently covers the disease-embedding stage of the larger project.

Completed here:

- extraction of the disease-90 subtree from `data/datacode-19.tsv`
- pure Poincare baselines on closure, direct, and hybrid relation sets
- active hybrid method that combines Poincare edge loss, CPCC, and radial ordering
- structural evaluation utilities for hierarchy, branch separation, and radius ordering
- staged radius-separation tuning framework with offline checkpoint rescoring
- reusable disease-tree visualizations for manual inspection

Not owned here:

- the full RL feature-selection training codebase
- Olink patient-level model training and deployment

## Current Working Answer

The active method is:

- experiment track: `experiments/poincare_hypstructure/`
- data mode: `direct`
- losses: `L_edge + alpha * L_CPCC + beta * L_radial`
- optimizer: minibatch Riemannian SGD in the Poincare ball

The main checked-in hybrid evaluation summary is:

- `experiments/poincare_hypstructure/results/disease90/eval_summary.md`

Key numbers from that committed summary:

- reconstruction MAP: `0.3024`
- reconstruction mean rank: `7.5381`
- depth-radius Spearman: `0.3633`
- parent mean rank: `1.2268`
- ancestor MAP: `0.9553`
- within-branch mean distance: `0.0090`
- across-branch mean distance: `0.0214`

Interpretation:

- branch structure is clearly better than the frozen baseline
- hierarchy is preserved reasonably well
- radial depth ordering is positive but still incomplete

## Baseline Reference

The frozen baseline to compare against is:

- `experiments/poincare_only/results/disease90/direct_eval_summary.md`
- `experiments/poincare_only/results/disease90/direct_eval_metrics.json`

That direct-edge baseline is the cleanest pure-Poincare reference for judging whether added structure losses help or hurt.

## Tuning Status

The radius-separation campaign infrastructure is implemented, but the campaign is not complete.

Committed orientation files:

- `experiments/poincare_hypstructure/tuning/radius_separation/stage0/baseline_calibration.md`
- `experiments/poincare_hypstructure/tuning/radius_separation/summaries/stage1_summary.md`

Current state:

- stage 0 baseline calibration is done
- one stage-1 pilot run was executed end-to-end
- the remaining sweep is intended to be resumed from the committed tuning scripts, not from archived run artifacts

## Open Problems

- improve radius separation between adjacent tree depths without degrading branch clusters
- reconcile online checkpoint selection with the stronger offline depth-first rescoring rule
- decide whether the committed main hybrid result should be refreshed from the tuning campaign
- extend beyond disease-90 once the depth/branch tradeoff is more stable
- use the learned disease embedding more richly in downstream RL than simple concatenation

## Recommended Next Steps For A New Contributor

1. Read `docs/algorithm.md` and `docs/reproduce.md`.
2. Rebuild the reference Cython extensions if needed.
3. Reproduce the direct baseline once.
4. Reproduce the main hybrid run once.
5. Resume the tuning campaign only after you are comfortable with the evaluation metrics and output locations.
