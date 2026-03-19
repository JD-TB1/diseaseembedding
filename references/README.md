# References

This directory contains copied external code that the disease embedding experiments depend on or were derived from.

## Contents

- `poincare-embeddings/`
  - Local copy of the Facebook Research implementation associated with:
    - Nickel & Kiela (2017), "Poincare Embeddings for Learning Hierarchical Representations"

## How It Is Used

The scripts in:

- `../experiments/poincare_only/`
- `../experiments/poincare_hypstructure/`

call into the copied Poincare implementation for the original manifold math, data loader behavior, and Riemannian optimization path.

## Important Boundary

The reference code is not the main user-facing entry point for this repository.

Use the experiment scripts for:

- building disease-specific relation files
- training
- exporting aligned embeddings
- visualization
- evaluation
- tuning

The copied reference is kept here so the project remains reproducible and inspectable.

## Rebuild Note

Generated build artifacts are not the main version-controlled interface of this repository.

If the local Cython extensions need to be rebuilt after cloning, run the following inside `poincare-embeddings/`:

```bash
conda run -n reasoning python setup.py build_ext --inplace
```
