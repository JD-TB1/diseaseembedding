# Legacy Root Workspace

This directory preserves the pre-reorganization script layout that existed before the repository was split into experiment-specific tracks.

## What Remains

- `scripts/`
  - legacy disease-90 pipeline scripts from the old top-level workflow
- this README

## What Was Removed

The archive was intentionally slimmed down. Removed from version control:

- logs
- copied metadata already preserved in maintained experiment tracks
- generated result trees
- duplicated plots
- checkpoints

## Why It Is Kept

It preserves script provenance and naming history for anyone who needs to understand how the current experiment tracks evolved.

## How To Use It

Do not start new work here.

Use the maintained tracks instead:

- `../../experiments/poincare_only/`
- `../../experiments/poincare_hypstructure/`
