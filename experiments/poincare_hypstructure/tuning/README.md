# Tuning

This directory stores hyperparameter tuning artifacts for the hybrid Poincare + HypStructure experiment track.

## Current Campaign

- `radius_separation/`
  - Multi-stage tuning campaign aimed at improving radius separation across tree depth levels.

## What To Expect

Inside the tuning campaign directories you will find:

- isolated run folders
- exported metrics
- offline checkpoint rescoring results
- stage summaries
- baseline calibration notes

## Recommended Entry Point

Use:

- `../scripts/run_radius_tuning_campaign.py`

to create or continue tuning runs in this directory.
