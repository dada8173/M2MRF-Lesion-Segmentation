# AGENTS.md

## Scope

This repository contains the official M2MRF research codebase. New LA-M2MRF
work must preserve the original baseline behavior and remain easy to diff
against the upstream implementation.

## Non-destructive Rules

- Do not overwrite the original M2MRF baseline configs.
- Do not modify `mmseg/models/utils/m2mrf.py` unless absolutely necessary.
- Do not modify `mmseg/models/backbones/hrnet.py`.
- Do not modify `tools/prepare_labels.py` or `tools/augment.py`.
- Keep all new experiment outputs under `work_dirs/la_m2mrf/`.
- Never commit datasets, generated labels, checkpoints, or large artifacts.

## LA-M2MRF Naming Rules

- Prefix all new experiment-specific content with `la_m2mrf` where practical.
- Put new configs under `configs/la_m2mrf/`.
- Put new dataset transforms in
  `mmseg/datasets/pipelines/la_m2mrf_transforms.py`.
- Put new losses in `mmseg/models/losses/la_m2mrf_losses.py`.
- Only add import registrations to `__init__.py` files. Do not rewrite their
  existing logic.

## Implementation Expectations

- Preserve compatibility with the repo's legacy stack:
  PyTorch 1.6.0, MMCV 1.2.0, MMSegmentation 0.8.0.
- Prefer additive changes over in-place mutation of baseline code.
- Keep debug configs separate from formal experiment configs.
- If IDRiD data is missing, surface the message:
  `IDRiD data not found at ../data/IDRID. Please follow docs/DATASET_SETUP_IDRID.md.`

## Validation Expectations

- Required smoke test without data:
  `python -m compileall mmseg configs tools`
- Preferred extra checks:
  `python tools/print_config.py <config>`
- Do not claim training success without the dataset.

## Target Workflow

- Primary dev environment is WSL2 Ubuntu on Linux filesystem paths such as
  `~/projects/M2MRF-Lesion-Segmentation`.
- Prefer single-GPU debug commands.
- Full training is user-run only.
