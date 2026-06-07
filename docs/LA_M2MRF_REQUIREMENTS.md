# LA-M2MRF Requirements

## Objective

Implement a new IDRiD-only experimental branch named LA-M2MRF
(Lesion-Aware M2MRF) on top of the M2MRF-C baseline for four-lesion fundus
segmentation:

- `1`: EX, Hard Exudates
- `2`: HE, Hemorrhages
- `3`: SE, Soft Exudates
- `4`: MA, Microaneurysms

## Baseline

- Base model: `M2MRF-C`
- Base dataset: `IDRiD`
- Base schedule: `40k` iterations
- Baseline reference config:
  `configs/m2mrf/fcn_hr48-M2MRF-C_40k_idrid_bdice.py`

## Phase-1 Scope

Only extend the IDRiD experiments first.

Reasons:

- IDRiD is smaller and better for pipeline debugging.
- The repo already uses a `40k` IDRiD schedule.
- DDR is larger and more expensive to curate and train.
- A single 16 GB GPU is better matched to the IDRiD bring-up stage.

## Planned Improvements

### 1. Lesion-aware patch sampling

Add a training crop transform that samples lesion-containing regions more
often, with extra preference for small lesions:

- lesion-aware sampling for any lesion class
- additional priority toward `SE` and `MA`
- fallback to standard random crop when no target lesion is found

### 2. Class-weighted lesion loss

Priority 1:

- `WeightedBinaryDiceLoss`

Optional if time allows:

- `FocalTverskyLoss`

Loss behavior requirements:

- preserve the original `BinaryLoss`
- keep background label `0` excluded from lesion channels
- compute per-lesion-channel loss for `EX/HE/SE/MA`
- support class weights such as `[1.0, 1.0, 1.5, 2.0]`

### 3. Fundus lesion enhancement preprocessing

Add a training pipeline transform for local contrast enhancement:

- green-channel CLAHE
- keep 3-channel image output
- no persistent rewrite of source images
- place the transform after `LoadImageFromFile` and before `Normalize`

## Required Experiment Configs

Create:

- `configs/la_m2mrf/fcn_hr48-M2MRF-C_40k_idrid_baseline_copy.py`
- `configs/la_m2mrf/fcn_hr48-M2MRF-C_40k_idrid_la_sampler.py`
- `configs/la_m2mrf/fcn_hr48-M2MRF-C_40k_idrid_weighted_dice.py`
- `configs/la_m2mrf/fcn_hr48-M2MRF-C_40k_idrid_fundus_enhance.py`
- `configs/la_m2mrf/fcn_hr48-M2MRF-C_40k_idrid_full.py`
- `configs/la_m2mrf/debug_fcn_hr48-M2MRF-C_idrid_full_2iter.py`

## Work Directory Policy

All new runs must write under:

- `work_dirs/la_m2mrf/idrid_baseline_copy/`
- `work_dirs/la_m2mrf/idrid_la_sampler/`
- `work_dirs/la_m2mrf/idrid_weighted_dice/`
- `work_dirs/la_m2mrf/idrid_fundus_enhance/`
- `work_dirs/la_m2mrf/idrid_full/`
- `work_dirs/la_m2mrf/debug/`

## Constraints

- Keep original baselines intact.
- Do not auto-download medical datasets.
- Do not assume `../data/IDRID` exists.
- Prefer Linux / WSL commands in docs and examples.
- Full training is out of scope for this development pass.
