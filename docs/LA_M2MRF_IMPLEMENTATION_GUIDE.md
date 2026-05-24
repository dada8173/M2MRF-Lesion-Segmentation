# LA-M2MRF Implementation Guide

## Design Principles

- Preserve the upstream M2MRF baseline.
- Implement LA-M2MRF as additive extensions.
- Keep new logic isolated in `la_m2mrf` modules and configs.
- Make all debug and ablation runs explicit and reproducible.

## File Layout

- Configs: `configs/la_m2mrf/`
- Pipeline transforms:
  `mmseg/datasets/pipelines/la_m2mrf_transforms.py`
- Losses:
  `mmseg/models/losses/la_m2mrf_losses.py`
- Docs:
  `docs/LA_M2MRF_*.md`
- Generated report figures:
  `work_dirs/la_m2mrf/report_figures/` by default

## Config Naming

- `*_baseline_copy.py`: exact baseline copy for comparison
- `*_la_sampler.py`: lesion-aware crop only
- `*_weighted_dice.py`: weighted dice only
- `*_fundus_enhance.py`: CLAHE enhancement only
- `*_full.py`: all enabled improvements together
- `debug_*_2iter.py`: smoke-test-only config

## Expected Pipeline Placement

### LesionAwareRandomCrop

Replace the baseline `RandomCrop` in train pipeline only.

### FundusCLAHEEnhancement

Insert after `LoadImageFromFile` and before `Normalize`.
Typical placement in train pipeline:

1. `LoadImageFromFile`
2. `LoadAnnotations`
3. `Resize`
4. `LesionAwareRandomCrop` or `RandomCrop`
5. `RandomFlip`
6. `PhotoMetricDistortion`
7. `FundusCLAHEEnhancement`
8. `Normalize`
9. `Pad`
10. `DefaultFormatBundle`
11. `Collect`

## Smoke Test Commands

Run from repo root:

```bash
python -m compileall mmseg configs tools
python tools/print_config.py configs/la_m2mrf/fcn_hr48-M2MRF-C_40k_idrid_baseline_copy.py
python tools/print_config.py configs/la_m2mrf/fcn_hr48-M2MRF-C_40k_idrid_full.py
python tools/print_config.py configs/la_m2mrf/debug_fcn_hr48-M2MRF-C_idrid_full_2iter.py
```

## Debug Training Command

Use this only after the IDRiD dataset is prepared:

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/train.py configs/la_m2mrf/debug_fcn_hr48-M2MRF-C_idrid_full_2iter.py \
  --work-dir work_dirs/la_m2mrf/debug/full_2iter \
  --seed 0 --deterministic
```

## Environment Notes

- Primary workflow target: WSL2 Ubuntu
- Prefer repo paths like `~/projects/M2MRF-Lesion-Segmentation`
- Do not assume legacy `PyTorch 1.6.0 + CUDA 10.2` can train on RTX 5070 Ti
  without environment changes
- Keep environment-risk notes separate from code changes
- Report scripts auto-detect a usable Chinese font when possible; a repo-local
  `NotoSansTC-Regular.otf` is optional rather than required
