# LA-M2MRF Changelog

## 2026-05-17

### Added

- Initial LA-M2MRF development docs:
  `AGENTS.md`,
  `docs/LA_M2MRF_REQUIREMENTS.md`,
  `docs/LA_M2MRF_IMPLEMENTATION_GUIDE.md`,
  `docs/DATASET_SETUP_IDRID.md`,
  `docs/LA_M2MRF_CHANGELOG.md`
- New LA-M2MRF dataset pipeline module:
  `mmseg/datasets/pipelines/la_m2mrf_transforms.py`
- New LA-M2MRF losses:
  `mmseg/models/losses/la_m2mrf_losses.py`
- New ablation and debug configs under `configs/la_m2mrf/`
- WSL environment note:
  `docs/WSL_ENV_SETUP_RTX5070TI.md`

### Notes

- Baseline review completed before implementation.
- Original M2MRF baseline files remain untouched.
- Added a dataset-root preflight check in `tools/train.py` so missing IDRiD
  data now raises the required guidance message.
- `python -m compileall mmseg configs tools` succeeded.
- `python tools/print_config.py ...` is currently blocked by missing `mmcv` in
  the active environment.
- Created a modern Conda environment `la-m2mrf` with Python 3.10 and verified
  PyTorch 2.11.0 + CUDA 13.0 wheels can use the RTX 5070 Ti under WSL2.

## 2026-05-18

### Added

- Added finetune config:
  `configs/la_m2mrf/fcn_hr48-M2MRF-C_10k_idrid_full_finetune.py`
- Added scratch reproduction config for the official baseline under the local
  runtime recipe:
  `configs/la_m2mrf/fcn_hr48-M2MRF-C_40k_idrid_official_scratch_bs3_fp16.py`
- Added scratch config that combines the official runtime recipe with the
  `la_sampler + weighted_dice` method:
  `configs/la_m2mrf/fcn_hr48-M2MRF-C_40k_idrid_la_sampler_weighted_dice_scratch_bs3_fp16.py`

### Notes

- Verified the official `M2MRF-C` IDRiD checkpoint reproduces
  `mIoU=50.17 / mAUPR=67.55` on the local test pipeline.
- Confirmed the current local retraining setup does not reproduce the official
  baseline, so finetuning from the official checkpoint was introduced as a more
  reliable next-step experiment for LA-M2MRF.
- Added a clean scratch-training entrypoint that keeps the baseline method
  unchanged and only switches runtime knobs to `samples_per_gpu=3`,
  `workers_per_gpu=0`, and `Fp16OptimizerHook`.
