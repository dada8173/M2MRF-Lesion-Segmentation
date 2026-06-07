# WSL Environment Setup for RTX 5070 Ti

## Current Recommendation

Use a modern, isolated Conda environment for development and preprocessing:

```bash
conda activate la-m2mrf
```

## Why This Exists

The original repo README targets:

- Python 3.7
- PyTorch 1.6.0
- CUDA 10.2
- mmsegmentation 0.8.0
- mmcv 1.2.0

That stack is not a good fit for an RTX 5070 Ti on WSL2.

## Verified Working Base

The following has been verified in this workspace:

- Python `3.10.20`
- PyTorch `2.11.0+cu130`
- torchvision `0.26.0+cu130`
- torchaudio `2.11.0+cu130`
- CUDA available in PyTorch: `True`
- Detected GPU: `NVIDIA GeForce RTX 5070 Ti`
- Detected capability: `(12, 0)`

## Installed Utility Packages

- `opencv-python`
- `scipy`
- `scikit-learn`
- `terminaltables`
- `matplotlib`
- `tensorboard`
- `tensorboardX`
- `ipykernel`

## Important Limitation

This does **not** automatically make the original M2MRF training stack
compatible.

The main blocker is the old OpenMMLab dependency chain:

- repo code depends on `mmsegmentation 0.8.0`
- that branch depends on old `mmcv`
- the old stack was designed around old PyTorch and old CUDA builds

So the GPU environment is now healthy, but the repo still needs either:

1. a legacy-compatible OpenMMLab environment, if one can be made to build, or
2. a careful port to a newer OpenMMLab stack

## Useful Commands

### Activate

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate la-m2mrf
```

### Quick GPU Check

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')
PY
```

### Use This Env for Later Dataset Handling

When the user provides a Windows-side zip path, handle it from WSL by copying or
extracting from `/mnt/<drive>/...` into the expected Linux-side dataset layout.
