# IDRiD Dataset Setup

## Important

This repo should not auto-download the IDRiD dataset. The dataset requires
manual access and organization by the user.

If training reports:

`IDRiD data not found at ../data/IDRID. Please follow docs/DATASET_SETUP_IDRID.md.`

prepare the dataset layout below first.

## Source

Download IDRiD manually from the official source referenced in the repo
README:

- https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid

## Expected Repo-relative Location

From repo root:

```text
../data/IDRID
```

For a repo at:

```text
~/projects/M2MRF-Lesion-Segmentation
```

the dataset root should be:

```text
~/projects/data/IDRID
```

## Expected Directory Layout

The training config expects:

```text
../data/IDRID/
  image/
    train/
    test/
  label/
    train/
      annotations/
    test/
      annotations/
```

## Preparation Notes

The original README indicates the canonical preprocessing flow is:

```bash
python tools/prepare_labels.py
python tools/augment.py
```

Do not run those commands until the raw IDRiD files have been downloaded and
placed in the expected source locations required by the original repo.

## Validation

Useful Linux checks:

```bash
ls ../data/IDRID
find ../data/IDRID/image/train -maxdepth 1 -type f | head
find ../data/IDRID/label/train/annotations -maxdepth 1 -type f | head
```

## First Debug Run

After the dataset is ready:

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/train.py configs/la_m2mrf/debug_fcn_hr48-M2MRF-C_idrid_full_2iter.py \
  --work-dir work_dirs/la_m2mrf/debug/full_2iter \
  --seed 0 --deterministic
```
