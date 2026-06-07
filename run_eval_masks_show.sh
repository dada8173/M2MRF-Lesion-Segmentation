#!/bin/bash
set -e

declare -A EXPERIMENTS=(
    ["Control"]="fcn_hr48-M2MRF-C_10k_idrid_finetune_control"
    ["LA-M2MRF-S"]="fcn_hr48-M2MRF-C_10k_idrid_la_sampler_finetune"
    ["LA-M2MRF-W"]="fcn_hr48-M2MRF-C_10k_idrid_weighted_dice_finetune"
    ["LA-M2MRF-SW"]="fcn_hr48-M2MRF-C_10k_idrid_la_sampler_weighted_dice_finetune"
)

for NAME in "${!EXPERIMENTS[@]}"; do
    CFG_NAME="${EXPERIMENTS[$NAME]}"
    CONFIG="configs/la_m2mrf/${CFG_NAME}.py"
    CKPT="work_dirs/la_m2mrf/${CFG_NAME}/latest.pth"
    OUT_DIR="work_dirs/la_m2mrf/masks/${NAME}"
    mkdir -p "$OUT_DIR"
    
    echo "Running inference for $NAME..."
    python tools/test.py "$CONFIG" "$CKPT" --show-dir "$OUT_DIR"
done

