#!/usr/bin/env bash
set -euo pipefail

MIOU_CONFIG="${MIOU_CONFIG:-mmsegmentation/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_cloud.py}"
MIOU_CHECKPOINT="${MIOU_CHECKPOINT:-segmentation_models/pretrained/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth}"
MIOU_OUT_DIR="${MIOU_OUT_DIR:-output/mask2former/cloud/styleNet/classic}"
MIOU_WORK_DIR="${MIOU_WORK_DIR:-work_dirs/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_cloud/cloud/styleNet/classic}"
MIOU_CFG_OPTIONS="${MIOU_CFG_OPTIONS:-}"

extra_args=()
if [[ -n "$MIOU_CFG_OPTIONS" ]]; then
    read -r -a extra_args <<< "$MIOU_CFG_OPTIONS"
fi

python mmsegmentation/tools/test.py \
    "$MIOU_CONFIG" \
    "$MIOU_CHECKPOINT" \
    --out="$MIOU_OUT_DIR" \
    --work-dir="$MIOU_WORK_DIR" \
    "${extra_args[@]}"

# My Model
    # mmsegmentation/configs/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512_cloud.py \
    # segmentation_models/pretrained/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512_20221203_234230-7d64e5dd.pth \
    # --out="output/mask2former/cloud/new_model/van_gogh" \
    # --work-dir="work_dirs/mask2former_swin-t_8xb2-160k_ade20k-512x512_cloud/cloud/new_model/van_gogh"

# FreestyleNet
    # mmsegmentation/configs/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512_cloud.py \
    # segmentation_models/pretrained/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512_20221203_234230-7d64e5dd.pth \
    # --out="output/mask2former/freestyleNet/free_avg" \
    # --work-dir="work_dirs/mask2former_swin-t_8xb2-160k_ade20k-512x512_cloud/freestyleNet/free_avg"

# Real dataset
    # mmsegmentation/configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640_cloud.py \
    # segmentation_models/pretrained/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235230-7ec0f569.pth \
    # --out="output/mask2former/cloud/new_model/real" \
    # --work-dir="work_dirs/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640_cloud/cloud/new_model/real"

    # mmsegmentation/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_cloud.py \
    # segmentation_models/pretrained/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth \
    # --out="output/mask2former/freestyleNet/new_model/avg" \
    # --work-dir="work_dirs/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_cloud/freestyleNet/new_model/avg"
