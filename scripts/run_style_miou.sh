#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:?Set RUN_NAME to the generated image run name}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

MIOU_ROOT="$PROJECT_ROOT/evaluation/mIoU"
RESULT_ROOT="$MIOU_ROOT/${RUN_NAME}_swin-l"

MIOU_CONFIG="${MIOU_CONFIG:-configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_cloud_token5.py}"
MIOU_CHECKPOINT="${MIOU_CHECKPOINT:-segmentation_models/pretrained/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth}"
MIOU_DATASET_DIR="${MIOU_DATASET_DIR:-$PROJECT_ROOT/evaluation/datasets_by_style_$RUN_NAME}"
MIOU_LABEL_DIR="${MIOU_LABEL_DIR:-$PROJECT_ROOT/evaluation/mIoU/labels}"
MIOU_AVG_IMAGE_DIR="${MIOU_AVG_IMAGE_DIR:-$PROJECT_ROOT/image_generation/$RUN_NAME}"
MIOU_IMAGE_SUFFIX="${MIOU_IMAGE_SUFFIX:-.png}"
MIOU_SPLITS="${MIOU_SPLITS:-avg cozy messy classic luxurious van_gogh}"

if [[ ! -d "$MIOU_LABEL_DIR" ]]; then
  echo "mIoU label directory not found: $MIOU_LABEL_DIR" >&2
  exit 1
fi

mkdir -p "$RESULT_ROOT"
cd "$MIOU_ROOT"

for split in $MIOU_SPLITS; do
  if [[ "$split" == "avg" ]]; then
    image_dir="$MIOU_AVG_IMAGE_DIR"
  else
    image_dir="$MIOU_DATASET_DIR/$split"
  fi

  if [[ ! -d "$image_dir" ]]; then
    echo "mIoU image directory not found for $split: $image_dir" >&2
    exit 1
  fi

  echo "Running mIoU split: $split"
  echo "  images: $image_dir"
  echo "  labels: $MIOU_LABEL_DIR"

  # The HPC CUDA stack can leave invalid compiled kernels between Mask2Former
  # runs, so clear the cache before every split.
  rm -rf ~/.cache/torch/kernels/

  MIOU_CONFIG="$MIOU_CONFIG" \
  MIOU_CHECKPOINT="$MIOU_CHECKPOINT" \
  MIOU_OUT_DIR="$MIOU_ROOT/output/mask2former/$RUN_NAME/$split" \
  MIOU_WORK_DIR="$RESULT_ROOT/$split" \
  MIOU_CFG_OPTIONS="--cfg-options test_dataloader.dataset.data_prefix.img_path=$image_dir test_dataloader.dataset.data_prefix.seg_map_path=$MIOU_LABEL_DIR test_dataloader.dataset.img_suffix=$MIOU_IMAGE_SUFFIX test_dataloader.num_workers=0 test_dataloader.persistent_workers=False" \
  bash cal_miou.sh
done

echo "mIoU complete: $RESULT_ROOT"
