#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUN_NAME="${RUN_NAME:-2026-01-09_ipadapter_sd15_controlnet97_70steps}"
GENERATED_IMAGES_DIR="${GENERATED_IMAGES_DIR:-$PROJECT_ROOT/image_generation/2026-01-09_ipadapter_sd15_controlnet_indoor_70steps}"
EVALUATION_IMAGE_DIR="$PROJECT_ROOT/evaluation/$RUN_NAME"
if [[ -z "${DATASET_DIR:-}" ]]; then
  if [[ -d "$EVALUATION_IMAGE_DIR" ]]; then
    DATASET_DIR="$EVALUATION_IMAGE_DIR"
  else
    DATASET_DIR="$PROJECT_ROOT/evaluation/datasets_by_style_$RUN_NAME"
  fi
fi
CLIP_I_DIR="${CLIP_I_DIR:-$PROJECT_ROOT/evaluation/CLIP_I/$RUN_NAME}"

echo "Running CLIP-I for pretrained IP-Adapter baseline: $RUN_NAME"
echo "Generated images: $GENERATED_IMAGES_DIR"
echo "Dataset: $DATASET_DIR"
echo "CLIP-I output: $CLIP_I_DIR"

cd "$PROJECT_ROOT"

if [[ ! -d "$DATASET_DIR" ]]; then
  python evaluation/gen_dataset_by_style.py \
    --model=Cloud \
    --ade20k_test_dir="$PROJECT_ROOT/data/ade20k/test/images" \
    --generated_images_dir="$GENERATED_IMAGES_DIR" \
    --out_dir="$DATASET_DIR" \
    --scene_label_file="$PROJECT_ROOT/data/ade20k/test/data_ip_test.json"
fi

(
  cd "$PROJECT_ROOT/evaluation/CLIP_I"
  python clip_i.py \
    --base_dir="$DATASET_DIR" \
    --style_dir="$PROJECT_ROOT/image_generation/style_images" \
    --output_dir="$CLIP_I_DIR"
)

echo "Done: $CLIP_I_DIR/clip_avg_result.txt"
cat "$CLIP_I_DIR/clip_avg_result.txt"
