#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-$(date +%F)_style-ipadapter-token5_controlnet97_100steps_scale0.8}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SAVE_STEPS="${SAVE_STEPS:-500}"
RUN_MIOU="${RUN_MIOU:-1}"

export RUN_NAME
export PROJECT_ROOT
export SAVE_STEPS

bash "$PROJECT_ROOT/scripts/train_style_ip_adapter.sh"

STYLE_ADAPTER_CKPT="${STYLE_ADAPTER_CKPT:-$PROJECT_ROOT/ip_adapter_models/indoor_models/$RUN_NAME/checkpoint-$SAVE_STEPS/ip_adapter_with_style.bin}"
if [[ ! -f "$STYLE_ADAPTER_CKPT" ]]; then
  echo "Expected checkpoint not found: $STYLE_ADAPTER_CKPT" >&2
  echo "Set STYLE_ADAPTER_CKPT manually or increase/decrease SAVE_STEPS to match the saved checkpoint." >&2
  exit 1
fi

export STYLE_ADAPTER_CKPT
bash "$PROJECT_ROOT/scripts/run_style_generation_eval.sh"

if [[ "$RUN_MIOU" == "1" ]]; then
  MIOU_DATASET_DIR="$PROJECT_ROOT/evaluation/datasets_by_style_$RUN_NAME" \
  MIOU_AVG_IMAGE_DIR="$PROJECT_ROOT/image_generation/$RUN_NAME" \
  MIOU_IMAGE_SUFFIX=".png" \
  bash "$PROJECT_ROOT/scripts/run_style_miou.sh"
else
  echo "Skipping mIoU because RUN_MIOU=$RUN_MIOU"
fi
