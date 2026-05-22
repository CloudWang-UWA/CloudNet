#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-2026-05-01_style-ipadapter-token5_controlnet97_100steps_scale0.8}"
ABLATION_DATE="${ABLATION_DATE:-$(date +%F)}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-1500 3000 5000 6500}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"
SCALE="${SCALE:-0.8}"
RUN_MIOU=0
RUN_COMPARISON="${RUN_COMPARISON:-0}"

export PROJECT_ROOT
export NUM_INFERENCE_STEPS
export SCALE
export RUN_MIOU
export RUN_COMPARISON

for step in $CHECKPOINT_STEPS; do
  STYLE_ADAPTER_CKPT="$PROJECT_ROOT/ip_adapter_models/indoor_models/$TRAIN_RUN_NAME/checkpoint-$step/ip_adapter_with_style.bin"
  if [[ ! -f "$STYLE_ADAPTER_CKPT" ]]; then
    echo "Checkpoint not found: $STYLE_ADAPTER_CKPT" >&2
    exit 1
  fi

  RUN_NAME="${ABLATION_DATE}_style-ipadapter-token5_ckpt${step}_controlnet97_${NUM_INFERENCE_STEPS}steps_scale${SCALE}"

  echo "Running checkpoint ablation: $RUN_NAME"
  RUN_NAME="$RUN_NAME" \
  STYLE_ADAPTER_CKPT="$STYLE_ADAPTER_CKPT" \
  bash "$PROJECT_ROOT/scripts/run_style_generation_eval.sh"
done

echo "Checkpoint ablation complete."
