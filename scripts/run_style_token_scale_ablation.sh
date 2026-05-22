#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-2026-05-01_style-ipadapter-token5_controlnet97_100steps_scale0.8}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-6500}"
ABLATION_DATE="${ABLATION_DATE:-$(date +%F)}"
STYLE_TOKEN_SCALES="${STYLE_TOKEN_SCALES:-0.5 0.8 1.2 1.5}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"
SCALE="${SCALE:-0.8}"
RUN_MIOU="${RUN_MIOU:-1}"
RUN_COMPARISON="${RUN_COMPARISON:-0}"

STYLE_ADAPTER_CKPT="${STYLE_ADAPTER_CKPT:-$PROJECT_ROOT/ip_adapter_models/indoor_models/$TRAIN_RUN_NAME/checkpoint-$CHECKPOINT_STEP/ip_adapter_with_style.bin}"
if [[ ! -f "$STYLE_ADAPTER_CKPT" ]]; then
  echo "Checkpoint not found: $STYLE_ADAPTER_CKPT" >&2
  exit 1
fi

export PROJECT_ROOT
export STYLE_ADAPTER_CKPT
export NUM_INFERENCE_STEPS
export SCALE
export RUN_MIOU
export RUN_COMPARISON

for style_token_scale in $STYLE_TOKEN_SCALES; do
  RUN_NAME="${ABLATION_DATE}_style-ipadapter-token5_ckpt${CHECKPOINT_STEP}_styleScale${style_token_scale}_controlnet97_${NUM_INFERENCE_STEPS}steps_scale${SCALE}"

  echo "Running style-token-scale ablation: $RUN_NAME"
  RUN_NAME="$RUN_NAME" \
  STYLE_TOKEN_SCALE="$style_token_scale" \
  bash "$PROJECT_ROOT/scripts/run_style_generation_eval.sh"
done

echo "Style-token-scale ablation complete."
