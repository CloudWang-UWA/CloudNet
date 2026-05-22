#!/usr/bin/env bash
set -euo pipefail

STYLE_TOKEN_SCALES="${STYLE_TOKEN_SCALES:-0.5 0.8 1.2 1.5}"
ABLATION_DATE="${ABLATION_DATE:-$(date +%F)}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-2026-05-01_style-ipadapter-token5_controlnet97_100steps_scale0.8}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-6500}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"
SCALE="${SCALE:-0.8}"
STAGGER_HOURS="${STAGGER_HOURS:-3}"

delay=0
for style_token_scale in $STYLE_TOKEN_SCALES; do
  echo "Submitting style-token-scale ablation job: style_token_scale=$style_token_scale, begin=now+${delay}hours"

  sbatch_args=()
  if [[ "$delay" -gt 0 ]]; then
    sbatch_args+=(--begin="now+${delay}hours")
  fi

  STYLE_TOKEN_SCALES="$style_token_scale" \
  ABLATION_DATE="$ABLATION_DATE" \
  TRAIN_RUN_NAME="$TRAIN_RUN_NAME" \
  CHECKPOINT_STEP="$CHECKPOINT_STEP" \
  NUM_INFERENCE_STEPS="$NUM_INFERENCE_STEPS" \
  SCALE="$SCALE" \
  sbatch "${sbatch_args[@]}" scripts/run_style_token_scale_ablation.slurm

  delay=$((delay + STAGGER_HOURS))
done
