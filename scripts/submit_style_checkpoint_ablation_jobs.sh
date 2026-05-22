#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-1500 3000 5000 6500}"
ABLATION_DATE="${ABLATION_DATE:-$(date +%F)}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-2026-05-01_style-ipadapter-token5_controlnet97_100steps_scale0.8}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"
SCALE="${SCALE:-0.8}"

for step in $CHECKPOINT_STEPS; do
  echo "Submitting checkpoint ablation job: checkpoint-$step"
  CHECKPOINT_STEPS="$step" \
  ABLATION_DATE="$ABLATION_DATE" \
  TRAIN_RUN_NAME="$TRAIN_RUN_NAME" \
  NUM_INFERENCE_STEPS="$NUM_INFERENCE_STEPS" \
  SCALE="$SCALE" \
  sbatch scripts/run_style_checkpoint_ablation.slurm
done
