#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MODEL_DIR="${MODEL_DIR:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/controlnet/indoor_output/current_model}"
DATASET_FILE="${DATASET_FILE:-$PROJECT_ROOT/data/ade20k/dataset_file.json}"
VALIDATE_FILE="${VALIDATE_FILE:-$PROJECT_ROOT/data/ade20k/validation/dataset_file.json}"

cd "$PROJECT_ROOT"

accelerate launch "$PROJECT_ROOT/controlnet/train.py" \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --inference_basemodel_path=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --dataset_file=$DATASET_FILE \
    --learning_rate=1e-05 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --num_train_epochs=35 \
    --checkpointing_steps=2000 \
    --validation_steps=1000 \
    --report_to='wandb' \
    --tracker_project_name='indoor_training' \
    --dataset_type='ADE20K' \
    --mixed_precision="fp16" \
    --validate_file=$VALIDATE_FILE
