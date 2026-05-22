#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-$(date +%F)_style-ipadapter-token5}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
IMAGE_ENCODER_PATH="${IMAGE_ENCODER_PATH:-$PROJECT_ROOT/ip_adapter_models/models/image_encoder}"
PRETRAINED_IP_ADAPTER_PATH="${PRETRAINED_IP_ADAPTER_PATH:-$PROJECT_ROOT/ip_adapter_models/models/ip-adapter_sd15.bin}"
DATA_JSON_FILE="${DATA_JSON_FILE:-$PROJECT_ROOT/train_ip_adapter/training_data.json}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:-$PROJECT_ROOT/train_ip_adapter/training_images}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/ip_adapter_models/indoor_models/$RUN_NAME}"

MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
RESOLUTION="${RESOLUTION:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-100}"
SAVE_STEPS="${SAVE_STEPS:-500}"

echo "Training style IP-Adapter run: $RUN_NAME"
echo "Output: $OUTPUT_DIR"

cd "$PROJECT_ROOT/train_ip_adapter"

accelerate launch train_ip_adapter_style.py \
  --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
  --pretrained_ip_adapter_path="$PRETRAINED_IP_ADAPTER_PATH" \
  --image_encoder_path="$IMAGE_ENCODER_PATH" \
  --data_json_file="$DATA_JSON_FILE" \
  --data_root_path="$DATA_ROOT_PATH" \
  --mixed_precision="$MIXED_PRECISION" \
  --resolution="$RESOLUTION" \
  --train_batch_size="$TRAIN_BATCH_SIZE" \
  --dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
  --learning_rate="$LEARNING_RATE" \
  --weight_decay="$WEIGHT_DECAY" \
  --num_train_epochs="$NUM_TRAIN_EPOCHS" \
  --output_dir="$OUTPUT_DIR" \
  --save_steps="$SAVE_STEPS"
