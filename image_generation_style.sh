#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/image_generation/output}"
PROMPT_DIR="${PROMPT_DIR:-$PROJECT_ROOT/image_generation/prompts}"
VIS_DIR="${VIS_DIR:-$PROJECT_ROOT/image_generation/visualization}"
ORIGINAL_DIR="${ORIGINAL_DIR:-$PROJECT_ROOT/image_generation/original_images}"
SEMANTIC_MASKS_DIR="${SEMANTIC_MASKS_DIR:-$PROJECT_ROOT/image_generation/semantic_masks}"
STYLE_IMAGES_DIR="${STYLE_IMAGES_DIR:-$PROJECT_ROOT/image_generation/style_images}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
CONTROLNET_MODEL_PATH="${CONTROLNET_MODEL_PATH:-$PROJECT_ROOT/cwang_model/controlnet_97_classes}"
VAE_MODEL_PATH="${VAE_MODEL_PATH:-stabilityai/sd-vae-ft-mse}"
IMAGE_ENCODER_PATH="${IMAGE_ENCODER_PATH:-$PROJECT_ROOT/ip_adapter_models/models/image_encoder}"
IP_ADAPTER_PATH="${IP_ADAPTER_PATH:-$PROJECT_ROOT/ip_adapter_models/indoor_models/model_25_02_26/ip_adapter_with_style.bin}"

SCENE_LABEL_FILE="${SCENE_LABEL_FILE:-$PROJECT_ROOT/data/ade20k/test/data_ip_test.json}"

cd "$PROJECT_ROOT"

python "$PROJECT_ROOT/image_generation_style.py" \
    --output_folder=$OUTPUT_DIR \
    --prompt_folder=$PROMPT_DIR \
    --vis_folder=$VIS_DIR \
    --original_folder=$ORIGINAL_DIR \
    --semantic_masks_folder=$SEMANTIC_MASKS_DIR \
    --style_images_folder=$STYLE_IMAGES_DIR \
    --base_model_path=$BASE_MODEL_PATH \
    --controlnet_model_path=$CONTROLNET_MODEL_PATH \
    --vae_model_path=$VAE_MODEL_PATH \
    --image_encoder_path=$IMAGE_ENCODER_PATH \
    --ip_adapter_path=$IP_ADAPTER_PATH \
    --scene_label_file=$SCENE_LABEL_FILE \
    --num_inference_steps=100 \
    --num_generated_images=1 \
    --scale=0.8
