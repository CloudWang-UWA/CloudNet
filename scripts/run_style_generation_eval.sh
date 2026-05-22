#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-$(date +%F)_style-ipadapter-token5_controlnet97_100steps_scale0.8}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

STYLE_ADAPTER_CKPT="${STYLE_ADAPTER_CKPT:?Set STYLE_ADAPTER_CKPT to ip_adapter_with_style.bin}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
CONTROLNET_MODEL_PATH="${CONTROLNET_MODEL_PATH:-$PROJECT_ROOT/cwang_model/controlnet_97_classes}"
VAE_MODEL_PATH="${VAE_MODEL_PATH:-stabilityai/sd-vae-ft-mse}"
IMAGE_ENCODER_PATH="${IMAGE_ENCODER_PATH:-$PROJECT_ROOT/ip_adapter_models/models/image_encoder}"

NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"
SCALE="${SCALE:-0.8}"
STYLE_TOKEN_SCALE="${STYLE_TOKEN_SCALE:-1.0}"
NUM_GENERATED_IMAGES="${NUM_GENERATED_IMAGES:-1}"
RUN_MIOU="${RUN_MIOU:-0}"
RUN_COMPARISON="${RUN_COMPARISON:-0}"

OUTPUT_DIR="$PROJECT_ROOT/image_generation/$RUN_NAME"
PROMPT_DIR="$PROJECT_ROOT/image_generation/prompts_$RUN_NAME"
VIS_DIR="$PROJECT_ROOT/image_generation/visualization_$RUN_NAME"
DATASET_DIR="$PROJECT_ROOT/evaluation/datasets_by_style_$RUN_NAME"

CLIP_SCORE_DIR="$PROJECT_ROOT/evaluation/CLIP_score/$RUN_NAME"
CLIP_I_DIR="$PROJECT_ROOT/evaluation/CLIP_I/$RUN_NAME"
CLIP_AES_DIR="$PROJECT_ROOT/evaluation/CLIP_aes/$RUN_NAME"
COMPARISON_DIR="$PROJECT_ROOT/evaluation/comparison/$RUN_NAME/vs_freestylenet"
COMPARISON_RAW_DIR="$PROJECT_ROOT/evaluation/comparison/$RUN_NAME/raw_images"

echo "Running generation/evaluation: $RUN_NAME"
echo "Images: $OUTPUT_DIR"
echo "Dataset: $DATASET_DIR"

cd "$PROJECT_ROOT"

python image_generation_style.py \
  --output_folder="$OUTPUT_DIR" \
  --prompt_folder="$PROMPT_DIR" \
  --vis_folder="$VIS_DIR" \
  --original_folder="$PROJECT_ROOT/image_generation/original_images" \
  --semantic_masks_folder="$PROJECT_ROOT/image_generation/semantic_masks" \
  --style_images_folder="$PROJECT_ROOT/image_generation/style_images" \
  --base_model_path="$BASE_MODEL_PATH" \
  --controlnet_model_path="$CONTROLNET_MODEL_PATH" \
  --vae_model_path="$VAE_MODEL_PATH" \
  --image_encoder_path="$IMAGE_ENCODER_PATH" \
  --ip_adapter_path="$STYLE_ADAPTER_CKPT" \
  --scene_label_file="$PROJECT_ROOT/data/ade20k/test/data_ip_test.json" \
  --num_inference_steps="$NUM_INFERENCE_STEPS" \
  --num_generated_images="$NUM_GENERATED_IMAGES" \
  --scale="$SCALE" \
  --style_token_scale="$STYLE_TOKEN_SCALE"

python evaluation/gen_dataset_by_style.py \
  --model=Cloud \
  --ade20k_test_dir="$PROJECT_ROOT/data/ade20k/test/images" \
  --generated_images_dir="$OUTPUT_DIR" \
  --out_dir="$DATASET_DIR" \
  --scene_label_file="$PROJECT_ROOT/data/ade20k/test/data_ip_test.json"

(
  cd "$PROJECT_ROOT/evaluation/CLIP_score"
  python clip_score.py \
    --base_dir="$DATASET_DIR" \
    --prompts_file="$PROMPT_DIR/prompts.txt" \
    --output_dir="$CLIP_SCORE_DIR"
)

(
  cd "$PROJECT_ROOT/evaluation/CLIP_I"
  python clip_i.py \
    --base_dir="$DATASET_DIR" \
    --style_dir="$PROJECT_ROOT/image_generation/style_images" \
    --output_dir="$CLIP_I_DIR"
)

(
  cd "$PROJECT_ROOT/evaluation/CLIP_aes"
  python clip_aes.py \
    --model=Cloud \
    --base_dir="$DATASET_DIR" \
    --output_dir="$CLIP_AES_DIR" \
    --clip_dir="$PROJECT_ROOT/clip/models"
)

if [[ "$RUN_MIOU" == "1" ]]; then
  echo "Running optional mIoU."
  RUN_NAME="$RUN_NAME" \
  PROJECT_ROOT="$PROJECT_ROOT" \
  MIOU_DATASET_DIR="$DATASET_DIR" \
  MIOU_AVG_IMAGE_DIR="$OUTPUT_DIR" \
  MIOU_IMAGE_SUFFIX=".png" \
  bash "$PROJECT_ROOT/scripts/run_style_miou.sh"
fi

if [[ "$RUN_COMPARISON" == "1" ]]; then
  echo "Generating optional FreestyleNet comparison images."
  python evaluation/gen_comp.py \
    --project_root="$PROJECT_ROOT" \
    --input_dir="$DATASET_DIR" \
    --freestylenet_dir="$PROJECT_ROOT/evaluation/datasets_by_style_freestylenet" \
    --semantic_masks_dir="$PROJECT_ROOT/image_generation/semantic_masks" \
    --clip_result_file="$CLIP_SCORE_DIR/clip_per_image_result.txt" \
    --clip_result_file_free_net="$PROJECT_ROOT/evaluation/CLIP_score/freestylenet/clip_per_image_result.txt" \
    --clip_aes_file="$CLIP_AES_DIR/clip_aesthetic_all_images.txt" \
    --clip_aes_file_free_net="$PROJECT_ROOT/evaluation/CLIP_aes/freestylenet/clip_aesthetic_all_images.txt" \
    --vis_dir="$COMPARISON_DIR" \
    --raw_images_dir="$COMPARISON_RAW_DIR"
fi

echo "Done: $RUN_NAME"
