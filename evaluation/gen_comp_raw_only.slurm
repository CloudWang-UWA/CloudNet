#!/bin/bash
#SBATCH -J comp_raw
#SBATCH -o logs/comp/%x-%j.out
#SBATCH -e logs/comp/%x-%j.err
#SBATCH -p work
#SBATCH --mem=32G
#SBATCH -t 10:00:00

# ==== Env ====
module load Anaconda3/2024.06
source ~/.bashrc
conda activate "${CONDA_ENV:-cloudnet}"

# ==== Paths ====
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_NAME="${RUN_NAME:-2026-05-02_style-ipadapter-token5_ckpt6500_controlnet97_100steps_scale0.8}"

cd "$PROJECT_ROOT"

# ==== Execute ====
python evaluation/gen_comp.py \
  --input_dir="$PROJECT_ROOT/evaluation/datasets_by_style_$RUN_NAME" \
  --freestylenet_dir="$PROJECT_ROOT/evaluation/freestylenet" \
  --semantic_masks_dir="$PROJECT_ROOT/image_generation/semantic_masks" \
  --raw_images_dir="$PROJECT_ROOT/evaluation/comparison/${RUN_NAME}_vs_freestylenet/raw_images" \
  --raw_only \
  --no_metrics
