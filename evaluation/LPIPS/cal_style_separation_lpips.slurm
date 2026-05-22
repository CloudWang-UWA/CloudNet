#!/bin/bash
#SBATCH -J lpips_sep
#SBATCH -o logs/lpips_sep/%x-%j.out
#SBATCH -e logs/lpips_sep/%x-%j.err
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH -t 10:00:00

module load Anaconda3/2024.06
source ~/.bashrc
conda activate "${CONDA_ENV:-cloudnet}"

RUN_NAME="${RUN_NAME:?Set RUN_NAME to the evaluation dataset run name}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BASE_DIR="${BASE_DIR:-$PROJECT_ROOT/evaluation/datasets_by_style_$RUN_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/evaluation/LPIPS/style_separation/$RUN_NAME}"
IMAGE_SUFFIX="${IMAGE_SUFFIX:-.png}"

mkdir -p "$PROJECT_ROOT/evaluation/LPIPS/logs/lpips_sep"
cd "$PROJECT_ROOT/evaluation/LPIPS"

python cal_style_separation_lpips.py \
  --base_dir "$BASE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --image_suffix "$IMAGE_SUFFIX"
