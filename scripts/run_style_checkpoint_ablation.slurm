#!/bin/bash
#SBATCH -J style_ckpt_ablate
#SBATCH -o logs/style_ckpt_ablate/%x-%j.out
#SBATCH -e logs/style_ckpt_ablate/%x-%j.err
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH -t 70:00:00

module load Anaconda3/2024.06
source ~/.bashrc
conda activate "${CONDA_ENV:-cloudnet}"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

mkdir -p "$PROJECT_ROOT/logs/style_ckpt_ablate"
bash "$PROJECT_ROOT/scripts/run_style_checkpoint_ablation.sh"
