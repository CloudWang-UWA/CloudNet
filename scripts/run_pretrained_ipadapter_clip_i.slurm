#!/bin/bash
#SBATCH -J pretrained_clip_i
#SBATCH -o logs/pretrained_clip_i/%x-%j.out
#SBATCH -e logs/pretrained_clip_i/%x-%j.err
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH -t 10:00:00

module load Anaconda3/2024.06
source ~/.bashrc
conda activate "${CONDA_ENV:-cloudnet}"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

mkdir -p "$PROJECT_ROOT/logs/pretrained_clip_i"
bash "$PROJECT_ROOT/scripts/run_pretrained_ipadapter_clip_i.sh"
