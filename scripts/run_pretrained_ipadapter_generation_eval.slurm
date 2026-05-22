#!/bin/bash
#SBATCH -J ipadapter_eval
#SBATCH -o logs/ipadapter_eval/%x-%j.out
#SBATCH -e logs/ipadapter_eval/%x-%j.err
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

mkdir -p "$PROJECT_ROOT/logs/ipadapter_eval"
bash "$PROJECT_ROOT/scripts/run_pretrained_ipadapter_generation_eval.sh"
