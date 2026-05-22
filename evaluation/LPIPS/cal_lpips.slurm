#!/bin/bash
#SBATCH -J training        # job name
#SBATCH -o logs/ctrlnet/%x-%j.out        # output
#SBATCH -e logs/ctrlnet/%x-%j.err        # error
#SBATCH -p gpu                   # partition
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH --gres=gpu:2             # 2 gpu per node
#SBATCH --mem=32G                # CPU RAM
#SBATCH -t 10:00:00              # Max Time

# ==== Env ====
module load Anaconda3/2024.06
source ~/.bashrc
conda activate "${CONDA_ENV:-cloudnet}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

# ==== Paths ====
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT/evaluation/LPIPS"

# ==== Execute ====
python "$PROJECT_ROOT/evaluation/LPIPS/cal_lpips.py"
