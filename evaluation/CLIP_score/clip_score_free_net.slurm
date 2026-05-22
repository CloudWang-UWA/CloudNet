#!/bin/bash
#SBATCH -J training        # job name
#SBATCH -o logs/clip_score/%x-%j.out        # output
#SBATCH -e logs/clip_score/%x-%j.err        # error
#SBATCH -p gpu                   # GPU or CPU
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH --gres=gpu:v100:2        # 2 gpu per node
#SBATCH --mem=32G                # CPU RAM
#SBATCH -t 10:00:00              # Max Time

# ==== Env ====
module load Anaconda3/2024.06
source ~/.bashrc
conda activate "${CONDA_ENV:-cloudnet}"

# ==== Paths ====
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT/evaluation/CLIP_score"

# ==== Execute ====
python "$PROJECT_ROOT/evaluation/CLIP_score/clip_score_free_net.py"
