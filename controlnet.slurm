#!/bin/bash
#SBATCH -J indoor_ctrlnet        # job name
#SBATCH -o logs/ctrlnet/%x-%j.out        # output
#SBATCH -e logs/ctrlnet/%x-%j.err        # error
#SBATCH -p gpu                   # partition
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH --gres=gpu:1             # 1 gpu per node
#SBATCH --mem=16G                # CPU RAM
#SBATCH -t 72:00:00              # Max Time

# ==== Env ====
module load Anaconda3/2024.06
source ~/.bashrc
conda activate "${CONDA_ENV:-cloudnet}"

# ==== Paths ====
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_ROOT"

# ==== Execute ====
bash "$PROJECT_ROOT/controlnet.sh"
