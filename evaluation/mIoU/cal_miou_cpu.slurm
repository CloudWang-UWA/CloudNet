#!/bin/bash
#SBATCH -J training        # job name
#SBATCH -o logs/%x-%j.out        # output
#SBATCH -e logs/%x-%j.err        # error
#SBATCH -p work                   # GPU or CPU
#SBATCH --mem=16G                # CPU RAM
#SBATCH -t 10:00:00              # Max Time

# ==== Env ====
module load Anaconda3/2024.06
source ~/.bashrc
conda activate "${CONDA_ENV:-cloudnet}"

# ==== Paths ====
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT/evaluation/mIoU"

# ==== Execute ====
rm -rf ~/.cache/torch/kernels/
bash "$PROJECT_ROOT/evaluation/mIoU/cal_miou.sh"
