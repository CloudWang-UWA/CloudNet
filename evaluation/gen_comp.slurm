#!/bin/bash
#SBATCH -J training        # job name
#SBATCH -o logs/comp/%x-%j.out        # output
#SBATCH -e logs/comp/%x-%j.err        # error
#SBATCH -p work                   # GPU or CPU
#SBATCH --mem=32G                # CPU RAM
#SBATCH -t 10:00:00              # Max Time

# ==== Env ====
module load Anaconda3/2024.06
source ~/.bashrc
conda activate "${CONDA_ENV:-cloudnet}"

# ==== Paths ====
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT/evaluation"

# ==== Execute ====
# python indoor_demo.py
python "$PROJECT_ROOT/evaluation/gen_comp.py"
