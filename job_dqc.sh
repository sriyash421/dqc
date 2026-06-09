#!/bin/bash
#SBATCH --job-name=dqc_bridge_soar_dqc
#SBATCH --qos=normal
#SBATCH --account=socialrl
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=960G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/projects/weirdlab/sriyash/dqc/slurm-%x-%j.out

set -euo pipefail

cd /gpfs/projects/weirdlab/sriyash/dqc

source .venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MPLCONFIGDIR="/tmp/matplotlib-${USER}"

bash experiments/scripts/launch_bridge_soar_dqc.sh
