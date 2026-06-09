#!/bin/bash
#SBATCH --job-name=dqc_bridge_soar_calql_resume
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

.venv/bin/python experiments/train_bridge_soar_chunk.py \
  --run_group="bridge_soar_chunk_calql" \
  --seed=0 \
  --save_dir="/gpfs/projects/socialrl/sriyash/dqc_bridge_soar_chunk_calql" \
  --restore_path="/gpfs/projects/socialrl/sriyash/dqc_bridge_soar_chunk_calql/dqc/bridge_soar_chunk_calql/sd000s_124194.0.20260526_221347" \
  --restore_epoch=100000 \
  --offline_steps=899999 \
  --log_interval=1000 \
  --save_interval=100000 \
  --agent=agents/bridge_calql_chunk.py \
  --data=experiments/configs/bridge_soar_chunk_data_config.py \
  --agent.backup_horizon=8 \
  --agent.policy_chunk_size=1 \
  --agent.use_chunk_critic=True \
  --agent.discount=0.98 \
  --agent.cql_alpha=5.0 \
  --agent.batch_size=256 \
  --data.backup_horizon=8 \
  --data.action_chunk_size=1 \
  --data.discount=0.98 \
  --data.batch_size=256 \
  --data.num_workers=16 \
  --tags="BridgeSoarChunk,CALQL,h8,ac1"
