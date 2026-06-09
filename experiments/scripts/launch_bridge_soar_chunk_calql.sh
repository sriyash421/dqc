#!/bin/bash
set -euo pipefail

SAVE_DIR="${SAVE_DIR:-/gpfs/projects/socialrl/sriyash/dqc_bridge_soar_chunk_calql_infinite_horizon}"
RUN_GROUP="${RUN_GROUP:-bridge_soar_chunk_calql_infinite_horizon}"
SEED="${SEED:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

.venv/bin/python experiments/train_bridge_soar_chunk.py \
  --run_group="${RUN_GROUP}" \
  --seed="${SEED}" \
  --save_dir="${SAVE_DIR}" \
  --offline_steps=1000000 \
  --log_interval=1000 \
  --save_interval=100000 \
  --agent=agents/bridge_calql_chunk.py \
  --data=experiments/configs/bridge_soar_chunk_data_config.py \
  --agent.backup_horizon=8 \
  --agent.policy_chunk_size=1 \
  --agent.use_chunk_critic=True \
  --agent.discount=0.98 \
  --agent.cql_alpha=0.05 --agent.cql_temp=1.0 \
  --agent.batch_size=256 \
  --data.backup_horizon=8 \
  --data.action_chunk_size=1 \
  --data.discount=0.98 \
  --data.infinite_horizon_backup=True \
  --data.batch_size=256 \
  --data.action_clip_eps=1e-5 \
  --data.num_workers=16 \
  --tags="BridgeSoarChunk,CALQL,infinite_horizon,h8,ac1"
