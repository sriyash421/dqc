#!/bin/bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

.venv/bin/python experiments/train_bridge_soar_dqc.py \
  --run_group=bridge_soar_dqc_infinite_horizon \
  --save_dir=/gpfs/projects/socialrl/sriyash/dqc_bridge_soar_dqc_infinite_horizon \
  --offline_steps=1000000 \
  --log_interval=1000 \
  --save_interval=100000 \
  --agent=agents/bridge_dqc_chunk.py \
  --data=experiments/configs/bridge_soar_chunk_data_config.py \
  --agent.backup_horizon=8 \
  --agent.policy_chunk_size=1 \
  --agent.discount=0.98 \
  --agent.lr=3e-4 \
  --agent.target_update_rate=5e-3 \
  --agent.num_critics=2 \
  --agent.backup_quantile=0.9 \
  --agent.distill_expectile=0.8 \
  --agent.best_of_n=32 \
  --agent.batch_size=256 \
  --data.backup_horizon=8 \
  --data.action_chunk_size=1 \
  --data.discount=0.98 \
  --data.infinite_horizon_backup=True \
  --data.batch_size=256 \
  --data.action_clip_eps=1e-5 \
  --data.num_workers=16 \
  --tags="BridgeSoarDQC,infinite_horizon,h8,ac1"
