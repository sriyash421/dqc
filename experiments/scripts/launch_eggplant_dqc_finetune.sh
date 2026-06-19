#!/bin/bash
set -euo pipefail

# Continue-finetune the Bridge-pretrained DQC verifier on the eggplant-in-basket task.
# Phase 1: success-only eggplant data (see eggplant_chunk_data_config.py).
# Run from the dqc/ repo root with the `dqc` conda env active (GPU jaxlib installed).

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

RESTORE_PATH="${RESTORE_PATH:-/home/harine/RoboMonkey/dqc_checkpoint/sd000s_137913.0.20260605_163017}"
RESTORE_EPOCH="${RESTORE_EPOCH:-100000}"
SAVE_DIR="${SAVE_DIR:-/home/harine/RoboMonkey/dqc_eggplant_finetune}"
RUN_GROUP="${RUN_GROUP:-eggplant_dqc_finetune}"
DATA="${DATA:-experiments/configs/eggplant_chunk_data_config.py}"
TAGS="${TAGS:-EggplantDQC,finetune,success_only,h8,ac1}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OFFLINE_STEPS="${OFFLINE_STEPS:-50000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
WANDB_OFFLINE="${WANDB_OFFLINE:-True}"
PY="${PY:-python}"

"${PY}" experiments/train_bridge_soar_dqc.py \
  --run_group="${RUN_GROUP}" \
  --seed="${SEED}" \
  --save_dir="${SAVE_DIR}" \
  --restore_path="${RESTORE_PATH}" \
  --restore_epoch="${RESTORE_EPOCH}" \
  --offline_steps="${OFFLINE_STEPS}" \
  --log_interval=500 \
  --save_interval="${SAVE_INTERVAL}" \
  --agent=agents/bridge_dqc_chunk.py \
  --data="${DATA}" \
  --agent.backup_horizon=8 \
  --agent.policy_chunk_size=1 \
  --agent.discount=0.98 \
  --agent.lr=1e-4 \
  --agent.target_update_rate=5e-3 \
  --agent.num_critics=2 \
  --agent.backup_quantile=0.9 \
  --agent.distill_expectile=0.8 \
  --agent.best_of_n=32 \
  --agent.batch_size="${BATCH_SIZE}" \
  --data.backup_horizon=8 \
  --data.action_chunk_size=1 \
  --data.discount=0.98 \
  --data.infinite_horizon_backup=True \
  --data.batch_size="${BATCH_SIZE}" \
  --data.action_clip_eps=1e-5 \
  --data.num_workers=8 \
  --wandb_offline="${WANDB_OFFLINE}" \
  --tags="${TAGS}"
