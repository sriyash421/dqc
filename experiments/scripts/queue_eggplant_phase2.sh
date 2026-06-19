#!/bin/bash
set -euo pipefail

# Queue the phase-2 eggplant finetune (success + failure; failures = reward 0).
# Waits for the phase-1 training process to exit and for the --filter all dataset
# conversion to finish, then launches phase 2 from the Bridge checkpoint.
#
# Usage: PHASE1_PID=<pid> bash experiments/scripts/queue_eggplant_phase2.sh

cd "$(dirname "$0")/../.."  # dqc/ repo root

PHASE1_PID="${PHASE1_PID:-}"
ALL_DATA_DIR="${ALL_DATA_DIR:-/home/harine/RoboMonkey/data/eggplant_all_lerobot}"
DQC_PY="${DQC_PY:-/home/harine/miniconda3/envs/dqc/bin/python}"

echo "[queue] $(date) waiting for phase-1 PID=${PHASE1_PID:-none} to finish..."
if [ -n "${PHASE1_PID}" ]; then
  while kill -0 "${PHASE1_PID}" 2>/dev/null; do sleep 60; done
fi
echo "[queue] $(date) phase-1 finished."

echo "[queue] waiting for full-eggplant conversion (${ALL_DATA_DIR}/labels.json)..."
while [ ! -f "${ALL_DATA_DIR}/labels.json" ]; do sleep 30; done
echo "[queue] $(date) conversion present."

export MPLCONFIGDIR="/tmp/matplotlib-${USER}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline

echo "[queue] $(date) launching phase-2 finetune (success+failure)."
PY="${DQC_PY}" \
DATA="experiments/configs/eggplant_all_chunk_data_config.py" \
SAVE_DIR="/home/harine/RoboMonkey/dqc_eggplant_finetune_all" \
RUN_GROUP="eggplant_dqc_finetune_all" \
TAGS="EggplantDQC,finetune,success_and_failure,h8,ac1" \
RESTORE_PATH="/home/harine/RoboMonkey/dqc_checkpoint/sd000s_137913.0.20260605_163017" \
RESTORE_EPOCH="100000" \
OFFLINE_STEPS="${OFFLINE_STEPS:-50000}" \
BATCH_SIZE="${BATCH_SIZE:-64}" \
  bash experiments/scripts/launch_eggplant_dqc_finetune.sh

echo "[queue] $(date) phase-2 finetune exited."
