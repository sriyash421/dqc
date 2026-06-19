#!/bin/bash
set -uo pipefail

# After the success-only resume finishes (GPU freed):
#   1) run the Q-quality eval on the remaining success+failure checkpoint (params_50000)
#   2) rerun the success+failure finetune from the Bridge checkpoint, KEEPING all checkpoints
# Usage: SUCCESS_ONLY_PID=<pid> bash experiments/scripts/queue_eval_and_rerun_sf.sh

cd "$(dirname "$0")/../.."  # dqc/ repo root
DQC_PY="${DQC_PY:-/home/harine/miniconda3/envs/dqc/bin/python}"
SUCCESS_ONLY_PID="${SUCCESS_ONLY_PID:-}"
SF_RUN_DIR="${SF_RUN_DIR:-/home/harine/RoboMonkey/dqc_eggplant_finetune_all/dqc/eggplant_dqc_finetune_all/sd00020260616_151541}"

export MPLCONFIGDIR="/tmp/matplotlib-${USER}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "[queue2] $(date) waiting for success-only PID=${SUCCESS_ONLY_PID:-none} to finish..."
if [ -n "${SUCCESS_ONLY_PID}" ]; then
  while kill -0 "${SUCCESS_ONLY_PID}" 2>/dev/null; do sleep 60; done
fi
echo "[queue2] $(date) success-only finished; GPU free."

echo "[queue2] $(date) === Q-eval on success+failure params_50000 ==="
"${DQC_PY}" experiments/visualize_bridge_soar_dqc_q.py \
  --checkpoint_dir="${SF_RUN_DIR}" \
  --restore_epoch=50000 \
  --output_dir=/home/harine/RoboMonkey/exp/eggplant_q_viz_sf_50000 \
  --k_episodes=8 --k_states=64 --q_agg=min --fps=10 --score_batch_size=64 \
  2>&1 | tail -40
echo "[queue2] $(date) eval done -> exp/eggplant_q_viz_sf_50000/metadata.json"

echo "[queue2] $(date) === rerun success+failure finetune (keep ALL checkpoints) ==="
PY="${DQC_PY}" \
DATA="experiments/configs/eggplant_all_chunk_data_config.py" \
SAVE_DIR="/home/harine/RoboMonkey/dqc_eggplant_finetune_all" \
RUN_GROUP="eggplant_dqc_finetune_all" \
TAGS="EggplantDQC,finetune,success_and_failure,rerun,keepckpts,h8,ac1" \
RESTORE_PATH="/home/harine/RoboMonkey/dqc_checkpoint/sd000s_137913.0.20260605_163017" \
RESTORE_EPOCH="100000" \
OFFLINE_STEPS="50000" \
SAVE_INTERVAL="10000" \
BATCH_SIZE="64" \
WANDB_OFFLINE="False" \
WANDB_ENTITY="l2sml" \
WANDB_PROJECT="dqc_finetune" \
WANDB_NAME="eggplant_success_fail" \
  bash experiments/scripts/launch_eggplant_dqc_finetune.sh

echo "[queue2] $(date) success+failure rerun exited."
