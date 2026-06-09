#!/bin/bash


SAVE_DIR="/gpfs/projects/socialrl/sriyash/dqc_bridge_soar_chunk_calql_debug"
RUN_GROUP="${RUN_GROUP:-bridge_soar_chunk_calql_debug}"
SEED=0
export MPLCONFIGDIR=/tmp/matplotlib-${USER}
export XLA_PYTHON_CLIENT_PREALLOCATE=false

.venv/bin/python experiments/train_bridge_soar_chunk.py \
  --run_group="${RUN_GROUP}" \
  --seed="${SEED}" \
  --save_dir="${SAVE_DIR}" \
  --offline_steps=100 \
  --log_interval=10 \
  --save_interval=50 \
  --agent=agents/bridge_calql_chunk.py \
  --data=experiments/configs/bridge_soar_chunk_data_config.py \
  --agent.backup_horizon=8 \
  --agent.policy_chunk_size=1 \
  --agent.use_chunk_critic=True \
  --agent.discount=0.98 \
  --agent.cql_alpha=5.0 \
  --agent.batch_size=8 \
  --agent.critic_hidden_dims=256,256 \
  --agent.actor_hidden_dims=256,256 \
  --data.backup_horizon=8 \
  --data.action_chunk_size=1 \
  --data.discount=0.98 \
  --data.batch_size=8 \
  --data.action_clip_eps=1e-5 \
  --data.num_workers=0 \
  --data.shuffle_buffer_size=32 \
  --data.min_shuffle_buffer_size=8 \
  --tags="BridgeSoarChunk,CALQL,debug,h8,ac1"
