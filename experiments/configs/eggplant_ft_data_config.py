"""Env-driven eggplant DQC finetune data config (one file, 3 reward variants).

Set EGG_LEROBOT_DIR to the built LeRobot dataset:
  v1 success-ramp     -> eggplant_success_ramp_lerobot   (labels.json -> 0->1 ramp)
  v2 all + failure-0  -> eggplant_all_lerobot            (labels.json -> ramp/zeros)
  v3 success-distance -> eggplant_success_dist_lerobot   (parquet `reward` column)
The loader uses the parquet `reward` column if present (v3), else derives reward
from labels.json (v1/v2). Optional: EGG_INF_HORIZON (default 1), EGG_NUM_WORKERS (8).
"""
import os

from ml_collections import ConfigDict


def get_config():
    d = os.environ["EGG_LEROBOT_DIR"]
    return ConfigDict(
        dict(
            sources=(
                dict(
                    name="eggplant",
                    data_dir=d,
                    weight=1.0,
                    video_key="observation.images.image_0",
                    label_path=os.path.join(d, "labels.json"),
                ),
            ),
            # Match the base DQC training config (sd000s.../flags.json):
            # batch_size 256, infinite_horizon_backup true, num_workers 16,
            # discount 0.98, backup_horizon 8, action_clip_eps 1e-5.
            batch_size=int(os.environ.get("EGG_BATCH", "256")),
            backup_horizon=8,
            action_chunk_size=1,
            discount=0.98,
            infinite_horizon_backup=(os.environ.get("EGG_INF_HORIZON", "1") == "1"),
            shuffle=True,
            shuffle_buffer_size=50000,
            min_shuffle_buffer_size=4096,
            normalize_actions=True,
            action_clip_eps=1e-5,
            num_workers=int(os.environ.get("EGG_NUM_WORKERS", "16")),
        )
    )
