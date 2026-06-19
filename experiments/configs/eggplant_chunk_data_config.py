from ml_collections import ConfigDict


def get_config():
    return ConfigDict(
        dict(
            # Phase 1: success-only eggplant build produced by
            # scriptsv2/eggplant_to_lerobot/zarr_to_lerobot.py --filter success.
            # All episodes are labeled "successful" -> shaped 0->1 ramp reward.
            # Phase 2 repoints these to the --filter all build to add failures.
            sources=(
                dict(
                    name="eggplant",
                    data_dir="/home/harine/RoboMonkey/data/eggplant_success_lerobot",
                    weight=1.0,
                    video_key="observation.images.image_0",
                    label_path="/home/harine/RoboMonkey/data/eggplant_success_lerobot/labels.json",
                ),
            ),
            batch_size=64,
            backup_horizon=8,
            action_chunk_size=1,
            discount=0.98,
            infinite_horizon_backup=True,
            shuffle=True,
            shuffle_buffer_size=50000,
            min_shuffle_buffer_size=4096,
            normalize_actions=True,
            action_clip_eps=1e-5,
            num_workers=8,
        )
    )
