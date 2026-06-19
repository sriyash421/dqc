from ml_collections import ConfigDict


def get_config():
    return ConfigDict(
        dict(
            # Phase 2: full eggplant build (success + failure) produced by
            # scriptsv2/eggplant_to_lerobot/zarr_to_lerobot.py --filter all.
            # labels.json marks each episode: "successful" -> shaped 0->1 ramp reward,
            # "failure" -> all-zero reward (the failure signal for the critic).
            sources=(
                dict(
                    name="eggplant",
                    data_dir="/home/harine/RoboMonkey/data/eggplant_all_lerobot",
                    weight=1.0,
                    video_key="observation.images.image_0",
                    label_path="/home/harine/RoboMonkey/data/eggplant_all_lerobot/labels.json",
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
