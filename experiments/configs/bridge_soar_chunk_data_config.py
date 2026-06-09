from ml_collections import ConfigDict


def get_config():
    return ConfigDict(
        dict(
            sources=(
                dict(
                    name="bridge_v2",
                    data_dir="/gpfs/projects/socialrl/sriyash/bridgev2_datasets/bridge_v2_lerobot",
                    weight=1.0,
                    video_key="observation.images.image_0",
                ),
                dict(
                    name="soar_failure",
                    data_dir="/gpfs/projects/socialrl/sriyash/bridgev2_datasets/soar_failure_lerobot",
                    weight=1.0,
                    video_key="observation.images.image_0",
                    label_path="../V-GPS/soar_label_corrections_full.json",
                ),
            ),
            batch_size=1024,
            backup_horizon=8,
            action_chunk_size=1,
            discount=0.98,
            infinite_horizon_backup=False,
            shuffle=True,
            shuffle_buffer_size=50000,
            min_shuffle_buffer_size=4096,
            normalize_actions=True,
            action_clip_eps=1e-5,
            num_workers=32,
        )
    )
