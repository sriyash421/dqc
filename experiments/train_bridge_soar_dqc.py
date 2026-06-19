import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents.bridge_dqc_chunk import BridgeDQCChunkAgent
from utils.bridge_lerobot_dataset import BRIDGE_ACTION_HIGH, BRIDGE_ACTION_LOW, make_bridge_soar_chunk_dataset
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb
from utils.text_processing import make_text_processor


FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "bridge_soar_dqc", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("save_dir", "/gpfs/projects/socialrl/sriyash/dqc_bridge_soar_dqc", "Save directory.")
flags.DEFINE_string("restore_path", None, "Restore path.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")
flags.DEFINE_integer("offline_steps", 1000000, "Number of offline training steps.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("save_interval", 100000, "Saving interval.")
flags.DEFINE_string("tags", "BridgeSoarDQC", "Comma-separated WandB tags.")
flags.DEFINE_bool("wandb_offline", False, "Use offline WandB mode.")

config_flags.DEFINE_config_file("agent", "agents/bridge_dqc_chunk.py", lock_config=False)
config_flags.DEFINE_config_file("data", "experiments/configs/bridge_soar_chunk_data_config.py", lock_config=False)


def _disable_tensorflow_gpus():
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")


def _decode_language(batch):
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in batch["goals"]["language"]]


def _process_batch(batch, text_processor, backup_horizon: int, action_dim: int):
    languages = _decode_language(batch)
    language_embeddings = text_processor.encode(languages)
    batch["goals"]["language"] = language_embeddings
    batch["high_value_goals"]["language"] = language_embeddings
    full_chunks = batch["high_value_action_chunks"].reshape(batch["high_value_action_chunks"].shape[0], backup_horizon, action_dim)
    batch["partial_action_chunks"] = full_chunks[:, :1]
    return jax.tree_util.tree_map(np.asarray, batch)


def _assert_finite(name, value):
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Found non-finite values in {name}")


def _check_batch(batch, agent_config, data_config):
    if "image" not in batch["observations"]:
        raise KeyError("Batch missing observations['image']")
    if "image" not in batch["high_value_next_observations"]:
        raise KeyError("Batch missing high_value_next_observations['image']")
    if "language" not in batch["high_value_goals"]:
        raise KeyError("Batch missing high_value_goals['language']")

    images = batch["observations"]["image"]
    next_images = batch["high_value_next_observations"]["image"]
    language = batch["high_value_goals"]["language"]
    actions = batch["actions"]
    rewards = batch["high_value_rewards"]
    chunks = batch["high_value_action_chunks"]
    partial_chunks = batch["partial_action_chunks"]

    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected observations['image'] shape [B,H,W,3], got {images.shape}")
    if next_images.shape != images.shape:
        raise ValueError(f"Expected next image shape {images.shape}, got {next_images.shape}")
    action_dim = actions.shape[-1]
    expected_full_shape = (actions.shape[0], int(agent_config["backup_horizon"]), action_dim)
    expected_partial_shape = (actions.shape[0], int(agent_config["policy_chunk_size"]), action_dim)
    full_chunks = chunks.reshape(actions.shape[0], int(agent_config["backup_horizon"]), action_dim)
    if full_chunks.shape != expected_full_shape:
        raise ValueError(f"Expected full chunk shape {expected_full_shape}, got {full_chunks.shape}")
    if partial_chunks.shape != expected_partial_shape:
        raise ValueError(f"Expected partial chunk shape {expected_partial_shape}, got {partial_chunks.shape}")

    _assert_finite("images", images.astype(np.float32))
    _assert_finite("next_images", next_images.astype(np.float32))
    _assert_finite("language embeddings", language)
    _assert_finite("actions", actions)
    _assert_finite("high_value_action_chunks", chunks)
    _assert_finite("partial_action_chunks", partial_chunks)
    _assert_finite("high_value_rewards", rewards)
    _assert_finite("high_value_masks", batch["high_value_masks"])
    _assert_finite("high_value_backup_horizon", batch["high_value_backup_horizon"])

    eps = float(data_config.get("action_clip_eps", 1e-5))
    lower = -1.0 + eps
    upper = 1.0 - eps
    for name, value in [("actions", actions), ("high_value_action_chunks", chunks), ("partial_action_chunks", partial_chunks)]:
        if np.any(value < lower) or np.any(value > upper):
            raise ValueError(f"{name} outside [{lower}, {upper}]: min={value.min()}, max={value.max()}")


def _prepare_configs(agent_config, data_config):
    if "target_update_rate" in agent_config:
        agent_config.tau = agent_config.target_update_rate
    if "num_critics" in agent_config:
        agent_config.num_qs = agent_config.num_critics
    agent_config.action_clip_eps = data_config.get("action_clip_eps", 1e-5)
    if data_config.backup_horizon != agent_config.backup_horizon:
        raise ValueError(f"data.backup_horizon={data_config.backup_horizon} must match agent.backup_horizon={agent_config.backup_horizon}")
    if data_config.action_chunk_size != agent_config.policy_chunk_size:
        raise ValueError(
            f"data.action_chunk_size={data_config.action_chunk_size} must match agent.policy_chunk_size={agent_config.policy_chunk_size}"
        )
    if data_config.batch_size != agent_config.batch_size:
        raise ValueError(f"data.batch_size={data_config.batch_size} must match agent.batch_size={agent_config.batch_size}")
    if agent_config.policy_chunk_size != 1:
        raise ValueError("Bridge/SOAR DQC requires policy_chunk_size=1")
    if "cql_alpha" in agent_config or "cql_temp" in agent_config or "cql_max_target_backup" in agent_config:
        raise ValueError("DQC config must not contain CQL/CAL-QL hyperparameters")


def main(_):
    _disable_tensorflow_gpus()
    agent_config = FLAGS.agent
    data_config = FLAGS.data
    _prepare_configs(agent_config, data_config)

    run = setup_wandb(
        entity=os.environ.get("WANDB_ENTITY") or None,
        project=os.environ.get("WANDB_PROJECT", "dqc"),
        group=FLAGS.run_group,
        name=os.environ.get("WANDB_NAME") or get_exp_name(FLAGS.seed),
        tags=FLAGS.tags.split(","),
        mode="offline" if FLAGS.wandb_offline else "online",
    )
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, run.name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(get_flag_dict(), f)
    with open(os.path.join(FLAGS.save_dir, "action_normalization.json"), "w") as f:
        json.dump(
            {
                "low": BRIDGE_ACTION_LOW.tolist(),
                "high": BRIDGE_ACTION_HIGH.tolist(),
                "clip_eps": float(data_config.get("action_clip_eps", 1e-5)),
            },
            f,
        )

    text_processor = make_text_processor(agent_config.text_processor)
    dataset = make_bridge_soar_chunk_dataset(**data_config.to_dict())
    train_iter = dataset.iterator(prefetch=0)
    action_dim = BRIDGE_ACTION_LOW.shape[0]
    example_batch = _process_batch(next(train_iter), text_processor, agent_config.backup_horizon, action_dim)
    _check_batch(example_batch, agent_config, data_config)

    agent = BridgeDQCChunkAgent.create(FLAGS.seed, example_batch, agent_config)
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(FLAGS.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):
        data_start = time.time()
        batch = _process_batch(next(train_iter), text_processor, agent_config.backup_horizon, action_dim)
        _check_batch(batch, agent_config, data_config)
        data_time = time.time() - data_start

        train_start = time.time()
        agent, update_info = agent.update(batch)
        update_info = jax.device_get(update_info)
        train_time = time.time() - train_start
        for key, value in update_info.items():
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Non-finite training metric {key}: {value}")

        if i % FLAGS.log_interval == 0:
            metrics = {f"training/{k}": v for k, v in update_info.items()}
            metrics["time/data_time"] = data_time
            metrics["time/train_time"] = train_time
            metrics["time/epoch_time"] = (time.time() - last_time) / max(FLAGS.log_interval, 1)
            metrics["time/total_time"] = time.time() - first_time
            last_time = time.time()
            wandb.log(metrics, step=i)
            train_logger.log(metrics, step=i)

        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    if getattr(run, "url", None):
        with open(os.path.join(FLAGS.save_dir, "token.tk"), "w") as f:
            f.write(run.url)


if __name__ == "__main__":
    app.run(main)
