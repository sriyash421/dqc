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

from agents.bridge_calql_chunk import BridgeCalQLChunkAgent
from utils.bridge_lerobot_dataset import BRIDGE_ACTION_HIGH, BRIDGE_ACTION_LOW, make_bridge_soar_chunk_dataset
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb
from utils.text_processing import make_text_processor


FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "BridgeSoarChunkDebug", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("save_dir", "exp/", "Save directory.")
flags.DEFINE_string("restore_path", None, "Restore path.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")
flags.DEFINE_integer("offline_steps", 1000000, "Number of offline training steps.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("save_interval", 100000, "Saving interval.")
flags.DEFINE_string("tags", "BridgeSoarChunk", "Comma-separated WandB tags.")
flags.DEFINE_bool("wandb_offline", False, "Use offline WandB mode.")

config_flags.DEFINE_config_file("agent", "agents/bridge_calql_chunk.py", lock_config=False)
config_flags.DEFINE_config_file("data", "experiments/configs/bridge_soar_chunk_data_config.py", lock_config=False)


def _disable_tensorflow_gpus():
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")


def _decode_language(batch):
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in batch["goals"]["language"]]


def _process_batch(batch, text_processor):
    languages = _decode_language(batch)
    language_embeddings = text_processor.encode(languages)
    batch["goals"]["language"] = language_embeddings
    batch["high_value_goals"]["language"] = language_embeddings
    return jax.tree_util.tree_map(np.asarray, batch)


def _check_batch(batch, config):
    action_dim = batch["actions"].shape[-1]
    expected_chunk_dim = config["backup_horizon"] * action_dim
    if batch["high_value_action_chunks"].shape[-1] != expected_chunk_dim:
        raise ValueError(
            f"Expected high_value_action_chunks dim {expected_chunk_dim}, "
            f"got {batch['high_value_action_chunks'].shape[-1]}"
        )
    if config["policy_chunk_size"] != 1:
        raise ValueError("This training path requires policy_chunk_size=1")
    if not config["use_chunk_critic"]:
        raise ValueError("This training path requires use_chunk_critic=True")
    eps = float(config.get("action_clip_eps", 1e-5))
    if np.any(batch["actions"] < -1.0 + eps / 2) or np.any(batch["actions"] > 1.0 - eps / 2):
        raise ValueError("Actions are outside the open tanh-safe action range")
    if not np.all(np.isfinite(batch["high_value_action_chunks"])):
        raise ValueError("Found non-finite action chunks")
    if np.any(batch["high_value_action_chunks"] < -1.0 + eps / 2) or np.any(batch["high_value_action_chunks"] > 1.0 - eps / 2):
        raise ValueError("Action chunks are outside the open tanh-safe action range")


def main(_):
    _disable_tensorflow_gpus()

    config = FLAGS.agent
    data_config = FLAGS.data
    config.action_clip_eps = data_config.get("action_clip_eps", 1e-5)
    if data_config.backup_horizon != config.backup_horizon:
        raise ValueError(f"data.backup_horizon={data_config.backup_horizon} must match agent.backup_horizon={config.backup_horizon}")
    if data_config.action_chunk_size != config.policy_chunk_size:
        raise ValueError(
            f"data.action_chunk_size={data_config.action_chunk_size} must match agent.policy_chunk_size={config.policy_chunk_size}"
        )
    if data_config.batch_size != config.batch_size:
        raise ValueError(f"data.batch_size={data_config.batch_size} must match agent.batch_size={config.batch_size}")

    run = setup_wandb(
        project="dqc",
        group=FLAGS.run_group,
        name=get_exp_name(FLAGS.seed),
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

    text_processor = make_text_processor(config.text_processor)
    dataset = make_bridge_soar_chunk_dataset(**data_config.to_dict())
    train_iter = dataset.iterator(prefetch=0)
    example_batch = _process_batch(next(train_iter), text_processor)
    _check_batch(example_batch, config)

    agent = BridgeCalQLChunkAgent.create(FLAGS.seed, example_batch, config)
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(FLAGS.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):
        data_start = time.time()
        batch = _process_batch(next(train_iter), text_processor)
        _check_batch(batch, config)
        data_time = time.time() - data_start
        train_start = time.time()
        agent, update_info = agent.update(batch)
        train_time = time.time() - train_start

        if i % FLAGS.log_interval == 0:
            metrics = {f"training/{k}": v for k, v in jax.device_get(update_info).items()}
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
    with open(os.path.join(FLAGS.save_dir, "token.tk"), "w") as f:
        f.write(run.url)


if __name__ == "__main__":
    app.run(main)
