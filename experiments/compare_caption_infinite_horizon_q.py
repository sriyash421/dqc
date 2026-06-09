import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'user')}")
os.environ.setdefault("XDG_CACHE_HOME", f"/tmp/xdg-cache-{os.environ.get('USER', 'user')}")

import imageio.v3 as iio
import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from ml_collections import ConfigDict

from agents.bridge_dqc_chunk import BridgeDQCChunkAgent
from utils.bridge_lerobot_dataset import BRIDGE_ACTION_LOW, _episode_to_batch, _rewards_from_label, make_bridge_soar_chunk_dataset
from utils.flax_utils import restore_agent
from utils.text_processing import make_text_processor


CAPTION = "move the silver pot to the left side of the table"
CHECKPOINT_DIR = Path(
    "/gpfs/projects/socialrl/sriyash/dqc_bridge_soar_dqc_infinite_horizon/"
    "dqc/bridge_soar_dqc_infinite_horizon/sd000s_137913.0.20260605_163017"
)
RESTORE_EPOCH = -1
OUTPUT_DIR = Path("exp/caption_compare_infinite_horizon")
OUTPUT_NAME = "move_silver_pot_left_bridge_success_vs_soar_failure.mp4"
SEED = 0
FPS = 10
Q_AGG = "min"
SCORE_BATCH_SIZE = 64


def _disable_tensorflow_gpus():
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")


def _latest_epoch(checkpoint_dir: Path) -> int:
    epochs = []
    for path in checkpoint_dir.glob("params_*.pkl"):
        match = re.fullmatch(r"params_(\d+)\.pkl", path.name)
        if match:
            epochs.append(int(match.group(1)))
    if not epochs:
        raise FileNotFoundError(f"No params_*.pkl files found in {checkpoint_dir}")
    return max(epochs)


def _load_flags(checkpoint_dir: Path) -> dict:
    with (checkpoint_dir / "flags.json").open("r") as f:
        return json.load(f)


def _make_agent(agent_config: ConfigDict, seed: int) -> BridgeDQCChunkAgent:
    action_dim = int(agent_config.get("action_dim") or BRIDGE_ACTION_LOW.shape[0])
    backup_horizon = int(agent_config["backup_horizon"])
    image_size = 256
    example_batch = {
        "observations": {"image": np.zeros((1, image_size, image_size, 3), dtype=np.uint8)},
        "next_observations": {"image": np.zeros((1, image_size, image_size, 3), dtype=np.uint8)},
        "high_value_next_observations": {"image": np.zeros((1, image_size, image_size, 3), dtype=np.uint8)},
        "goals": {"language": np.zeros((1, 512), dtype=np.float32)},
        "high_value_goals": {"language": np.zeros((1, 512), dtype=np.float32)},
        "actions": np.zeros((1, action_dim), dtype=np.float32),
        "high_value_action_chunks": np.zeros((1, backup_horizon * action_dim), dtype=np.float32),
        "rewards": np.zeros((1,), dtype=np.float32),
        "masks": np.ones((1,), dtype=np.float32),
        "high_value_rewards": np.zeros((1,), dtype=np.float32),
        "high_value_masks": np.ones((1,), dtype=np.float32),
        "high_value_backup_horizon": np.full((1,), backup_horizon, dtype=np.float32),
        "valids": np.ones((1, backup_horizon), dtype=np.float32),
        "mc_returns": np.zeros((1,), dtype=np.float32),
    }
    return BridgeDQCChunkAgent.create(seed, example_batch, agent_config)


def _source_by_name(dataset, name: str):
    for source in dataset.sources:
        if source.name == name:
            return source
    raise KeyError(f"Unknown source: {name}")


def _raw_rewards(dataset, source_name: str, episode: dict) -> np.ndarray:
    data = pq.read_table(episode["path"]).to_pydict()
    if "reward" in data:
        return np.asarray(data["reward"], dtype=np.float32)
    return np.asarray(_rewards_from_label(episode, dataset.labels[source_name], len(data["action"])), dtype=np.float32)


def _load_episode_batch(dataset, source_name: str, episode: dict) -> dict:
    source = _source_by_name(dataset, source_name)
    batch = _episode_to_batch(
        source,
        episode,
        tasks=dataset.tasks[source_name],
        labels=dataset.labels[source_name],
        backup_horizon=dataset.backup_horizon,
        discount=dataset.discount,
        normalize_actions=dataset.normalize_actions,
        action_clip_eps=getattr(dataset, "action_clip_eps", 1e-5),
        infinite_horizon_backup=getattr(dataset, "infinite_horizon_backup", False),
    )
    batch["og_rewards"] = _raw_rewards(dataset, source_name, episode)[: batch["actions"].shape[0]]
    return batch


def _find_caption_episode(dataset, source_name: str, caption: str, rng: np.random.Generator) -> dict:
    task_indices = {idx for idx, task in dataset.tasks[source_name].items() if task == caption}
    if not task_indices:
        raise ValueError(f"No task index for caption in {source_name}: {caption!r}")
    candidates = [episode for episode in dataset.episodes[source_name] if episode["task_index"] in task_indices]
    if not candidates:
        raise ValueError(f"No usable episodes for caption in {source_name}: {caption!r}")
    return candidates[int(rng.integers(0, len(candidates)))]


def _embed_language(batch: dict, text_processor):
    task_text = batch["goals"]["language"][0]
    task_text = task_text.decode("utf-8") if isinstance(task_text, bytes) else str(task_text)
    language = text_processor.encode([task_text]).astype(np.float32)
    language = np.repeat(language, batch["actions"].shape[0], axis=0)
    batch["goals"]["language"] = language
    batch["high_value_goals"]["language"] = language
    return batch


def _score_chunks(agent, batch: dict) -> np.ndarray:
    scores = []
    chunks = batch["high_value_action_chunks"]
    images = batch["observations"]["image"]
    language = batch["high_value_goals"]["language"]
    for start in range(0, chunks.shape[0], SCORE_BATCH_SIZE):
        end = min(start + SCORE_BATCH_SIZE, chunks.shape[0])
        obs = {"image": np.asarray(images[start:end])}
        goals = {"language": np.asarray(language[start:end])}
        q = agent.score_chunks(obs, goals, np.asarray(chunks[start:end], dtype=np.float32), q_agg=Q_AGG)
        scores.append(np.asarray(jax.device_get(q), dtype=np.float32))
    return np.concatenate(scores, axis=0)


def _axis_limits(*values) -> tuple[float, float]:
    joined = np.concatenate([np.asarray(value, dtype=np.float32).reshape(-1) for value in values])
    joined = joined[np.isfinite(joined)]
    if joined.size == 0:
        return -1.0, 1.0
    low = float(joined.min())
    high = float(joined.max())
    pad = 1.0 if np.isclose(low, high) else 0.1 * (high - low)
    return low - pad, high + pad


def _frame_at(images: np.ndarray, t: int) -> np.ndarray:
    return images[min(t, len(images) - 1)]


def _make_video(success: dict, failure: dict, output_path: Path):
    success_rewards = success["batch"]["og_rewards"]
    failure_rewards = failure["batch"]["og_rewards"]
    success_q = success["q_values"]
    failure_q = failure["q_values"]
    reward_ylim = _axis_limits(success_rewards, failure_rewards)
    q_ylim = _axis_limits(success_q, failure_q)
    max_len = max(len(success_rewards), len(failure_rewards))
    frames = []

    for t in range(max_len):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=110)
        axes[0].imshow(_frame_at(success["batch"]["observations"]["image"], t))
        axes[0].set_title(f"Bridge success\nepisode {success['episode']['episode_index']}", fontsize=9)
        axes[0].axis("off")
        axes[1].imshow(_frame_at(failure["batch"]["observations"]["image"], t))
        axes[1].set_title(f"SOAR failure\nepisode {failure['episode']['episode_index']}", fontsize=9)
        axes[1].axis("off")

        success_t = np.arange(len(success_rewards))
        failure_t = np.arange(len(failure_rewards))
        axes[2].plot(success_t, success_rewards, color="tab:green", marker="o", markevery=max(len(success_t) // 12, 1), label="Bridge success")
        axes[2].plot(failure_t, failure_rewards, color="tab:red", marker="x", markevery=max(len(failure_t) // 12, 1), label="SOAR failure")
        axes[2].axvline(t, color="black", linestyle="--", linewidth=1)
        axes[2].set_title("OG reward")
        axes[2].set_ylim(*reward_ylim)
        axes[2].grid(alpha=0.25)
        axes[2].legend(fontsize=8)

        axes[3].plot(success_t, success_q, color="tab:blue", marker="o", markevery=max(len(success_t) // 12, 1), label="Bridge success")
        axes[3].plot(failure_t, failure_q, color="tab:orange", marker="x", markevery=max(len(failure_t) // 12, 1), label="SOAR failure")
        axes[3].axvline(t, color="black", linestyle="--", linewidth=1)
        axes[3].set_title("Predicted Q")
        axes[3].set_ylim(*q_ylim)
        axes[3].grid(alpha=0.25)
        axes[3].legend(fontsize=8)

        fig.suptitle(CAPTION, fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy())
        plt.close(fig)
    iio.imwrite(output_path, np.stack(frames), fps=FPS)


def main():
    _disable_tensorflow_gpus()
    flags_dict = _load_flags(CHECKPOINT_DIR)
    agent_config = ConfigDict(flags_dict["agent"])
    data_config = ConfigDict(flags_dict["data"])
    data_config.batch_size = 1
    data_config.num_workers = 0
    data_config.shuffle = False
    restore_epoch = _latest_epoch(CHECKPOINT_DIR) if RESTORE_EPOCH < 0 else RESTORE_EPOCH

    agent = _make_agent(agent_config, SEED)
    agent = restore_agent(agent, str(CHECKPOINT_DIR), restore_epoch)
    text_processor = make_text_processor(agent_config.text_processor)
    dataset = make_bridge_soar_chunk_dataset(**data_config.to_dict())
    rng = np.random.default_rng(SEED)

    success_episode = _find_caption_episode(dataset, "bridge_v2", CAPTION, rng)
    failure_episode = _find_caption_episode(dataset, "soar_failure", CAPTION, rng)
    success_batch = _embed_language(_load_episode_batch(dataset, "bridge_v2", success_episode), text_processor)
    failure_batch = _embed_language(_load_episode_batch(dataset, "soar_failure", failure_episode), text_processor)
    success = {"episode": success_episode, "batch": success_batch, "q_values": _score_chunks(agent, success_batch)}
    failure = {"episode": failure_episode, "batch": failure_batch, "q_values": _score_chunks(agent, failure_batch)}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_NAME
    _make_video(success, failure, output_path)
    with (OUTPUT_DIR / "metadata.json").open("w") as f:
        json.dump(
            {
                "caption": CAPTION,
                "checkpoint_dir": str(CHECKPOINT_DIR),
                "restore_epoch": int(restore_epoch),
                "q_agg": Q_AGG,
                "bridge_episode_index": int(success_episode["episode_index"]),
                "soar_episode_index": int(failure_episode["episode_index"]),
                "output": output_path.name,
            },
            f,
            indent=2,
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
