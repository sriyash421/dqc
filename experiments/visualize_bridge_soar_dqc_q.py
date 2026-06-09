import csv
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
from absl import app, flags
from ml_collections import ConfigDict

from agents.bridge_dqc_chunk import BridgeDQCChunkAgent
from utils.bridge_lerobot_dataset import (
    BRIDGE_ACTION_HIGH,
    BRIDGE_ACTION_LOW,
    make_bridge_soar_chunk_dataset,
    _episode_to_batch,
    _rewards_from_label,
)
from utils.flax_utils import restore_agent
from utils.text_processing import make_text_processor


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_dir",
    "/gpfs/projects/socialrl/sriyash/dqc_bridge_soar_dqc/dqc/bridge_soar_dqc",
    "Directory containing flags.json and params_*.pkl.",
)
flags.DEFINE_integer("restore_epoch", -1, "Checkpoint epoch. Use -1 for latest params_*.pkl.")
flags.DEFINE_string("output_dir", "exp/bridge_soar_dqc_q_viz", "Directory for videos and plots.")
flags.DEFINE_integer("k_episodes", 4, "Number of dataset episodes to visualize.")
flags.DEFINE_integer("k_states", 64, "Number of Bridge-only states for noisy chunk scatter.")
flags.DEFINE_integer("num_episodes", None, "Alias for --k_episodes.")
flags.DEFINE_integer("num_scatter_states", None, "Alias for --k_states.")
flags.DEFINE_integer("noise_samples", 32, "Number of noisy chunks per sampled state.")
flags.DEFINE_float("noise_std", 0.25, "Gaussian noise std for noisy chunks in normalized action space.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("q_agg", "min", "Q ensemble aggregation: min or mean.")
flags.DEFINE_integer("fps", 10, "Output video FPS.")
flags.DEFINE_integer("score_batch_size", 64, "Batch size for Q scoring.")
flags.DEFINE_bool("strict_finite_q", False, "Raise if any scored Q value is NaN or Inf.")


def _disable_tensorflow_gpus():
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")


def _resolve_checkpoint_dir(path: str) -> Path:
    ckpt_dir = Path(path)
    if ckpt_dir.exists():
        return ckpt_dir
    stripped = Path(str(ckpt_dir).rstrip("."))
    if stripped.exists():
        return stripped
    raise FileNotFoundError(f"Checkpoint directory not found: {path}")


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
    flags_path = checkpoint_dir / "flags.json"
    if not flags_path.exists():
        raise FileNotFoundError(f"Missing checkpoint flags: {flags_path}")
    with flags_path.open("r") as f:
        return json.load(f)


def _validate_action_normalization(checkpoint_dir: Path):
    path = checkpoint_dir / "action_normalization.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing action normalization metadata: {path}")
    with path.open("r") as f:
        metadata = json.load(f)
    low = np.asarray(metadata["low"], dtype=np.float32)
    high = np.asarray(metadata["high"], dtype=np.float32)
    if not np.allclose(low, BRIDGE_ACTION_LOW) or not np.allclose(high, BRIDGE_ACTION_HIGH):
        raise ValueError(
            "Checkpoint action normalization does not match the Bridge/SOAR loader constants: "
            f"checkpoint low={low}, high={high}; loader low={BRIDGE_ACTION_LOW}, high={BRIDGE_ACTION_HIGH}"
        )
    if "clip_eps" not in metadata:
        print("WARNING: checkpoint action normalization metadata has no clip_eps; older run may contain exact tanh endpoints")


def _make_agent_from_checkpoint_config(agent_config: ConfigDict, seed: int) -> BridgeDQCChunkAgent:
    action_dim = int(agent_config.get("action_dim") or 7)
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


def _sanitize_task_name(task: str, max_len: int = 80) -> str:
    name = task.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "blank_task"
    return name[:max_len].strip("_")


def _source_by_name(dataset, name: str):
    for source in dataset.sources:
        if source.name == name:
            return source
    raise KeyError(f"Unknown dataset source: {name}")


def _episode_task_text(dataset, source_name: str, episode: dict) -> str:
    task_index = episode["task_index"]
    return dataset.tasks[source_name][task_index]


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
    )
    data = pq.read_table(episode["path"]).to_pydict()
    if "reward" in data:
        raw_rewards = np.asarray(data["reward"], dtype=np.float32)
    else:
        raw_rewards = np.asarray(_rewards_from_label(episode, dataset.labels[source_name], len(data["action"])), dtype=np.float32)
    batch["og_rewards"] = raw_rewards[: batch["actions"].shape[0]]
    return batch


def _validate_episode_batch(batch: dict, backup_horizon: int):
    images = batch["observations"]["image"]
    actions = batch["actions"]
    chunks = batch["high_value_action_chunks"]
    if images.dtype != np.uint8:
        raise ValueError(f"Expected uint8 images, got {images.dtype}")
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected image batch shape T x H x W x 3, got {images.shape}")
    if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(chunks)):
        raise ValueError("Found non-finite normalized actions or action chunks")
    if np.any(actions < -1.0001) or np.any(actions > 1.0001):
        raise ValueError(f"Actions outside [-1, 1]: min={actions.min()}, max={actions.max()}")
    if np.any(chunks < -1.0001) or np.any(chunks > 1.0001):
        raise ValueError(f"Action chunks outside [-1, 1]: min={chunks.min()}, max={chunks.max()}")
    action_dim = actions.shape[-1]
    expected_chunk_dim = backup_horizon * action_dim
    if chunks.shape[-1] != expected_chunk_dim:
        raise ValueError(f"Expected chunk dim {expected_chunk_dim}, got {chunks.shape[-1]}")


def _embed_episode_language(batch: dict, text_processor) -> tuple[dict, str]:
    first = batch["goals"]["language"][0]
    task_text = first.decode("utf-8") if isinstance(first, bytes) else str(first)
    language = text_processor.encode([task_text]).astype(np.float32)
    if not np.all(np.isfinite(language)):
        raise ValueError(f"MUSE produced non-finite embedding for task: {task_text}")
    tiled = np.repeat(language, batch["actions"].shape[0], axis=0)
    batch["goals"]["language"] = tiled
    batch["high_value_goals"]["language"] = tiled
    return batch, task_text


def _score_chunks(agent, images, language, action_chunks, batch_size: int, q_agg: str) -> np.ndarray:
    scores = []
    for start in range(0, action_chunks.shape[0], batch_size):
        end = min(start + batch_size, action_chunks.shape[0])
        obs = {"image": np.asarray(images[start:end])}
        goals = {"language": np.asarray(language[start:end])}
        chunks = np.asarray(action_chunks[start:end], dtype=np.float32)
        q = agent.score_chunks(obs, goals["language"], chunks, q_agg=q_agg)
        scores.append(np.asarray(jax.device_get(q), dtype=np.float32))
    return np.concatenate(scores, axis=0)


def _q_finite_metadata(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    metadata = {
        "q_count": int(values.size),
        "q_finite_count": int(finite.sum()),
        "q_nonfinite_count": int((~finite).sum()),
    }
    if finite.any():
        metadata.update(
            {
                "q_finite_min": float(values[finite].min()),
                "q_finite_mean": float(values[finite].mean()),
                "q_finite_max": float(values[finite].max()),
            }
        )
    else:
        metadata.update({"q_finite_min": None, "q_finite_mean": None, "q_finite_max": None})
    return metadata


def _check_q_values(values: np.ndarray, context: str):
    metadata = _q_finite_metadata(values)
    if metadata["q_nonfinite_count"] > 0:
        message = (
            f"{context}: {metadata['q_nonfinite_count']} / {metadata['q_count']} "
            "Q values are NaN or Inf"
        )
        if FLAGS.strict_finite_q:
            raise ValueError(message)
        print(f"WARNING: {message}")
    return metadata


def _plot_to_image(fig) -> np.ndarray:
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return image[..., :3].copy()


def _discounted_returns_from_rewards(rewards: np.ndarray, discount: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32)
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + discount * running
        returns[i] = running
    return returns


def _make_episode_video(images, rewards, q_values, computed_q_values, task_name: str, output_path: Path, fps: int):
    t_axis = np.arange(len(rewards))
    reward_y_min, reward_y_max = _axis_limits(rewards)
    q_y_min, q_y_max = _axis_limits(np.concatenate([q_values, computed_q_values]))
    frames = []
    for t in range(len(images)):
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), dpi=100)
        axes[0].imshow(images[t])
        axes[0].set_title(task_name, fontsize=9)
        axes[0].axis("off")

        axes[1].plot(t_axis, rewards, color="tab:green", linewidth=2)
        axes[1].axvline(t, color="black", linestyle="--", linewidth=1)
        axes[1].set_title("OG Reward")
        axes[1].set_xlim(0, max(len(rewards) - 1, 1))
        axes[1].set_ylim(reward_y_min, reward_y_max)
        axes[1].grid(alpha=0.25)

        finite_q = np.isfinite(q_values)
        if finite_q.any():
            axes[2].plot(t_axis[finite_q], q_values[finite_q], color="tab:blue", linewidth=2, label="model Q value")
        else:
            axes[2].text(0.5, 0.5, "All Q values are NaN/Inf", ha="center", va="center", transform=axes[2].transAxes)
        finite_computed_q = np.isfinite(computed_q_values)
        if finite_computed_q.any():
            axes[2].plot(
                t_axis[finite_computed_q],
                computed_q_values[finite_computed_q],
                color="tab:orange",
                linestyle=":",
                linewidth=2,
                label="computed Q value",
            )
        axes[2].axvline(t, color="black", linestyle="--", linewidth=1)
        axes[2].set_title("Expert Chunk Q")
        axes[2].set_xlim(0, max(len(q_values) - 1, 1))
        axes[2].set_ylim(q_y_min, q_y_max)
        axes[2].legend(loc="best", fontsize=8)
        axes[2].grid(alpha=0.25)

        fig.tight_layout()
        frames.append(_plot_to_image(fig))
        plt.close(fig)
    iio.imwrite(output_path, np.stack(frames), fps=fps)


def _axis_limits(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if np.isclose(v_min, v_max):
        pad = 1.0 if np.isclose(v_min, 0.0) else abs(v_min) * 0.1
        return v_min - pad, v_max + pad
    pad = (v_max - v_min) * 0.1
    return v_min - pad, v_max + pad


def _sample_episode_refs(dataset, rng: np.random.Generator, k_episodes: int) -> list[tuple[str, dict]]:
    bridge_count = (k_episodes + 1) // 2
    soar_count = k_episodes - bridge_count
    refs = []
    for source_name, count in [("bridge_v2", bridge_count), ("soar_failure", soar_count)]:
        episodes = dataset.episodes[source_name]
        indices = rng.choice(len(episodes), size=count, replace=count > len(episodes))
        refs.extend((source_name, episodes[int(idx)]) for idx in indices)
    rng.shuffle(refs)
    return refs


def _write_episode_videos(agent, dataset, text_processor, rng, output_dir: Path, k_episodes: int, fps: int, q_agg: str, score_batch_size: int):
    metadata = []
    for video_idx, (source_name, episode) in enumerate(_sample_episode_refs(dataset, rng, k_episodes)):
        batch = _load_episode_batch(dataset, source_name, episode)
        _validate_episode_batch(batch, dataset.backup_horizon)
        batch, task_text = _embed_episode_language(batch, text_processor)
        q_values = _score_chunks(
            agent,
            batch["observations"]["image"],
            batch["high_value_goals"]["language"],
            batch["high_value_action_chunks"],
            score_batch_size,
            q_agg,
        )
        computed_q_values = _discounted_returns_from_rewards(batch["og_rewards"], dataset.discount)
        q_metadata = _check_q_values(q_values, f"episode {episode['episode_index']} ({source_name}, {task_text})")
        filename = f"episode_{video_idx:03d}_{source_name}_{_sanitize_task_name(task_text)}.mp4"
        output_path = output_dir / filename
        _make_episode_video(
            batch["observations"]["image"],
            batch["og_rewards"],
            q_values,
            computed_q_values,
            task_text,
            output_path,
            fps,
        )
        metadata.append(
            {
                "filename": filename,
                "source": source_name,
                "episode_index": int(episode["episode_index"]),
                "task_name": task_text,
                "num_frames": int(batch["actions"].shape[0]),
                "computed_q_min": float(np.min(computed_q_values)),
                "computed_q_mean": float(np.mean(computed_q_values)),
                "computed_q_max": float(np.max(computed_q_values)),
                **q_metadata,
            }
        )
        print(f"Wrote {output_path}")
    return metadata


def _sample_bridge_states(dataset, rng: np.random.Generator, k_states: int):
    episodes = dataset.episodes["bridge_v2"]
    refs = []
    for _ in range(k_states):
        episode = episodes[int(rng.integers(0, len(episodes)))]
        horizon = int(episode["length"]) - 1
        if horizon <= 0:
            raise ValueError(f"Invalid Bridge episode length: {episode}")
        t = int(rng.integers(0, horizon))
        refs.append((episode, t))
    return refs


def _write_noise_scatter(agent, dataset, text_processor, rng, output_dir: Path, k_states: int, noise_samples: int, noise_std: float, q_agg: str):
    if k_states <= 0:
        return {"csv": None, "png": None, "num_rows": 0, "finite_rows": 0, "nonfinite_rows": 0}

    rows = []
    for state_id, (episode, t) in enumerate(_sample_bridge_states(dataset, rng, k_states)):
        batch = _load_episode_batch(dataset, "bridge_v2", episode)
        _validate_episode_batch(batch, dataset.backup_horizon)
        batch, task_text = _embed_episode_language(batch, text_processor)
        expert = batch["high_value_action_chunks"][t].astype(np.float32)
        image = batch["observations"]["image"][t : t + 1]
        language = batch["high_value_goals"]["language"][t : t + 1]

        noisy = np.clip(
            expert[None] + rng.normal(0.0, noise_std, size=(noise_samples, expert.shape[0])).astype(np.float32),
            -1.0,
            1.0,
        )
        expert_q = float(_score_chunks(agent, image, language, expert[None], 1, q_agg)[0])
        repeated_images = np.repeat(image, noise_samples, axis=0)
        repeated_language = np.repeat(language, noise_samples, axis=0)
        noisy_qs = _score_chunks(agent, repeated_images, repeated_language, noisy, noise_samples, q_agg)
        _check_q_values(np.concatenate([[expert_q], noisy_qs]), f"bridge noise state {state_id}, episode {episode['episode_index']}")
        distances = np.linalg.norm(noisy - expert[None], axis=-1)
        for sample_idx in range(noise_samples):
            rows.append(
                {
                    "state_id": state_id,
                    "episode_index": int(episode["episode_index"]),
                    "timestep": t,
                    "task_name": task_text,
                    "noise_sample": sample_idx,
                    "distance_to_expert": float(distances[sample_idx]),
                    "q_expert": expert_q,
                    "q_noisy": float(noisy_qs[sample_idx]),
                    "q_diff": float(expert_q - noisy_qs[sample_idx]),
                }
            )

    csv_path = output_dir / "bridge_noise_q_scatter.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    distances = np.asarray([row["distance_to_expert"] for row in rows], dtype=np.float32)
    q_diffs = np.asarray([row["q_diff"] for row in rows], dtype=np.float32)
    finite = np.isfinite(distances) & np.isfinite(q_diffs)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    if finite.any():
        ax.scatter(distances[finite], q_diffs[finite], s=14, alpha=0.65)
    else:
        ax.text(0.5, 0.5, "All Q differences are NaN/Inf", ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Distance to expert action chunk")
    ax.set_ylabel("Q(expert chunk) - Q(noisy chunk)")
    ax.set_title("Bridge DQC noisy chunks vs expert chunks")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    png_path = output_dir / "bridge_noise_q_scatter.png"
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    return {
        "csv": csv_path.name,
        "png": png_path.name,
        "num_rows": len(rows),
        "finite_rows": int(finite.sum()),
        "nonfinite_rows": int((~finite).sum()),
    }


def main(_):
    _disable_tensorflow_gpus()
    if FLAGS.q_agg not in ("min", "mean"):
        raise ValueError("--q_agg must be min or mean")

    checkpoint_dir = _resolve_checkpoint_dir(FLAGS.checkpoint_dir)
    restore_epoch = _latest_epoch(checkpoint_dir) if FLAGS.restore_epoch < 0 else FLAGS.restore_epoch
    _validate_action_normalization(checkpoint_dir)
    flags_dict = _load_flags(checkpoint_dir)
    agent_config = ConfigDict(flags_dict["agent"])
    data_config = ConfigDict(flags_dict["data"])
    data_config.batch_size = 1
    data_config.num_workers = 0
    data_config.shuffle = False

    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_processor = make_text_processor(agent_config.text_processor)
    dataset = make_bridge_soar_chunk_dataset(**data_config.to_dict())
    agent = _make_agent_from_checkpoint_config(agent_config, FLAGS.seed)
    agent = restore_agent(agent, str(checkpoint_dir), restore_epoch)

    rng = np.random.default_rng(FLAGS.seed)
    episode_metadata = _write_episode_videos(
        agent,
        dataset,
        text_processor,
        rng,
        output_dir,
        FLAGS.num_episodes if FLAGS.num_episodes is not None else FLAGS.k_episodes,
        FLAGS.fps,
        FLAGS.q_agg,
        FLAGS.score_batch_size,
    )
    scatter_metadata = _write_noise_scatter(
        agent,
        dataset,
        text_processor,
        rng,
        output_dir,
        FLAGS.num_scatter_states if FLAGS.num_scatter_states is not None else FLAGS.k_states,
        FLAGS.noise_samples,
        FLAGS.noise_std,
        FLAGS.q_agg,
    )

    metadata = {
        "checkpoint_dir": str(checkpoint_dir),
        "restore_epoch": int(restore_epoch),
        "q_agg": FLAGS.q_agg,
        "k_episodes": int(FLAGS.num_episodes if FLAGS.num_episodes is not None else FLAGS.k_episodes),
        "k_states": int(FLAGS.num_scatter_states if FLAGS.num_scatter_states is not None else FLAGS.k_states),
        "noise_samples": FLAGS.noise_samples,
        "noise_std": FLAGS.noise_std,
        "episodes": episode_metadata,
        "scatter": scatter_metadata,
    }
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    app.run(main)
