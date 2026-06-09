import csv
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'user')}")
os.environ.setdefault("XDG_CACHE_HOME", f"/tmp/xdg-cache-{os.environ.get('USER', 'user')}")

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from ml_collections import ConfigDict

from agents.bridge_calql_chunk import BridgeCalQLChunkAgent
from agents.bridge_dqc_chunk import BridgeDQCChunkAgent
from utils.bridge_lerobot_dataset import (
    BRIDGE_ACTION_HIGH,
    BRIDGE_ACTION_LOW,
    _episode_to_batch,
    _rewards_from_label,
    make_bridge_soar_chunk_dataset,
)
from utils.flax_utils import restore_agent
from utils.text_processing import make_text_processor


# -----------------------
# Hardcoded hyperparameters
# -----------------------

MODELS = (
    {
        "name": "dqc_og_repo",
        "agent_cls": BridgeDQCChunkAgent,
        "checkpoint_dir": "/gpfs/projects/socialrl/sriyash/dqc_bridge_soar_dqc_infinite_horizon/dqc/bridge_soar_dqc_infinite_horizon/sd000s_137913.0.20260605_163017/",
        "restore_epoch": -1,
    },
    {
        "name": "calql_dqc_low_alpha",
        "agent_cls": BridgeCalQLChunkAgent,
        "checkpoint_dir": "/gpfs/projects/socialrl/sriyash/dqc_bridge_soar_chunk_calql/dqc/bridge_soar_chunk_calql/sd000s_133518.0.20260602_183103",
        "restore_epoch": -1,
    },
)

OUTPUT_DIR = Path("exp/model_q_correlation_analysis")
SEED = 0
Q_AGG = "min"
SCORE_BATCH_SIZE = 128

VALUE_TOTAL_TRANSITIONS = 1000

POLICY_NUM_BRIDGE_TRANSITIONS = 1000
POLICY_SAMPLES_PER_TRANSITION = 10
POLICY_SCORE_BATCH_SIZE = 11


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


def _make_example_batch(agent_config: ConfigDict) -> dict:
    action_dim = int(agent_config.get("action_dim") or BRIDGE_ACTION_LOW.shape[0])
    backup_horizon = int(agent_config["backup_horizon"])
    image_size = 256
    return {
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


def _load_model(model_spec: dict):
    checkpoint_dir = Path(model_spec["checkpoint_dir"])
    flags_dict = _load_flags(checkpoint_dir)
    agent_config = ConfigDict(flags_dict["agent"])
    restore_epoch = _latest_epoch(checkpoint_dir) if model_spec["restore_epoch"] < 0 else int(model_spec["restore_epoch"])
    agent = model_spec["agent_cls"].create(SEED, _make_example_batch(agent_config), agent_config)
    agent = restore_agent(agent, str(checkpoint_dir), restore_epoch)
    return {
        "name": model_spec["name"],
        "agent": agent,
        "agent_config": agent_config,
        "data_config": ConfigDict(flags_dict["data"]),
        "checkpoint_dir": str(checkpoint_dir),
        "restore_epoch": restore_epoch,
    }


def _source_by_name(dataset, name: str):
    for source in dataset.sources:
        if source.name == name:
            return source
    raise KeyError(f"Unknown source: {name}")


def _raw_rewards(dataset, source_name: str, episode: dict) -> np.ndarray:
    source = _source_by_name(dataset, source_name)
    data = pq.read_table(episode["path"]).to_pydict()
    if "reward" in data:
        rewards = np.asarray(data["reward"], dtype=np.float32)
    else:
        rewards = np.asarray(_rewards_from_label(episode, dataset.labels[source_name], len(data["action"])), dtype=np.float32)
    return rewards[: int(episode["length"]) - 1]


def _discounted_returns(rewards: np.ndarray, discount: float) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + discount * running
        returns[i] = running
    return returns


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
    rewards = _raw_rewards(dataset, source_name, episode)[: batch["actions"].shape[0]]
    batch["og_rewards"] = rewards
    batch["computed_q"] = _discounted_returns(rewards, dataset.discount)
    return batch


def _embed_episode_language(batch: dict, text_processor):
    first = batch["goals"]["language"][0]
    task_text = first.decode("utf-8") if isinstance(first, bytes) else str(first)
    language = text_processor.encode([task_text]).astype(np.float32)
    language = np.repeat(language, batch["actions"].shape[0], axis=0)
    batch["goals"]["language"] = language
    batch["high_value_goals"]["language"] = language
    return batch, task_text


def _score_chunks(agent, images, language, chunks, batch_size: int) -> np.ndarray:
    scores = []
    for start in range(0, chunks.shape[0], batch_size):
        end = min(start + batch_size, chunks.shape[0])
        obs = {"image": np.asarray(images[start:end])}
        goals = {"language": np.asarray(language[start:end])}
        q = agent.score_chunks(obs, goals, np.asarray(chunks[start:end], dtype=np.float32), q_agg=Q_AGG)
        scores.append(np.asarray(jax.device_get(q), dtype=np.float32))
    return np.concatenate(scores, axis=0)


def _corr(x, y) -> tuple[float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return float("nan"), int(x.size)
    return float(np.corrcoef(x, y)[0, 1]), int(x.size)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _spearman(x, y) -> tuple[float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    return _corr(_rankdata(x[finite]), _rankdata(y[finite]))


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_value_correlation_bars(rows: list[dict], output_dir: Path):
    labels = ["all", "bridge_v2", "soar_failure"]
    model_names = [model["name"] for model in MODELS]
    width = 0.36
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    for i, model_name in enumerate(model_names):
        vals = [
            next(row["pearson"] for row in rows if row["analysis"] == "value" and row["model"] == model_name and row["source"] == source)
            for source in labels
        ]
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=model_name)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Predicted Q vs computed Q value")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "predicted_vs_computed_q_correlation.png")
    plt.close(fig)


def _plot_policy_outputs(policy_rows: list[dict], summary_rows: list[dict], output_dir: Path):
    model_names = [model["name"] for model in MODELS]
    width = 0.5
    x = np.arange(len(model_names))
    vals = [
        next(row["pearson"] for row in summary_rows if row["analysis"] == "policy_distance_vs_q_diff" and row["model"] == model_name)
        for model_name in model_names
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    ax.bar(x, vals, width=width)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=12, ha="right")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Policy sample distance vs Q(dataset) - Q(policy)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "policy_distance_vs_q_diff_correlation.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(model_names), figsize=(12, 4.5), dpi=160, sharex=True)
    if len(model_names) == 1:
        axes = [axes]
    for ax, model_name in zip(axes, model_names):
        rows = [row for row in policy_rows if row["model"] == model_name]
        distance = np.asarray([row["distance_to_gt_action"] for row in rows], dtype=np.float32)
        q_diff = np.asarray([row["q_dataset_minus_policy"] for row in rows], dtype=np.float32)
        finite = np.isfinite(distance) & np.isfinite(q_diff)
        ax.scatter(distance[finite], q_diff[finite], s=5, alpha=0.2)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(model_name)
        ax.set_xlabel("L2 distance to GT action chunk")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Q(dataset action) - Q(policy action)")
    fig.tight_layout()
    fig.savefig(output_dir / "policy_distance_vs_q_diff_scatter.png")
    plt.close(fig)


def _sample_value_rows(dataset, text_processor, models, rng) -> tuple[list[dict], list[dict]]:
    value_rows = []
    summary_rows = []
    transitions_per_source = {
        "bridge_v2": VALUE_TOTAL_TRANSITIONS // 2,
        "soar_failure": VALUE_TOTAL_TRANSITIONS - VALUE_TOTAL_TRANSITIONS // 2,
    }
    for source_name in ("bridge_v2", "soar_failure"):
        episodes = dataset.episodes[source_name]
        sampled = 0
        local_episode_id = 0
        while sampled < transitions_per_source[source_name]:
            episode_idx = int(rng.integers(0, len(episodes)))
            episode = episodes[int(episode_idx)]
            batch = _load_episode_batch(dataset, source_name, episode)
            batch, task_text = _embed_episode_language(batch, text_processor)
            horizon = batch["actions"].shape[0]
            remaining = transitions_per_source[source_name] - sampled
            state_count = min(remaining, horizon)
            timesteps = rng.choice(horizon, size=state_count, replace=False)
            timesteps.sort()
            images = batch["observations"]["image"][timesteps]
            language = batch["high_value_goals"]["language"][timesteps]
            chunks = batch["high_value_action_chunks"][timesteps]
            computed_q = batch["computed_q"][timesteps]
            for model in models:
                predicted_q = _score_chunks(model["agent"], images, language, chunks, SCORE_BATCH_SIZE)
                for t, pred, computed in zip(timesteps, predicted_q, computed_q):
                    value_rows.append(
                        {
                            "model": model["name"],
                            "source": source_name,
                            "episode_index": int(episode["episode_index"]),
                            "local_episode_id": local_episode_id,
                            "timestep": int(t),
                            "task_name": task_text,
                            "predicted_q": float(pred),
                            "computed_q": float(computed),
                        }
                    )
            sampled += state_count
            local_episode_id += 1
            if sampled % 100 == 0 or sampled == transitions_per_source[source_name]:
                print(f"Scored value samples: {source_name} {sampled}/{transitions_per_source[source_name]} transitions")

    for model in models:
        for source_name in ("all", "bridge_v2", "soar_failure"):
            rows = [row for row in value_rows if row["model"] == model["name"] and (source_name == "all" or row["source"] == source_name)]
            predicted = [row["predicted_q"] for row in rows]
            computed = [row["computed_q"] for row in rows]
            pearson, n = _corr(predicted, computed)
            spearman, _ = _spearman(predicted, computed)
            summary_rows.append(
                {
                    "analysis": "value",
                    "model": model["name"],
                    "source": source_name,
                    "pearson": pearson,
                    "spearman": spearman,
                    "n": n,
                }
            )
    return value_rows, summary_rows


def _sample_policy_chunks(model: dict, images, language, expert_chunk, rng) -> np.ndarray:
    sample_count = POLICY_SAMPLES_PER_TRANSITION
    repeated_images = np.repeat(images, sample_count, axis=0)
    repeated_language = np.repeat(language, sample_count, axis=0)
    obs = {"image": repeated_images}
    goals = {"language": repeated_language}
    if hasattr(model["agent"], "sample_policy_actions"):
        sample_key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
        actions, _ = model["agent"].sample_policy_actions(obs, goals, sample_key, train=False)
        return np.asarray(jax.device_get(actions), dtype=np.float32)

    noises = rng.normal(
        0.0,
        1.0,
        size=(sample_count, int(model["agent_config"]["policy_chunk_size"]) * BRIDGE_ACTION_LOW.shape[0]),
    ).astype(np.float32)
    actions = model["agent"].compute_flow_actions(obs, goals, noises)
    actions = np.asarray(jax.device_get(actions), dtype=np.float32)
    sampled_chunks = np.repeat(expert_chunk[None], sample_count, axis=0)
    action_dim = BRIDGE_ACTION_LOW.shape[0]
    sampled_chunks[:, :action_dim] = actions[:, :action_dim]
    return sampled_chunks


def _sample_policy_rows(dataset, text_processor, models, rng) -> tuple[list[dict], list[dict]]:
    policy_rows = []
    episodes = dataset.episodes["bridge_v2"]
    for state_id in range(POLICY_NUM_BRIDGE_TRANSITIONS):
        episode = episodes[int(rng.integers(0, len(episodes)))]
        batch = _load_episode_batch(dataset, "bridge_v2", episode)
        batch, task_text = _embed_episode_language(batch, text_processor)
        t = int(rng.integers(0, batch["actions"].shape[0]))
        expert_chunk = batch["high_value_action_chunks"][t].astype(np.float32)
        for model in models:
            image = batch["observations"]["image"][t : t + 1]
            language = batch["high_value_goals"]["language"][t : t + 1]
            sampled_chunks = _sample_policy_chunks(model, image, language, expert_chunk, rng)
            chunks = np.concatenate([expert_chunk[None], sampled_chunks], axis=0)
            images = np.repeat(image, chunks.shape[0], axis=0)
            languages = np.repeat(language, chunks.shape[0], axis=0)
            predicted_q = _score_chunks(model["agent"], images, languages, chunks, POLICY_SCORE_BATCH_SIZE)
            dataset_q = float(predicted_q[0])
            distances = np.linalg.norm(sampled_chunks - expert_chunk[None], axis=-1)
            for sample_idx, (distance, policy_q) in enumerate(zip(distances, predicted_q[1:])):
                policy_rows.append(
                    {
                        "model": model["name"],
                        "state_id": state_id,
                        "episode_index": int(episode["episode_index"]),
                        "timestep": t,
                        "task_name": task_text,
                        "policy_sample": sample_idx,
                        "distance_to_gt_action": float(distance),
                        "dataset_predicted_q": dataset_q,
                        "policy_predicted_q": float(policy_q),
                        "q_dataset_minus_policy": float(dataset_q - policy_q),
                    }
                )
        if (state_id + 1) % 100 == 0:
            print(f"Scored policy Bridge transitions: {state_id + 1}/{POLICY_NUM_BRIDGE_TRANSITIONS}")

    summary_rows = []
    for model in models:
        rows = [row for row in policy_rows if row["model"] == model["name"]]
        distance = [row["distance_to_gt_action"] for row in rows]
        q_diff = [row["q_dataset_minus_policy"] for row in rows]
        pearson, n = _corr(distance, q_diff)
        spearman, _ = _spearman(distance, q_diff)
        summary_rows.append(
            {
                "analysis": "policy_distance_vs_q_diff",
                "model": model["name"],
                "source": "bridge_v2",
                "pearson": pearson,
                "spearman": spearman,
                "n": n,
            }
        )
    return policy_rows, summary_rows


def main():
    _disable_tensorflow_gpus()
    if Q_AGG not in ("min", "mean"):
        raise ValueError("Q_AGG must be 'min' or 'mean'")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    models = [_load_model(spec) for spec in MODELS]
    data_config = ConfigDict(models[0]["data_config"])
    data_config.batch_size = 1
    data_config.num_workers = 0
    data_config.shuffle = False

    text_processor = make_text_processor(models[0]["agent_config"].text_processor)
    dataset = make_bridge_soar_chunk_dataset(**data_config.to_dict())

    value_rows, value_summary = _sample_value_rows(dataset, text_processor, models, rng)
    policy_rows, policy_summary = _sample_policy_rows(dataset, text_processor, models, rng)
    summary_rows = value_summary + policy_summary

    _write_csv(OUTPUT_DIR / "predicted_vs_computed_q_samples.csv", value_rows)
    _write_csv(OUTPUT_DIR / "policy_distance_vs_q_diff_samples.csv", policy_rows)
    _write_csv(OUTPUT_DIR / "correlation_summary.csv", summary_rows)
    _plot_value_correlation_bars(summary_rows, OUTPUT_DIR)
    _plot_policy_outputs(policy_rows, summary_rows, OUTPUT_DIR)

    metadata = {
        "models": [
            {
                "name": model["name"],
                "checkpoint_dir": model["checkpoint_dir"],
                "restore_epoch": model["restore_epoch"],
            }
            for model in models
        ],
        "output_dir": str(OUTPUT_DIR),
        "seed": SEED,
        "q_agg": Q_AGG,
        "score_batch_size": SCORE_BATCH_SIZE,
        "value_total_transitions": VALUE_TOTAL_TRANSITIONS,
        "policy_num_bridge_transitions": POLICY_NUM_BRIDGE_TRANSITIONS,
        "policy_samples_per_transition": POLICY_SAMPLES_PER_TRANSITION,
        "dataset_discount": float(dataset.discount),
    }
    with (OUTPUT_DIR / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
