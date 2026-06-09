"""Streaming BridgeV2 + SOAR LeRobot dataset for chunk-critic CAL-QL."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


BRIDGE_ACTION_LOW = np.array([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.0], dtype=np.float32)
BRIDGE_ACTION_HIGH = np.array([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class LeRobotSource:
    name: str
    data_dir: Path
    weight: float = 1.0
    video_key: str = "observation.images.image_0"
    label_path: Path | None = None


class BridgeSoarChunkDataset:
    def __init__(
        self,
        sources: list[LeRobotSource],
        batch_size: int,
        *,
        backup_horizon: int = 8,
        action_chunk_size: int = 1,
        discount: float = 0.98,
        infinite_horizon_backup: bool = False,
        shuffle: bool = True,
        shuffle_buffer_size: int = 50_000,
        min_shuffle_buffer_size: int = 4_096,
        normalize_actions: bool = True,
        action_clip_eps: float = 1e-5,
        num_workers: int = 32,
    ):
        if backup_horizon < 1:
            raise ValueError("backup_horizon must be >= 1")
        if action_chunk_size != 1:
            raise ValueError("This Bridge/SOAR path is configured for action_chunk_size=1")

        self.sources = sources
        self.batch_size = batch_size
        self.backup_horizon = backup_horizon
        self.action_chunk_size = action_chunk_size
        self.discount = discount
        self.infinite_horizon_backup = infinite_horizon_backup
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.min_shuffle_buffer_size = min(min_shuffle_buffer_size, shuffle_buffer_size)
        self.normalize_actions = normalize_actions
        self.action_clip_eps = action_clip_eps
        self.num_workers = num_workers
        self.tasks = {source.name: _read_tasks(source.data_dir) for source in sources}
        self.blank_task_indices = {
            source.name: _blank_task_indices(self.tasks[source.name])
            for source in sources
        }
        episodes_and_stats = {
            source.name: _read_episodes(source, self.blank_task_indices[source.name])
            for source in sources
        }
        self.episodes = {name: episodes for name, (episodes, _) in episodes_and_stats.items()}
        self.episode_stats = {name: stats for name, (_, stats) in episodes_and_stats.items()}
        self.labels = {source.name: _read_labels(source.label_path) for source in sources}
        self.weights = np.asarray([source.weight for source in sources], dtype=np.float64)
        self.weights /= self.weights.sum()
        _print_dataset_stats(sources, self.episodes, self.episode_stats, backup_horizon)

    def iterator(self, *, prefetch: int = 0, shuffle: bool | None = None, num_workers: int | None = None):
        dataset = _ChunkTransitionIterable(
            sources=self.sources,
            episodes=self.episodes,
            labels=self.labels,
            tasks=self.tasks,
            weights=self.weights,
            batch_size=self.batch_size,
            backup_horizon=self.backup_horizon,
            discount=self.discount,
            infinite_horizon_backup=self.infinite_horizon_backup,
            shuffle=self.shuffle if shuffle is None else shuffle,
            shuffle_buffer_size=self.shuffle_buffer_size,
            min_shuffle_buffer_size=self.min_shuffle_buffer_size,
            normalize_actions=self.normalize_actions,
            action_clip_eps=self.action_clip_eps,
        )
        workers = self.num_workers if num_workers is None else num_workers
        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=workers,
            persistent_workers=workers > 0,
            prefetch_factor=(prefetch if prefetch > 0 else 2) if workers > 0 else None,
            pin_memory=False,
            collate_fn=lambda x: x,
        )
        yield from loader


def make_bridge_soar_chunk_dataset(**kwargs) -> BridgeSoarChunkDataset:
    sources = [
        LeRobotSource(
            name=source["name"],
            data_dir=Path(source["data_dir"]),
            weight=source.get("weight", 1.0),
            video_key=source.get("video_key", "observation.images.image_0"),
            label_path=Path(source["label_path"]) if source.get("label_path") else None,
        )
        for source in kwargs.pop("sources")
    ]
    return BridgeSoarChunkDataset(sources=sources, **kwargs)


class _ChunkTransitionIterable(IterableDataset):
    def __init__(
        self,
        *,
        sources: list[LeRobotSource],
        episodes: dict[str, list[dict]],
        labels: dict[str, dict[str, str]],
        tasks: dict[str, dict[int, str]],
        weights: np.ndarray,
        batch_size: int,
        backup_horizon: int,
        discount: float,
        infinite_horizon_backup: bool,
        shuffle: bool,
        shuffle_buffer_size: int,
        min_shuffle_buffer_size: int,
        normalize_actions: bool,
        action_clip_eps: float,
    ):
        self.sources = sources
        self.episodes = episodes
        self.labels = labels
        self.tasks = tasks
        self.weights = weights
        self.batch_size = batch_size
        self.backup_horizon = backup_horizon
        self.discount = discount
        self.infinite_horizon_backup = infinite_horizon_backup
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.min_shuffle_buffer_size = min_shuffle_buffer_size
        self.normalize_actions = normalize_actions
        self.action_clip_eps = action_clip_eps

    def __iter__(self) -> Iterator[dict]:
        worker_info = get_worker_info()
        seed = None if worker_info is None else worker_info.seed
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        batch = []
        for transition in self._transition_iterator(rng, np_rng, worker_info):
            batch.append(transition)
            if len(batch) == self.batch_size:
                yield _stack_batch(batch)
                batch = []

    def _transition_iterator(self, rng, np_rng, worker_info):
        if not self.shuffle:
            source_episodes = [(source, episode) for source in self.sources for episode in self.episodes[source.name]]
            if worker_info is not None:
                source_episodes = source_episodes[worker_info.id :: worker_info.num_workers]
            while True:
                for source, episode in source_episodes:
                    yield from _episode_to_transitions(
                        source,
                        episode,
                        tasks=self.tasks[source.name],
                        labels=self.labels[source.name],
                        backup_horizon=self.backup_horizon,
                        discount=self.discount,
                        infinite_horizon_backup=self.infinite_horizon_backup,
                        normalize_actions=self.normalize_actions,
                        action_clip_eps=self.action_clip_eps,
                    )

        buffer = []
        while True:
            source = self.sources[np_rng.choice(len(self.sources), p=self.weights)]
            episode = rng.choice(self.episodes[source.name])
            for transition in _episode_to_transitions(
                source,
                episode,
                tasks=self.tasks[source.name],
                labels=self.labels[source.name],
                backup_horizon=self.backup_horizon,
                discount=self.discount,
                infinite_horizon_backup=self.infinite_horizon_backup,
                normalize_actions=self.normalize_actions,
                action_clip_eps=self.action_clip_eps,
            ):
                if len(buffer) < self.min_shuffle_buffer_size:
                    buffer.append(transition)
                    continue
                idx = rng.randrange(len(buffer))
                yield buffer.pop(idx)
                if len(buffer) < self.shuffle_buffer_size:
                    buffer.append(transition)


def _read_episodes(source: LeRobotSource, blank_task_indices: set[int]) -> tuple[list[dict], dict[str, int]]:
    with (source.data_dir / "meta" / "episodes.jsonl").open("r") as f:
        episodes = [json.loads(line) for line in f]
    usable = []
    stats = {
        "total_episodes": 0,
        "total_transitions": 0,
        "excluded_blank_language_episodes": 0,
        "excluded_blank_language_transitions": 0,
    }
    for episode in episodes:
        idx = episode["episode_index"]
        chunk = f"chunk-{idx // 1000:03d}"
        episode["path"] = source.data_dir / "data" / chunk / f"episode_{idx:06d}.parquet"
        episode["video_path"] = source.data_dir / "videos" / chunk / source.video_key / f"episode_{idx:06d}.mp4"
        if not episode["path"].exists():
            raise FileNotFoundError(f"Missing parquet for {source.name}: {episode['path']}")
        if not episode["video_path"].exists():
            raise FileNotFoundError(f"Missing video for {source.name}: {episode['video_path']}")
        if int(episode.get("length", 0)) <= 1:
            raise ValueError(f"Episode has length <= 1 for {source.name}: {episode['path']}")
        transitions = max(int(episode.get("length", 0)) - 1, 0)
        stats["total_episodes"] += 1
        stats["total_transitions"] += transitions
        task_index = _read_episode_task_index(episode["path"])
        episode["task_index"] = task_index
        if task_index in blank_task_indices:
            stats["excluded_blank_language_episodes"] += 1
            stats["excluded_blank_language_transitions"] += transitions
            continue
        usable.append(episode)
    if not usable:
        raise RuntimeError(f"No LeRobot episodes found in {source.data_dir}")
    return usable, stats


def _read_tasks(data_dir: Path) -> dict[int, str]:
    with (data_dir / "meta" / "tasks.jsonl").open("r") as f:
        return {int(task["task_index"]): _normalize_task_text(task.get("task")) for task in (json.loads(line) for line in f)}


def _normalize_task_text(task: object) -> str:
    return "" if task is None else str(task)


def _blank_task_indices(tasks: dict[int, str]) -> set[int]:
    return {task_index for task_index, task in tasks.items() if not task.strip()}


def _read_episode_task_index(path: Path) -> int:
    return int(pq.read_table(path, columns=["task_index"]).column("task_index")[0].as_py())


def _read_labels(label_path: Path | None) -> dict[str, str]:
    if label_path is None:
        return {}
    with label_path.open("r") as f:
        labels = json.load(f)
    return labels.get("label_corrections", labels)


def _print_dataset_stats(
    sources: list[LeRobotSource],
    episodes_by_source: dict[str, list[dict]],
    episode_stats: dict[str, dict[str, int]],
    backup_horizon: int,
) -> None:
    print("Bridge/SOAR LeRobot dataset stats:")
    for source in sources:
        episodes = episodes_by_source[source.name]
        stats = episode_stats[source.name]
        transitions = sum(max(int(ep.get("length", 0)) - 1, 0) for ep in episodes)
        print(
            f"- {source.name}: data_dir={source.data_dir}, weight={source.weight}, "
            f"total_episodes={stats['total_episodes']}, total_transitions={stats['total_transitions']}, "
            f"excluded_blank_language_episodes={stats['excluded_blank_language_episodes']}, "
            f"excluded_blank_language_transitions={stats['excluded_blank_language_transitions']}, "
            f"usable_episodes={len(episodes)}, transitions={transitions}, backup_horizon={backup_horizon}"
        )


def _episode_to_transitions(source, episode, *, tasks, labels, backup_horizon, discount, infinite_horizon_backup, normalize_actions, action_clip_eps):
    batch = _episode_to_batch(
        source,
        episode,
        tasks=tasks,
        labels=labels,
        backup_horizon=backup_horizon,
        discount=discount,
        infinite_horizon_backup=infinite_horizon_backup,
        normalize_actions=normalize_actions,
        action_clip_eps=action_clip_eps,
    )
    for i in range(batch["actions"].shape[0]):
        yield {k: _slice_tree(v, i) for k, v in batch.items()}


def _episode_to_batch(source, episode, *, tasks, labels, backup_horizon, discount, normalize_actions, action_clip_eps=1e-5, infinite_horizon_backup=False) -> dict:
    data = pq.read_table(episode["path"]).to_pydict()
    actions = np.asarray(data["action"], dtype=np.float32)
    if normalize_actions:
        actions = 2.0 * (actions - BRIDGE_ACTION_LOW) / (BRIDGE_ACTION_HIGH - BRIDGE_ACTION_LOW) - 1.0
        actions = np.clip(actions, -1.0 + action_clip_eps, 1.0 - action_clip_eps)

    if "reward" in data:
        rewards_full = np.asarray(data["reward"], dtype=np.float32)
    else:
        rewards_full = np.asarray(_rewards_from_label(episode, labels, len(actions)), dtype=np.float32)
    images = _read_video_frames(episode, len(actions))
    horizon = len(actions) - 1
    if horizon <= 0:
        raise ValueError(f"Episode {episode['path']} has fewer than two frames")

    actions = actions[:horizon]
    rewards = rewards_full[:horizon]
    task_index = int(np.asarray(data["task_index"])[0])
    language_text = tasks[task_index]

    next_idxs = np.minimum(np.arange(horizon) + backup_horizon, horizon)
    if infinite_horizon_backup:
        masks = np.ones(horizon, dtype=np.float32)
        high_value_backup_horizon = np.full(horizon, backup_horizon, dtype=np.float32)
    else:
        masks = (np.arange(horizon) + backup_horizon < horizon).astype(np.float32)
        high_value_backup_horizon = np.minimum(backup_horizon, horizon - np.arange(horizon)).astype(np.float32)
    n_step_rewards = np.zeros(horizon, dtype=np.float32)
    action_chunks = np.zeros((horizon, backup_horizon, actions.shape[-1]), dtype=np.float32)
    valid_chunks = np.zeros((horizon, backup_horizon), dtype=np.float32)
    mc_returns = _discounted_returns(rewards, np.concatenate([np.ones(horizon - 1, dtype=np.float32), np.zeros(1, dtype=np.float32)]), discount)

    discounts = discount ** np.arange(backup_horizon, dtype=np.float32)
    for t in range(horizon):
        end = min(t + backup_horizon, horizon)
        n = end - t
        n_step_rewards[t] = np.sum(discounts[:n] * rewards[t:end])
        action_chunks[t, :n] = actions[t:end]
        if n < backup_horizon and not infinite_horizon_backup:
            action_chunks[t, n:] = actions[end - 1]
        valid_chunks[t, :n] = 1.0
        if infinite_horizon_backup:
            valid_chunks[t, n:] = 1.0

    language = np.asarray([language_text.encode("utf-8")] * horizon)
    return {
        "observations": {"image": images[:horizon]},
        "next_observations": {"image": images[next_idxs]},
        "actions": actions,
        "high_value_action_chunks": action_chunks.reshape(horizon, -1),
        "high_value_next_observations": {"image": images[next_idxs]},
        "goals": {"language": language},
        "high_value_goals": {"language": language},
        "rewards": n_step_rewards,
        "masks": masks,
        "high_value_rewards": n_step_rewards,
        "high_value_masks": masks,
        "high_value_backup_horizon": high_value_backup_horizon,
        "valids": valid_chunks,
        "mc_returns": mc_returns,
    }


def _read_video_frames(episode: dict, length: int) -> np.ndarray:
    frames = iio.imread(episode["video_path"])
    if frames.shape[0] < length:
        raise ValueError(f"Video {episode['video_path']} has {frames.shape[0]} frames, expected at least {length}")
    return np.asarray(frames[:length], dtype=np.uint8)


def _rewards_from_label(episode: dict, labels: dict[str, str], length: int) -> np.ndarray:
    if labels and str(episode["episode_index"]) not in labels:
        raise KeyError(f"Missing label correction for episode {episode['episode_index']}")
    label = labels.get(str(episode["episode_index"]))
    if label == "failure":
        return np.zeros(length, dtype=np.float32)
    if label in ("successful", "already_complete", None):
        return _bridge_rewards(length)
    raise ValueError(f"Unknown episode label {label!r} for episode {episode['episode_index']}")


def _bridge_rewards(length: int) -> np.ndarray:
    cutoff = int(np.floor(0.95 * length))
    rewards = np.ones(length, dtype=np.float32)
    if cutoff > 1:
        rewards[:cutoff] = np.linspace(0.0, 1.0, cutoff, dtype=np.float32)
    return rewards


def _discounted_returns(rewards: np.ndarray, masks: np.ndarray, discount: float) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + discount * running * masks[i]
        returns[i] = running
    return returns


def _slice_tree(value, i: int):
    if isinstance(value, dict):
        return {k: _slice_tree(v, i) for k, v in value.items()}
    return value[i]


def _stack_batch(batch: list[dict]) -> dict:
    out = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], dict):
            out[key] = _stack_batch(values)
        else:
            out[key] = np.stack(values)
    return out
