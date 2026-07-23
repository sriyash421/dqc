import os

import numpy as np


MULTI_MODULE = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"


class TextProcessor:
    def encode(self, strings):
        raise NotImplementedError


class MuseEmbedding(TextProcessor):
    def __init__(self):
        import tensorflow as tf
        import tensorflow_hub as hub
        import tensorflow_text  # noqa: F401

        self._tf = tf
        self._model = hub.load(MULTI_MODULE)

    def encode(self, strings):
        with self._tf.device("/cpu:0"):
            return np.asarray(self._model(strings).numpy(), dtype=np.float32)


class PrecomputedEmbedding(TextProcessor):
    """Returns a single fixed MUSE/USE-3 embedding for every input string. Use when
    the task is a single fixed instruction (e.g. eggplant) and tensorflow_text is
    unavailable — numerically identical to live MUSE of that instruction. The .npy
    path comes from DQC_TASK_EMB (a (512,) USE-3 embedding, e.g. task_emb_use3.npy)."""

    def __init__(self, emb_path=None):
        emb_path = emb_path or os.environ["DQC_TASK_EMB"]
        emb = np.asarray(np.load(emb_path), dtype=np.float32).reshape(-1)
        if emb.shape[0] != 512:
            raise ValueError(f"expected a (512,) embedding, got {emb.shape} from {emb_path}")
        self._emb = emb

    def encode(self, strings):
        return np.repeat(self._emb[None], len(strings), axis=0).astype(np.float32)


def make_text_processor(name: str):
    if name == "muse_embedding":
        return MuseEmbedding()
    if name in ("precomputed", "precomputed_muse"):
        return PrecomputedEmbedding()
    raise ValueError(f"Unknown text processor: {name}")
