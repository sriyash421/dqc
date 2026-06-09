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


def make_text_processor(name: str):
    if name == "muse_embedding":
        return MuseEmbedding()
    raise ValueError(f"Unknown text processor: {name}")
