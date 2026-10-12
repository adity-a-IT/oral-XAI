"""Dataset loading utilities for TensorFlow/Keras pipelines.

This module is intentionally structured to support patient-level dataset splits once the
actual metadata is confirmed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import tensorflow as tf


def make_dataset(
    image_paths: Sequence[str],
    labels: Sequence[int],
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
):
    """Create a tf.data pipeline from image paths and labels."""
    if len(image_paths) != len(labels):
        raise ValueError("The number of image paths and labels must match.")

    def load_and_preprocess(path: str, label: int):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3, dtype=tf.uint8)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, tf.cast(label, tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((list(image_paths), list(labels)))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=seed)
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def list_dataset_files(root: str | Path) -> list[str]:
    """Return all image files under the dataset root."""
    base = Path(root).expanduser().resolve()
    return [str(path) for path in sorted(base.rglob("*")) if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]


if __name__ == "__main__":
    print("Dataset loader ready. Actual image/label mapping is created after dataset inspection.")
