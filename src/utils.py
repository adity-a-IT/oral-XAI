"""Utility helpers for the oral lesion XAI project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    dir_path = Path(path).expanduser().resolve()
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_json(data: Any, path: str | Path) -> None:
    """Save a Python object to a JSON file."""
    file_path = Path(path).expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def set_seed(seed: int = 42) -> None:
    """Set a fixed random seed for reproducibility."""
    import random
    import numpy as np
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


if __name__ == "__main__":
    print("Utilities are ready for reproducible experimentation.")
