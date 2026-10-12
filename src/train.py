"""Training entry point skeleton for the oral lesion project.

This file is intentionally structured to support experiments once the dataset is available.
The actual training logic is not run until the dataset inspection and patient-level split are
complete and validated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf


@dataclass
class TrainingConfig:
    model_name: str = "custom_cnn"
    input_shape: tuple[int, int, int] = (224, 224, 3)
    num_classes: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 50
    seed: int = 42
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 3


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-4) -> tf.keras.Model:
    """Compile the model with the research-standard training setup."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_config(config: TrainingConfig, output_path: str | Path) -> None:
    """Persist training config for reproducibility."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2)


def train_model(model: tf.keras.Model, train_ds, val_ds, config: TrainingConfig):
    """Run training with callbacks for early stopping and learning-rate reduction.

    This function is intentionally left as a structured skeleton because the real data pipeline
    and split are created after dataset inspection.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/best_model.keras",
            save_best_only=True,
            monitor="val_loss",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.reduce_lr_patience,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history


if __name__ == "__main__":
    print("Training entry point is ready. It will be used after the dataset inspection and split are confirmed.")
