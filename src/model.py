"""Model definitions for the oral lesion research project.

This file contains baseline transfer-learning models and a custom lightweight CNN that is
intended to be implemented independently after the dataset audit is complete.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50, VGG16
from tensorflow.keras.models import Model


def build_vgg16_model(input_shape: tuple[int, int, int] = (224, 224, 3), num_classes: int = 4):
    """Transfer-learning VGG16 backbone for oral lesion classification."""
    base_model = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
    base_model.trainable = False
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=outputs)


def build_resnet50_model(input_shape: tuple[int, int, int] = (224, 224, 3), num_classes: int = 4):
    """Transfer-learning ResNet50 backbone for oral lesion classification."""
    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    base_model.trainable = False
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=outputs)


def build_mobilenetv2_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    dropout_rate: float = 0.3,
):
    """Transfer-learning MobileNetV2 backbone for oral lesion classification."""
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    base_model.trainable = False
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = Model(inputs=base_model.input, outputs=outputs, name="mobilenetv2_oral")
    return model, base_model


def build_efficientnetb0_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    dropout_rate: float = 0.3,
):
    """Transfer-learning EfficientNetB0 backbone for oral lesion classification."""
    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
    base_model.trainable = False
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = Model(inputs=base_model.input, outputs=outputs, name="efficientnetb0_oral")
    return model, base_model


def build_custom_oral_cnn(input_shape: tuple[int, int, int] = (224, 224, 3), num_classes: int = 4):
    """Lightweight CNN inspired by the ETASR methodology.

    This is intentionally implemented as a clean, independent architecture skeleton without
    copying proprietary code. The exact number of blocks and dropout placements will be
    finalized based on the actual dataset and benchmark results.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    model = build_custom_oral_cnn()
    model.summary()
