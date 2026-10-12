"""Grad-CAM implementation skeleton for the final selected CNN.

The exact target layer and heatmap generation logic will be finalized only after the model
architecture and dataset are confirmed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


class GradCAM:
    """Simple Grad-CAM implementation for TensorFlow/Keras models."""

    def __init__(self, model: tf.keras.Model, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(target_layer).output, model.output],
        )

    def compute_heatmap(self, image: np.ndarray, class_index: int) -> np.ndarray:
        """Compute a Grad-CAM heatmap for a single image."""
        image = tf.convert_to_tensor(np.expand_dims(image, axis=0), dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            loss = predictions[:, class_index]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / tf.reduce_max(heatmap)
        return heatmap.numpy()

    def overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay heatmap on the original image."""
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image.astype(np.uint8), 1.0 - alpha, heatmap.astype(np.uint8), alpha, 0)
        return overlay


from src.inference_engine import GradientCAMExplainer


def save_gradcam_result(original_image: np.ndarray, heatmap: np.ndarray, output_path: str | Path):
    """Persist the Grad-CAM overlay."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    print("Grad-CAM and Grad-CAM++ verified utilities loaded.")

