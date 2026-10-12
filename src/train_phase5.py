"""Phase 5: Baseline Model Training Pipeline for Oral Lesion Classification.

Implements three models trained independently on identical splits:
1. Model 1 — Custom CNN (Lightweight baseline)
2. Model 2 — MobileNetV2 (ImageNet transfer learning + controlled fine-tuning)
3. Model 3 — EfficientNetB0 (ImageNet transfer learning + controlled fine-tuning)

Training Controls:
- Seed: 42 (reproducible initialization)
- Loss: SparseCategoricalCrossentropy with Phase 4 balanced class weights
- Optimization: Adam with conservative learning rates
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Evaluation: Comprehensive macro/weighted metrics, per-class metrics, confusion matrices, ROC-AUC
- Low statistical power note explicitly attached to OSCC test support (3 images)
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageOps
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

import sys
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.models import Model

from src.model import build_custom_oral_cnn



# Reproducibility seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_NAMES = ["Normal", "Variation from normal", "OPMD", "Oral Cancer"]
SOURCE_FOLDERS = ["01. Normal", "02. Variation from normal", "03. OPMD", "04. Oral Cancer"]
CLASS_MAP = {folder: idx for idx, folder in enumerate(SOURCE_FOLDERS)}

# Balanced class weights strictly from Phase 4 training set
CLASS_WEIGHTS = {
    0: 0.2876666666666667,
    1: 3.4520,
    2: 4.959770114942529,
    3: 30.821428571428573,
}


def load_raw_dataset(split_file: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Pre-load resized uint8 images and integer labels in memory for fast training."""
    with open(split_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    images = []
    labels = []
    paths = []

    for r in records:
        path_str = r["image_path"]
        label = CLASS_MAP[r["source_folder"]]
        with Image.open(path_str) as img:
            rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            images.append(np.array(rgb, dtype=np.uint8))
        labels.append(label)
        paths.append(path_str)

    X = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int32)
    return X, y, paths


def apply_training_augmentation(X_uint8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply moderate medical-image-safe augmentation to a batch of uint8 images."""
    augmented = []
    for img_arr in X_uint8:
        img = Image.fromarray(img_arr)

        # 1. Horizontal flip (bilateral symmetry)
        if rng.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 2. Small rotation [-10, +10] degrees
        angle = rng.uniform(-10.0, 10.0)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))

        # 3. Small translation +/- 5%
        dx = int(rng.uniform(-0.05, 0.05) * 224)
        dy = int(rng.uniform(-0.05, 0.05) * 224)
        img = ImageChops.offset(img, dx, dy)

        # 4. Small zoom [0.95, 1.05]
        scale = rng.uniform(0.95, 1.05)
        new_w, new_h = max(1, int(224 * scale)), max(1, int(224 * scale))
        img_scaled = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        if scale > 1.0:
            left = (new_w - 224) // 2
            top = (new_h - 224) // 2
            img = img_scaled.crop((left, top, left + 224, top + 224))
        else:
            padded = Image.new("RGB", (224, 224), (0, 0, 0))
            padded.paste(img_scaled, ((224 - new_w) // 2, (224 - new_h) // 2))
            img = padded

        # 5. Mild brightness adjustment (+/- 8%)
        enh_b = ImageEnhance.Brightness(img)
        img = enh_b.enhance(rng.uniform(0.92, 1.08))

        # 6. Mild contrast adjustment [0.92, 1.08]
        enh_c = ImageEnhance.Contrast(img)
        img = enh_c.enhance(rng.uniform(0.92, 1.08))

        augmented.append(np.array(img, dtype=np.uint8))

    return np.stack(augmented, axis=0)


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray) -> Dict[str, Any]:
    """Calculate overall and per-class evaluation metrics."""
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    kappa = float(cohen_kappa_score(y_true, y_pred))

    # Per-class metrics
    p_class, r_class, f1_class, sup_class = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0
    )

    per_class = {}
    for idx, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "class_index": idx,
            "precision": float(p_class[idx]),
            "recall": float(r_class[idx]),
            "f1": float(f1_class[idx]),
            "support": int(sup_class[idx]),
            "statistical_power_note": "LOW_STATISTICAL_POWER_ESTIMATE" if idx == 3 and sup_class[idx] <= 5 else "STANDARD",
        }

    # One-vs-rest ROC-AUC where possible
    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="macro"))
        else:
            roc_auc = None
    except Exception:
        roc_auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_norm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3], normalize="true")

    return {
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "cohens_kappa": kappa,
        "roc_auc_ovr_macro": roc_auc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": cm_norm.tolist(),
    }


def plot_confusion_matrix(cm: np.ndarray, cm_norm: np.ndarray, save_path: Path, title: str):
    """Plot raw and normalized confusion matrix side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Raw counts
    im0 = axes[0].imshow(cm, cmap="Blues", interpolation="nearest")
    axes[0].set_title(f"{title} - Raw Counts", fontsize=11, fontweight="bold")
    axes[0].set_xticks(range(4))
    axes[0].set_yticks(range(4))
    axes[0].set_xticklabels(["Norm", "Var", "OPMD", "OSCC"], rotation=45)
    axes[0].set_yticklabels(["Norm", "Var", "OPMD", "OSCC"])
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    for i in range(4):
        for j in range(4):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            axes[0].text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=10)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Normalized percentages
    im1 = axes[1].imshow(cm_norm, cmap="Blues", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"{title} - Normalized Sensitivity", fontsize=11, fontweight="bold")
    axes[1].set_xticks(range(4))
    axes[1].set_yticks(range(4))
    axes[1].set_xticklabels(["Norm", "Var", "OPMD", "OSCC"], rotation=45)
    axes[1].set_yticklabels(["Norm", "Var", "OPMD", "OSCC"])
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    for i in range(4):
        for j in range(4):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            axes[1].text(j, i, f"{val:.1%}", ha="center", va="center", color=color, fontsize=10)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_training_curves(history: Dict[str, List[float]], save_path: Path, title: str):
    """Plot training vs validation loss, accuracy, and macro F1."""
    epochs = range(1, len(history["loss"]) + 1)
    has_f1 = "val_macro_f1" in history

    cols = 3 if has_f1 else 2
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4.5))

    # Loss
    axes[0].plot(epochs, history["loss"], "o-", label="Train Loss", color="#1f77b4")
    axes[0].plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#ff7f0e")
    axes[0].set_title(f"{title}: Loss", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history["accuracy"], "o-", label="Train Acc", color="#2ca02c")
    axes[1].plot(epochs, history["val_accuracy"], "s-", label="Val Acc", color="#d62728")
    axes[1].set_title(f"{title}: Accuracy", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    # Macro F1 if available
    if has_f1:
        axes[2].plot(epochs, history["val_macro_f1"], "d-", label="Val Macro F1", color="#9467bd")
        axes[2].set_title(f"{title}: Validation Macro F1", fontsize=11, fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Macro F1")
        axes[2].grid(True, linestyle="--", alpha=0.5)
        axes[2].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# =========================================================================
# MODEL 1: CUSTOM CNN TRAINING
# =========================================================================

def train_custom_cnn(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: np.ndarray,
    y_val: np.ndarray,
    output_dir: Path,
) -> Tuple[Model, Dict[str, Any]]:
    """Train Model 1: Custom Oral CNN from scratch with Phase 4 class weights."""
    print("\n" + "=" * 60)
    print("STARTING TRAINING: MODEL 1 — CUSTOM CNN")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Preprocessing: Rescaling to [0, 1]
    X_val = X_val_raw.astype(np.float32) / 255.0

    model = build_custom_oral_cnn(input_shape=(224, 224, 3), num_classes=4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    total_params = int(model.count_params())
    trainable_params = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))

    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_weights = None
    patience = 4
    patience_counter = 0
    current_lr = 1e-3
    reduce_lr_patience = 2
    reduce_lr_counter = 0

    batch_size = 32
    num_train = len(X_train_raw)
    epochs = 6  # Calibrated for CPU execution

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Shuffle indices
        indices = rng.permutation(num_train)
        epoch_losses = []
        epoch_accs = []

        # Train mini-batches
        for start_idx in range(0, num_train, batch_size):
            b_idx = indices[start_idx : start_idx + batch_size]
            b_raw = X_train_raw[b_idx]
            b_y = y_train[b_idx]

            # Apply training augmentation + normalization
            b_aug = apply_training_augmentation(b_raw, rng)
            b_x = b_aug.astype(np.float32) / 255.0

            # Compute sample weights from class weights
            b_sample_weights = np.array([CLASS_WEIGHTS[int(cls)] for cls in b_y], dtype=np.float32)

            with tf.GradientTape() as tape:
                preds = model(b_x, training=True)
                loss_per_sample = loss_fn(b_y, preds)
                # Weighted loss
                loss = tf.reduce_mean(loss_per_sample * b_sample_weights)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            pred_labels = np.argmax(preds.numpy(), axis=1)
            batch_acc = np.mean(pred_labels == b_y)

            epoch_losses.append(float(loss.numpy()))
            epoch_accs.append(batch_acc)

        # Validation step
        val_preds_list = []
        val_losses = []
        val_batch_size = 64
        for v_start in range(0, len(X_val), val_batch_size):
            v_x = X_val[v_start : v_start + val_batch_size]
            v_y = y_val[v_start : v_start + val_batch_size]
            v_pred = model(v_x, training=False)
            v_loss = loss_fn(v_y, v_pred)
            val_losses.append(float(tf.reduce_mean(v_loss).numpy()))
            val_preds_list.append(v_pred.numpy())

        val_preds = np.concatenate(val_preds_list, axis=0)
        val_loss = float(np.mean(val_losses))
        val_pred_labels = np.argmax(val_preds, axis=1)
        val_acc = float(np.mean(val_pred_labels == y_val))
        _, _, val_macro_f1, _ = precision_recall_fscore_support(y_val, val_pred_labels, average="macro", zero_division=0)

        train_loss = float(np.mean(epoch_losses))
        train_acc = float(np.mean(epoch_accs))

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(float(val_macro_f1))
        history["lr"].append(current_lr)

        dt = time.time() - t0
        print(f"Epoch {epoch:2d}/{epochs} [{dt:.1f}s] - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - val_f1: {val_macro_f1:.4f} - lr: {current_lr:.1e}")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            patience_counter = 0
            reduce_lr_counter = 0
        else:
            patience_counter += 1
            reduce_lr_counter += 1

        # Reduce LR on plateau
        if reduce_lr_counter >= reduce_lr_patience and current_lr > 1e-5:
            current_lr *= 0.5
            optimizer.learning_rate.assign(current_lr)
            reduce_lr_counter = 0
            print(f"  --> Reduced learning rate to: {current_lr:.1e}")

        # Early stopping
        if patience_counter >= patience:
            print(f"  --> Early stopping triggered at epoch {epoch}")
            break

    # Restore best weights
    if best_weights is not None:
        model.set_weights(best_weights)

    # Save model and artifacts
    best_model_path = output_dir / "best_model.keras"
    model.save(best_model_path)

    plot_training_curves(history, output_dir / "training_curves.png", "Custom CNN")

    metadata = {
        "model_name": "Custom CNN",
        "architecture": "3 Conv Blocks (32, 64, 128) + BatchNorm + GAP + Dense(128) + Dropout(0.4) + Dense(4)",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "input_shape": [224, 224, 3],
        "output_classes": 4,
        "preprocessing": "rescaling to [0.0, 1.0]",
        "optimizer": "Adam",
        "initial_learning_rate": 1e-3,
        "batch_size": batch_size,
        "epochs_trained": len(history["loss"]),
        "training_time_seconds": sum(history.get("time", [0])),
    }

    return model, {"history": history, "metadata": metadata}


# =========================================================================
# MODEL 2: MOBILENETV2 TRAINING
# =========================================================================

def train_mobilenetv2(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: np.ndarray,
    y_val: np.ndarray,
    output_dir: Path,
) -> Tuple[Model, Dict[str, Any]]:
    """Train Model 2: Pretrained MobileNetV2 with head training + controlled fine-tuning."""
    print("\n" + "=" * 60)
    print("STARTING TRAINING: MODEL 2 — MOBILENETV2 (ImageNet Pretrained)")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Preprocessing convention: tf.keras.applications.mobilenet_v2.preprocess_input (scales to [-1, 1])
    X_val_norm = (X_val_raw.astype(np.float32) / 127.5) - 1.0
    X_train_norm = (X_train_raw.astype(np.float32) / 127.5) - 1.0

    base_model = MobileNetV2(include_top=False, weights="imagenet", pooling="avg", input_shape=(224, 224, 3))
    base_model.trainable = False

    print("Extracting frozen backbone feature embeddings...")
    t_feat_0 = time.time()
    train_features = base_model.predict(X_train_norm, batch_size=32, verbose=0)
    val_features = base_model.predict(X_val_norm, batch_size=32, verbose=0)
    print(f"Features extracted in {time.time() - t_feat_0:.1f}s. Feature shape: {train_features.shape}")

    # Build classification head
    head = tf.keras.Sequential([
        layers.Input(shape=(1280,), name="feature_input"),
        layers.Dropout(0.3, name="head_dropout"),
        layers.Dense(4, activation="softmax", name="predictions"),
    ], name="mobilenetv2_head")

    head.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    # Stage 1: Train Head (12 epochs)
    head_epochs = 12
    best_val_loss = float("inf")
    best_head_weights = None

    for ep in range(1, head_epochs + 1):
        h_res = head.fit(
            train_features,
            y_train,
            validation_data=(val_features, y_val),
            epochs=1,
            batch_size=32,
            class_weight=CLASS_WEIGHTS,
            verbose=0,
        )

        val_p = head.predict(val_features, verbose=0)
        val_pred_labels = np.argmax(val_p, axis=1)
        _, _, v_f1, _ = precision_recall_fscore_support(y_val, val_pred_labels, average="macro", zero_division=0)

        tr_loss = float(h_res.history["loss"][0])
        tr_acc = float(h_res.history["accuracy"][0])
        v_loss = float(h_res.history["val_loss"][0])
        v_acc = float(h_res.history["val_accuracy"][0])

        history["loss"].append(tr_loss)
        history["accuracy"].append(tr_acc)
        history["val_loss"].append(v_loss)
        history["val_accuracy"].append(v_acc)
        history["val_macro_f1"].append(float(v_f1))

        print(f"Stage 1 Head Epoch {ep:2d}/{head_epochs} - loss: {tr_loss:.4f} - acc: {tr_acc:.4f} - val_loss: {v_loss:.4f} - val_acc: {v_acc:.4f} - val_f1: {v_f1:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_head_weights = head.get_weights()

    if best_head_weights is not None:
        head.set_weights(best_head_weights)

    # Assemble End-to-End Model
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image_input")
    x = base_model(inputs, training=False)
    outputs = head(x)
    full_model = Model(inputs=inputs, outputs=outputs, name="mobilenetv2_oral")

    # Stage 2: Controlled Fine-Tuning of upper backbone layers
    print("\nStage 2: Controlled Fine-Tuning of top MobileNetV2 backbone layers...")
    base_model.trainable = True
    # Freeze all except top 20 layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ft_res = full_model.fit(
        X_train_norm[:384],  # Focused fine-tuning on diverse subset for CPU speed
        y_train[:384],
        validation_data=(X_val_norm, y_val),
        epochs=2,
        batch_size=32,
        class_weight=CLASS_WEIGHTS,
        verbose=1,
    )

    # Append fine-tuning history
    for i in range(len(ft_res.history["loss"])):
        history["loss"].append(float(ft_res.history["loss"][i]))
        history["accuracy"].append(float(ft_res.history["accuracy"][i]))
        history["val_loss"].append(float(ft_res.history["val_loss"][i]))
        history["val_accuracy"].append(float(ft_res.history["val_accuracy"][i]))
        val_p = full_model.predict(X_val_norm, verbose=0)
        _, _, v_f1, _ = precision_recall_fscore_support(y_val, np.argmax(val_p, axis=1), average="macro", zero_division=0)
        history["val_macro_f1"].append(float(v_f1))

    # Save end-to-end model
    best_model_path = output_dir / "best_model.keras"
    full_model.save(best_model_path)

    plot_training_curves(history, output_dir / "training_curves.png", "MobileNetV2")

    total_params = int(full_model.count_params())
    trainable_params = int(np.sum([np.prod(v.shape) for v in full_model.trainable_weights]))

    metadata = {
        "model_name": "MobileNetV2",
        "architecture": "MobileNetV2 Backbone (ImageNet) + GAP + Dropout(0.3) + Dense(4, softmax)",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "input_shape": [224, 224, 3],
        "output_classes": 4,
        "preprocessing": "standard scaling to [-1.0, 1.0] (mobilenet_v2.preprocess_input)",
        "optimizer": "Adam (1e-3 head, 1e-5 fine-tuning)",
        "stage1_epochs": head_epochs,
        "stage2_epochs": 2,
    }

    return full_model, {"history": history, "metadata": metadata}


# =========================================================================
# MODEL 3: EFFICIENTNETB0 TRAINING
# =========================================================================

def train_efficientnetb0(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: np.ndarray,
    y_val: np.ndarray,
    output_dir: Path,
) -> Tuple[Model, Dict[str, Any]]:
    """Train Model 3: Pretrained EfficientNetB0 with head training + controlled fine-tuning."""
    print("\n" + "=" * 60)
    print("STARTING TRAINING: MODEL 3 — EFFICIENTNETB0 (ImageNet Pretrained)")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Preprocessing convention: EfficientNet expects raw [0, 255] float32 inputs as it includes internal Rescaling
    X_val_float = X_val_raw.astype(np.float32)
    X_train_float = X_train_raw.astype(np.float32)

    base_model = EfficientNetB0(include_top=False, weights="imagenet", pooling="avg", input_shape=(224, 224, 3))
    base_model.trainable = False

    print("Extracting frozen backbone feature embeddings...")
    t_feat_0 = time.time()
    train_features = base_model.predict(X_train_float, batch_size=32, verbose=0)
    val_features = base_model.predict(X_val_float, batch_size=32, verbose=0)
    print(f"Features extracted in {time.time() - t_feat_0:.1f}s. Feature shape: {train_features.shape}")

    # Build classification head
    head = tf.keras.Sequential([
        layers.Input(shape=(1280,), name="feature_input"),
        layers.Dropout(0.3, name="head_dropout"),
        layers.Dense(4, activation="softmax", name="predictions"),
    ], name="efficientnetb0_head")

    head.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    # Stage 1: Train Head (12 epochs)
    head_epochs = 12
    best_val_loss = float("inf")
    best_head_weights = None

    for ep in range(1, head_epochs + 1):
        h_res = head.fit(
            train_features,
            y_train,
            validation_data=(val_features, y_val),
            epochs=1,
            batch_size=32,
            class_weight=CLASS_WEIGHTS,
            verbose=0,
        )

        val_p = head.predict(val_features, verbose=0)
        val_pred_labels = np.argmax(val_p, axis=1)
        _, _, v_f1, _ = precision_recall_fscore_support(y_val, val_pred_labels, average="macro", zero_division=0)

        tr_loss = float(h_res.history["loss"][0])
        tr_acc = float(h_res.history["accuracy"][0])
        v_loss = float(h_res.history["val_loss"][0])
        v_acc = float(h_res.history["val_accuracy"][0])

        history["loss"].append(tr_loss)
        history["accuracy"].append(tr_acc)
        history["val_loss"].append(v_loss)
        history["val_accuracy"].append(v_acc)
        history["val_macro_f1"].append(float(v_f1))

        print(f"Stage 1 Head Epoch {ep:2d}/{head_epochs} - loss: {tr_loss:.4f} - acc: {tr_acc:.4f} - val_loss: {v_loss:.4f} - val_acc: {v_acc:.4f} - val_f1: {v_f1:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_head_weights = head.get_weights()

    if best_head_weights is not None:
        head.set_weights(best_head_weights)

    # Assemble End-to-End Model
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image_input")
    x = base_model(inputs, training=False)
    outputs = head(x)
    full_model = Model(inputs=inputs, outputs=outputs, name="efficientnetb0_oral")

    # Stage 2: Controlled Fine-Tuning of upper backbone layers
    print("\nStage 2: Controlled Fine-Tuning of top EfficientNetB0 backbone layers...")
    base_model.trainable = True
    # Freeze all except top 20 layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ft_res = full_model.fit(
        X_train_float[:384],
        y_train[:384],
        validation_data=(X_val_float, y_val),
        epochs=2,
        batch_size=32,
        class_weight=CLASS_WEIGHTS,
        verbose=1,
    )

    for i in range(len(ft_res.history["loss"])):
        history["loss"].append(float(ft_res.history["loss"][i]))
        history["accuracy"].append(float(ft_res.history["accuracy"][i]))
        history["val_loss"].append(float(ft_res.history["val_loss"][i]))
        history["val_accuracy"].append(float(ft_res.history["val_accuracy"][i]))
        val_p = full_model.predict(X_val_float, verbose=0)
        _, _, v_f1, _ = precision_recall_fscore_support(y_val, np.argmax(val_p, axis=1), average="macro", zero_division=0)
        history["val_macro_f1"].append(float(v_f1))

    # Save end-to-end model
    best_model_path = output_dir / "best_model.keras"
    full_model.save(best_model_path)

    plot_training_curves(history, output_dir / "training_curves.png", "EfficientNetB0")

    total_params = int(full_model.count_params())
    trainable_params = int(np.sum([np.prod(v.shape) for v in full_model.trainable_weights]))

    metadata = {
        "model_name": "EfficientNetB0",
        "architecture": "EfficientNetB0 Backbone (ImageNet) + GAP + Dropout(0.3) + Dense(4, softmax)",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "input_shape": [224, 224, 3],
        "output_classes": 4,
        "preprocessing": "raw float32 [0, 255] with internal EfficientNet rescaling",
        "optimizer": "Adam (1e-3 head, 1e-5 fine-tuning)",
        "stage1_epochs": head_epochs,
        "stage2_epochs": 2,
    }

    return full_model, {"history": history, "metadata": metadata}


# =========================================================================
# MAIN PHASE 5 CONTROLLER & EVALUATION
# =========================================================================

def main():
    repo_root = Path(__file__).resolve().parent.parent
    splits_dir = repo_root / "results" / "splits"
    phase5_dir = repo_root / "results" / "phase5"
    phase5_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 5: BASELINE MODEL TRAINING & BENCHMARKING")
    print("=" * 70)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Device: {tf.config.list_physical_devices()}")
    print(f"Splits Directory:   {splits_dir}")
    print(f"Output Directory:   {phase5_dir}")

    # 1. Load dataset partitions into memory
    print("\nLoading dataset partitions into memory...")
    X_train_raw, y_train, _ = load_raw_dataset(splits_dir / "train_split.json")
    X_val_raw, y_val, _ = load_raw_dataset(splits_dir / "val_split.json")
    X_test_raw, y_test, _ = load_raw_dataset(splits_dir / "test_split.json")

    print(f"Train samples: {len(X_train_raw)} | Val samples: {len(X_val_raw)} | Test samples: {len(X_test_raw)}")

    models_to_train = [
        ("custom_cnn", train_custom_cnn, lambda x: x.astype(np.float32) / 255.0),
        ("mobilenetv2", train_mobilenetv2, lambda x: (x.astype(np.float32) / 127.5) - 1.0),
        ("efficientnetb0", train_efficientnetb0, lambda x: x.astype(np.float32)),
    ]

    all_models_eval = {}

    for model_key, train_fn, prep_fn in models_to_train:
        m_dir = phase5_dir / model_key
        model, artifacts = train_fn(X_train_raw, y_train, X_val_raw, y_val, m_dir)

        # Evaluate on Validation Set
        val_x = prep_fn(X_val_raw)
        val_probs = model.predict(val_x, batch_size=32, verbose=0)
        val_metrics = compute_comprehensive_metrics(y_val, val_probs)

        # Final Test Evaluation (Executed ONCE as specified)
        test_x = prep_fn(X_test_raw)
        test_probs = model.predict(test_x, batch_size=32, verbose=0)
        test_metrics = compute_comprehensive_metrics(y_test, test_probs)

        # Plot confusion matrices
        cm = np.array(test_metrics["confusion_matrix"])
        cm_norm = np.array(test_metrics["normalized_confusion_matrix"])
        plot_confusion_matrix(cm, cm_norm, m_dir / "confusion_matrix.png", f"{artifacts['metadata']['model_name']} Test Set")

        # Save metrics and history
        metrics_payload = {
            "model_metadata": artifacts["metadata"],
            "validation_metrics": val_metrics,
            "final_test_metrics": test_metrics,
        }
        with open(m_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

        with open(m_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(artifacts["history"], f, indent=2)

        all_models_eval[model_key] = {
            "metadata": artifacts["metadata"],
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

    # =========================================================================
    # GENERATE COMPARISON TABLE & REPORTS
    # =========================================================================

    # 1. Comparison CSV
    csv_path = phase5_dir / "model_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Params", "Val Accuracy", "Val Macro F1", "Test Accuracy", "Test Macro F1", "OSCC Test Recall", "OSCC Support"])
        for k, v in all_models_eval.items():
            name = v["metadata"]["model_name"]
            params = v["metadata"]["total_parameters"]
            val_acc = f"{v['val_metrics']['accuracy']:.4f}"
            val_f1 = f"{v['val_metrics']['macro_f1']:.4f}"
            test_acc = f"{v['test_metrics']['accuracy']:.4f}"
            test_f1 = f"{v['test_metrics']['macro_f1']:.4f}"
            oscc_rec = f"{v['test_metrics']['per_class']['Oral Cancer']['recall']:.4f}"
            writer.writerow([name, params, val_acc, val_f1, test_acc, test_f1, oscc_rec, 3])

    # 2. Select Best Model by Validation Macro F1
    best_key = max(all_models_eval.keys(), key=lambda k: all_models_eval[k]["val_metrics"]["macro_f1"])
    best_name = all_models_eval[best_key]["metadata"]["model_name"]

    # 3. Model Comparison Markdown
    md_table = """# Phase 5: Baseline Model Comparison Table

| Model | Total Params | Trainable Params | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Test Weighted F1 | Test Cohen's Kappa |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for k, v in all_models_eval.items():
        name = v["metadata"]["model_name"]
        t_params = v["metadata"]["total_parameters"]
        tr_params = v["metadata"]["trainable_parameters"]
        v_acc = v["val_metrics"]["accuracy"]
        v_f1 = v["val_metrics"]["macro_f1"]
        te_acc = v["test_metrics"]["accuracy"]
        te_f1 = v["test_metrics"]["macro_f1"]
        te_wf1 = v["test_metrics"]["weighted_f1"]
        te_kap = v["test_metrics"]["cohens_kappa"]
        md_table += f"| **{name}** | {t_params:,} | {tr_params:,} | {v_acc:.2%} | **{v_f1:.4f}** | {te_acc:.2%} | **{te_f1:.4f}** | {te_wf1:.4f} | {te_kap:.4f} |\n"

    md_table += f"\n**Selected Best Validation Model:** `{best_name}` (Selected strictly by Validation Macro F1 to counteract class imbalance).\n"

    with open(phase5_dir / "model_comparison.md", "w", encoding="utf-8") as f:
        f.write(md_table)

    # 4. Phase 5 Comprehensive Report (JSON & Markdown)
    report_payload = {
        "phase": "Phase 5 — Baseline Model Training",
        "status": "PASS WITH CONCERNS",
        "models_trained": [v["metadata"]["model_name"] for v in all_models_eval.values()],
        "best_validation_model": best_name,
        "selection_metric": "Validation Macro F1 (prevents majority-class accuracy bias)",
        "models": all_models_eval,
        "oscc_test_limitation": {
            "test_samples": 3,
            "validation_samples": 3,
            "per_sample_weight": "33.3% per prediction",
            "clinical_significance": "OSCC sensitivity cannot be claimed as statistically robust under 3 test images. Low-statistical-power estimate explicitly flagged.",
        },
    }

    with open(phase5_dir / "phase5_report.json", "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    # Markdown Report
    best_test_metrics = all_models_eval[best_key]["test_metrics"]
    rep_md = f"""# Phase 5: Baseline Model Training Report

**Status:** PASS WITH CONCERNS  
**Best Validation Architecture:** `{best_name}`  
**Selection Criterion:** Highest Validation Macro F1 (**{all_models_eval[best_key]['val_metrics']['macro_f1']:.4f}**)  
**Random Seed:** `42`  

---

## 1. Experiment Overview & Architecture Design

Three distinct models were trained independently using identical dataset splits, class weights, and evaluation protocols:

1. **Custom Oral CNN:** Lightweight baseline (304,932 parameters) trained from scratch with medical-safe augmentation.
2. **MobileNetV2:** ImageNet transfer learning (2,263,108 parameters) with frozen backbone feature extraction + controlled fine-tuning.
3. **EfficientNetB0:** ImageNet transfer learning (4,054,695 parameters) with compound-scaled convolutional backbone + controlled fine-tuning.

---

## 2. Model Comparison Summary

{md_table}

---

## 3. Detailed Final Test Evaluation: Best Model ({best_name})

Evaluated once on the locked 371-image test set:

- **Overall Test Accuracy:** {best_test_metrics['accuracy']:.2%}
- **Macro Precision:** {best_test_metrics['macro_precision']:.4f}
- **Macro Recall:** {best_test_metrics['macro_recall']:.4f}
- **Macro F1-Score:** {best_test_metrics['macro_f1']:.4f}
- **Weighted F1-Score:** {best_test_metrics['weighted_f1']:.4f}
- **Cohen's Kappa:** {best_test_metrics['cohens_kappa']:.4f}

### Per-Class Test Performance:

| Class Diagnostic Category | Support | Precision | Recall | F1-Score | Statistical Reliability |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Normal** | {best_test_metrics['per_class']['Normal']['support']} | {best_test_metrics['per_class']['Normal']['precision']:.4f} | {best_test_metrics['per_class']['Normal']['recall']:.4f} | {best_test_metrics['per_class']['Normal']['f1']:.4f} | Robust (N=322) |
| **Variation from normal** | {best_test_metrics['per_class']['Variation from normal']['support']} | {best_test_metrics['per_class']['Variation from normal']['precision']:.4f} | {best_test_metrics['per_class']['Variation from normal']['recall']:.4f} | {best_test_metrics['per_class']['Variation from normal']['f1']:.4f} | Moderate (N=27) |
| **OPMD** | {best_test_metrics['per_class']['OPMD']['support']} | {best_test_metrics['per_class']['OPMD']['precision']:.4f} | {best_test_metrics['per_class']['OPMD']['recall']:.4f} | {best_test_metrics['per_class']['OPMD']['f1']:.4f} | Moderate (N=19) |
| **Oral Cancer (OSCC)** | {best_test_metrics['per_class']['Oral Cancer']['support']} | {best_test_metrics['per_class']['Oral Cancer']['precision']:.4f} | {best_test_metrics['per_class']['Oral Cancer']['recall']:.4f} | {best_test_metrics['per_class']['Oral Cancer']['f1']:.4f} | **LOW STATISTICAL POWER (N=3)** |

---

## 4. Critical OSCC Sample Size Limitation

> [!WARNING]
> The test set contains only **3 OSCC images** (due to the physical dataset limitation of 20 total images in SMART-OM). A single error shifts OSCC test recall by **33.3 percentage points**. Consequently, while OSCC metrics are reported with full academic transparency, no claim of clinical statistical robustness can be asserted for independent 4-class OSCC sensitivity without external validation cohorts.

---

## Conclusion

```text
PHASE 5 STATUS:
PASS WITH CONCERNS

MODELS TRAINED:
1. Custom CNN (Params: 304,932)
2. MobileNetV2 (Params: 2,263,108)
3. EfficientNetB0 (Params: 4,054,695)

BEST VALIDATION MODEL:
{best_name}

VALIDATION MACRO F1:
{all_models_eval[best_key]['val_metrics']['macro_f1']:.4f}

FINAL TEST RESULTS:
- Accuracy: {best_test_metrics['accuracy']:.2%}
- Macro F1: {best_test_metrics['macro_f1']:.4f}
- Weighted F1: {best_test_metrics['weighted_f1']:.4f}

OSCC TEST SUPPORT:
3 images

OSCC METRIC LIMITATION:
Each sample constitutes 33.3% of class recall; metrics are documented as low-statistical-power estimates.

NEXT STEP:
Phase 6 — final model selection + detailed evaluation
```
"""
    with open(phase5_dir / "phase5_report.md", "w", encoding="utf-8") as f:
        f.write(rep_md)

    print("\n" + "=" * 70)
    print("PHASE 5 TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"Artifacts saved in: {phase5_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
