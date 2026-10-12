"""Evaluation script for multiclass ROC curves and ROC-AUC metrics.

Explainable Deep Learning Framework for Oral Lesion Classification.
Evaluates:
1. Custom CNN
2. MobileNetV2
3. EfficientNetB0

Classes:
0. Normal
1. Variation from normal
2. Oral Potentially Malignant Disorder (OPMD)
3. Oral Squamous Cell Carcinoma (OSCC)

All evaluations are conducted exclusively on the held-out test split (N=371).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import auc, roc_auc_score, roc_curve

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tensorflow as tf

# Canonical class configuration locked from Phase 4/5/6
CLASS_NAMES = [
    "Normal",
    "Variation from normal",
    "OPMD",
    "OSCC (Oral Cancer)",
]
SOURCE_FOLDERS = [
    "01. Normal",
    "02. Variation from normal",
    "03. OPMD",
    "04. Oral Cancer",
]
CLASS_MAP = {folder: idx for idx, folder in enumerate(SOURCE_FOLDERS)}

# Publication styling colors (accessible, distinguishable palette)
CLASS_COLORS = [
    "#1f77b4",  # Blue (Normal)
    "#ff7f0e",  # Orange (Variation)
    "#2ca02c",  # Green (OPMD)
    "#d62728",  # Red (OSCC)
]


def load_test_dataset(test_split_path: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Load test images and ground-truth labels from the locked test split."""
    if not test_split_path.exists():
        raise FileNotFoundError(f"Test split file not found: {test_split_path}")

    with open(test_split_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    images: List[np.ndarray] = []
    labels: List[int] = []

    for r in records:
        img_path = Path(r["image_path"])
        if not img_path.exists():
            raise FileNotFoundError(f"Test image not found: {img_path}")

        with Image.open(img_path) as img:
            rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            images.append(np.array(rgb, dtype=np.uint8))

        folder = r["source_folder"]
        if folder not in CLASS_MAP:
            raise ValueError(f"Unrecognized class source folder: {folder}")
        labels.append(CLASS_MAP[folder])

    X_raw = np.stack(images, axis=0)
    y_true = np.array(labels, dtype=np.int32)
    return X_raw, y_true, records


def compute_hanley_mcneil_ci(auc_val: float, n_pos: int, n_neg: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Compute Hanley & McNeil analytical Standard Error and asymptotic 95% CI for ROC-AUC.
    
    Q1: P(two randomly chosen positive samples both ranked higher than a negative)
    Q2: P(one positive sample ranked higher than two randomly chosen negatives)
    """
    if n_pos <= 0 or n_neg <= 0:
        return 0.0, 0.0, 0.0
    if auc_val <= 0.0 or auc_val >= 1.0:
        # Boundary cases
        se = np.sqrt((auc_val * (1.0 - auc_val)) / (n_pos * n_neg)) if auc_val not in (0.0, 1.0) else 0.0
        ci_lower = max(0.0, auc_val - 1.96 * se)
        ci_upper = min(1.0, auc_val + 1.96 * se)
        return float(se), float(ci_lower), float(ci_upper)

    q1 = auc_val / (2.0 - auc_val)
    q2 = (2.0 * auc_val ** 2) / (1.0 + auc_val)
    var = (
        auc_val * (1.0 - auc_val)
        + (n_pos - 1) * (q1 - auc_val ** 2)
        + (n_neg - 1) * (q2 - auc_val ** 2)
    ) / (n_pos * n_neg)

    se = float(np.sqrt(max(0.0, var)))
    ci_lower = float(max(0.0, auc_val - 1.96 * se))
    ci_upper = float(min(1.0, auc_val + 1.96 * se))
    return se, ci_lower, ci_upper


def evaluate_model_roc(
    model_name: str,
    y_true: np.ndarray,
    y_probs: np.ndarray,
) -> Dict[str, Any]:
    """Compute one-vs-rest ROC curves and summary AUC statistics for a model."""
    num_classes = len(CLASS_NAMES)
    n_samples = len(y_true)
    y_onehot = np.eye(num_classes)[y_true]

    per_class_results = {}
    valid_aucs = []
    class_weights = []

    for c in range(num_classes):
        y_binary = y_onehot[:, c]
        n_pos = int(np.sum(y_binary == 1))
        n_neg = int(np.sum(y_binary == 0))

        # Explicit handling for unsupported classes (e.g. no positive or no negative samples)
        if n_pos == 0 or n_neg == 0:
            per_class_results[CLASS_NAMES[c]] = {
                "class_index": c,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "supported": False,
                "error": "Class lacks positive or negative instances in test set.",
                "auc": None,
                "se": None,
                "ci_95": None,
                "fpr": [],
                "tpr": [],
                "thresholds": [],
            }
            continue

        fpr, tpr, thresholds = roc_curve(y_binary, y_probs[:, c])
        c_auc = float(auc(fpr, tpr))
        se, ci_lower, ci_upper = compute_hanley_mcneil_ci(c_auc, n_pos, n_neg)

        per_class_results[CLASS_NAMES[c]] = {
            "class_index": c,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "supported": True,
            "auc": c_auc,
            "se": se,
            "ci_95": [ci_lower, ci_upper],
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "is_low_power": (c == 3 and n_pos <= 5),
        }
        valid_aucs.append(c_auc)
        class_weights.append(n_pos)

    # Aggregates
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else None
    weighted_auc = float(np.average(valid_aucs, weights=class_weights)) if valid_aucs else None

    # Micro-average ROC
    micro_fpr, micro_tpr, _ = roc_curve(y_onehot.ravel(), y_probs.ravel())
    micro_auc = float(auc(micro_fpr, micro_tpr))

    return {
        "model_name": model_name,
        "n_samples": n_samples,
        "num_classes": num_classes,
        "macro_auc": macro_auc,
        "weighted_auc": weighted_auc,
        "micro_auc": micro_auc,
        "micro_fpr": micro_fpr.tolist(),
        "micro_tpr": micro_tpr.tolist(),
        "per_class": per_class_results,
    }


def plot_single_model_roc(
    eval_res: Dict[str, Any],
    save_path: Path,
    dpi: int = 300,
) -> None:
    """Generate and save publication-quality multiclass OvR ROC curve for a single model."""
    fig, ax = plt.subplots(figsize=(8.0, 7.0), dpi=dpi)

    model_name = eval_res["model_name"]
    per_class = eval_res["per_class"]

    # Micro-average
    micro_fpr = np.array(eval_res["micro_fpr"])
    micro_tpr = np.array(eval_res["micro_tpr"])
    micro_auc = eval_res["micro_auc"]
    ax.plot(
        micro_fpr,
        micro_tpr,
        label=f"Micro-average (AUC = {micro_auc:.3f})",
        color="deeppink",
        linestyle=":",
        linewidth=2.2,
    )

    # Macro-average
    macro_auc = eval_res["macro_auc"]
    weighted_auc = eval_res["weighted_auc"]

    # Plot each class OvR curve
    for idx, cname in enumerate(CLASS_NAMES):
        cdata = per_class[cname]
        if not cdata["supported"]:
            continue

        fpr = np.array(cdata["fpr"])
        tpr = np.array(cdata["tpr"])
        c_auc = cdata["auc"]
        n_pos = cdata["n_pos"]
        n_neg = cdata["n_neg"]
        ci = cdata["ci_95"]

        label = f"{cname} (AUC = {c_auc:.3f}, 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}] | Pos={n_pos}, Neg={n_neg})"
        if cdata.get("is_low_power"):
            label += " *"

        ax.plot(
            fpr,
            tpr,
            color=CLASS_COLORS[idx],
            linewidth=2.0,
            label=label,
        )

    # Reference random line
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1.5, alpha=0.7, label="Chance level (AUC = 0.500)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"One-vs-Rest ROC Curves — {model_name}\nMacro AUC: {macro_auc:.3f} | Weighted AUC: {weighted_auc:.3f} (Held-out Test N=371)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92, edgecolor="#cccccc")

    # Callout note for OSCC limitation
    ax.text(
        0.03,
        0.05,
        "* OSCC has only N=3 positive test cases; AUC estimate has wide confidence intervals and high variance.",
        fontsize=8.0,
        fontstyle="italic",
        color="#771111",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="#ffeeba", alpha=0.9),
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_comparison_mobilenet_vs_efficientnet(
    mob_res: Dict[str, Any],
    eff_res: Dict[str, Any],
    save_path: Path,
    dpi: int = 300,
) -> None:
    """Generate publication comparison ROC plot for MobileNetV2 vs EfficientNetB0."""
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.8), dpi=dpi, sharey=True)

    models_data = [
        ("MobileNetV2 (ImageNet Pretrained)", mob_res, axes[0]),
        ("EfficientNetB0 (Selected Best Validation)", eff_res, axes[1]),
    ]

    for model_label, res, ax in models_data:
        per_class = res["per_class"]
        macro_auc = res["macro_auc"]
        weighted_auc = res["weighted_auc"]
        micro_auc = res["micro_auc"]

        # Micro-average
        ax.plot(
            np.array(res["micro_fpr"]),
            np.array(res["micro_tpr"]),
            label=f"Micro-avg (AUC = {micro_auc:.3f})",
            color="deeppink",
            linestyle=":",
            linewidth=2.0,
        )

        for idx, cname in enumerate(CLASS_NAMES):
            cdata = per_class[cname]
            if not cdata["supported"]:
                continue

            fpr = np.array(cdata["fpr"])
            tpr = np.array(cdata["tpr"])
            c_auc = cdata["auc"]
            n_pos = cdata["n_pos"]
            n_neg = cdata["n_neg"]
            ci = cdata["ci_95"]

            lbl = f"{cname}\nAUC = {c_auc:.3f} [{ci[0]:.2f}-{ci[1]:.2f}] (P={n_pos}, N={n_neg})"
            if cdata.get("is_low_power"):
                lbl += " *"

            ax.plot(
                fpr,
                tpr,
                color=CLASS_COLORS[idx],
                linewidth=2.0,
                label=lbl,
            )

        ax.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1.3, alpha=0.6, label="Chance (AUC = 0.500)")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10.5, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10.5, fontweight="bold")
        ax.set_title(
            f"{model_label}\nMacro AUC: {macro_auc:.3f} | Weighted AUC: {weighted_auc:.3f}",
            fontsize=11.5,
            fontweight="bold",
            pad=10,
        )
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=8.0, framealpha=0.9, edgecolor="#cccccc")

    # Bottom limitation text across figure
    fig.text(
        0.5,
        0.015,
        "* Note: OSCC positive test support is N=3 images across 3 patients; true-positive steps occur in discrete 33.3% increments.",
        ha="center",
        fontsize=8.5,
        fontstyle="italic",
        color="#555555",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_all_models_per_class_comparison(
    eval_dict: Dict[str, Dict[str, Any]],
    save_path: Path,
    dpi: int = 300,
) -> None:
    """Generate 2x2 grid comparing all 3 models across each individual class."""
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.5), dpi=dpi)
    axes = axes.flatten()

    model_styles = {
        "Custom CNN": {"color": "#7f7f7f", "linestyle": "-.", "lw": 1.8},
        "MobileNetV2": {"color": "#1f77b4", "linestyle": "--", "lw": 2.2},
        "EfficientNetB0": {"color": "#2ca02c", "linestyle": "-", "lw": 2.4},
    }

    for c_idx, cname in enumerate(CLASS_NAMES):
        ax = axes[c_idx]

        for m_name, res in eval_dict.items():
            cdata = res["per_class"][cname]
            if not cdata["supported"]:
                continue

            fpr = np.array(cdata["fpr"])
            tpr = np.array(cdata["tpr"])
            c_auc = cdata["auc"]
            style = model_styles[m_name]

            ax.plot(
                fpr,
                tpr,
                label=f"{m_name} (AUC = {c_auc:.3f})",
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["lw"],
            )

        n_pos = eval_dict["EfficientNetB0"]["per_class"][cname]["n_pos"]
        n_neg = eval_dict["EfficientNetB0"]["per_class"][cname]["n_neg"]

        ax.plot([0.0, 1.0], [0.0, 1.0], "k:", linewidth=1.2, alpha=0.5, label="Chance (0.500)")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=9.5, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=9.5, fontweight="bold")

        title_suffix = " [CAUTION: N=3, High Variance]" if c_idx == 3 else ""
        ax.set_title(
            f"Class {c_idx}: {cname}{title_suffix}\n(Positive: {n_pos} | Negative: {n_neg})",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    plt.suptitle(
        "Multiclass Model Comparison Across All 4 Oral Lesion Classes (Held-out Test N=371)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {save_path}")


def save_numerical_results_csv(
    eval_dict: Dict[str, Dict[str, Any]],
    csv_path: Path,
) -> None:
    """Save all per-class, macro, and weighted AUC results to a clean CSV."""
    rows = []

    for m_name, res in eval_dict.items():
        # Per-class rows
        for cname in CLASS_NAMES:
            cdata = res["per_class"][cname]
            rows.append({
                "Model": m_name,
                "Evaluation_Type": "Per-Class OvR",
                "Class_Index": cdata["class_index"],
                "Class_Name": cname,
                "Positive_Examples_N": cdata["n_pos"],
                "Negative_Examples_N": cdata["n_neg"],
                "ROC_AUC": round(cdata["auc"], 4) if cdata["auc"] is not None else "N/A",
                "Std_Error_SE": round(cdata["se"], 4) if cdata["se"] is not None else "N/A",
                "CI_95_Lower": round(cdata["ci_95"][0], 4) if cdata["ci_95"] else "N/A",
                "CI_95_Upper": round(cdata["ci_95"][1], 4) if cdata["ci_95"] else "N/A",
                "Low_Statistical_Power_Flag": "YES (N=3)" if cdata.get("is_low_power") else "NO",
            })

        # Summary rows
        rows.append({
            "Model": m_name,
            "Evaluation_Type": "Aggregate",
            "Class_Index": "ALL",
            "Class_Name": "Macro-Average",
            "Positive_Examples_N": res["n_samples"],
            "Negative_Examples_N": 0,
            "ROC_AUC": round(res["macro_auc"], 4) if res["macro_auc"] is not None else "N/A",
            "Std_Error_SE": "N/A",
            "CI_95_Lower": "N/A",
            "CI_95_Upper": "N/A",
            "Low_Statistical_Power_Flag": "NO",
        })

        rows.append({
            "Model": m_name,
            "Evaluation_Type": "Aggregate",
            "Class_Index": "ALL",
            "Class_Name": "Weighted-Average",
            "Positive_Examples_N": res["n_samples"],
            "Negative_Examples_N": 0,
            "ROC_AUC": round(res["weighted_auc"], 4) if res["weighted_auc"] is not None else "N/A",
            "Std_Error_SE": "N/A",
            "CI_95_Lower": "N/A",
            "CI_95_Upper": "N/A",
            "Low_Statistical_Power_Flag": "NO",
        })

        rows.append({
            "Model": m_name,
            "Evaluation_Type": "Aggregate",
            "Class_Index": "ALL",
            "Class_Name": "Micro-Average",
            "Positive_Examples_N": res["n_samples"],
            "Negative_Examples_N": res["n_samples"] * 3,
            "ROC_AUC": round(res["micro_auc"], 4) if res["micro_auc"] is not None else "N/A",
            "Std_Error_SE": "N/A",
            "CI_95_Lower": "N/A",
            "CI_95_Upper": "N/A",
            "Low_Statistical_Power_Flag": "NO",
        })

    fieldnames = [
        "Model",
        "Evaluation_Type",
        "Class_Index",
        "Class_Name",
        "Positive_Examples_N",
        "Negative_Examples_N",
        "ROC_AUC",
        "Std_Error_SE",
        "CI_95_Lower",
        "CI_95_Upper",
        "Low_Statistical_Power_Flag",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {csv_path}")


def save_test_prediction_probabilities_csv(
    test_records: List[Dict[str, Any]],
    y_true: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    csv_path: Path,
) -> None:
    """Save full prediction probability distributions across all test samples for complete auditability."""
    rows = []
    for idx, r in enumerate(test_records):
        row = {
            "index": idx,
            "image_path": r["image_path"],
            "patient_id": r.get("patient_id", "Unknown"),
            "true_class_index": int(y_true[idx]),
            "true_class_name": CLASS_NAMES[int(y_true[idx])],
        }
        for m_name, probs in probs_dict.items():
            prefix = m_name.lower().replace(" ", "_")
            for c_idx, cname in enumerate(CLASS_NAMES):
                col_name = f"{prefix}_prob_{cname.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
                row[col_name] = float(probs[idx, c_idx])
            row[f"{prefix}_pred_class"] = CLASS_NAMES[int(np.argmax(probs[idx]))]
            row[f"{prefix}_confidence"] = float(np.max(probs[idx]))

        rows.append(row)

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved: {csv_path}")


def run_roc_auc_pipeline(
    splits_path: Optional[Path] = None,
    phase5_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    dpi: int = 300,
) -> Dict[str, Any]:
    """Execute the end-to-end ROC evaluation pipeline."""
    splits_path = splits_path or (REPO_ROOT / "results" / "splits" / "test_split.json")
    phase5_dir = phase5_dir or (REPO_ROOT / "results" / "phase5")
    output_dir = output_dir or (REPO_ROOT / "results" / "roc_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("MULTICLASS ROC & AUC EVALUATION PIPELINE")
    print("=" * 75)
    print(f"Splits path: {splits_path}")
    print(f"Phase 5 dir: {phase5_dir}")
    print(f"Output dir:  {output_dir}")

    # 1. Load test data
    print("\n[Step 1/5] Loading held-out test dataset...")
    X_raw, y_true, records = load_test_dataset(splits_path)
    print(f"Successfully loaded {len(X_raw)} images across {len(set(r['patient_id'] for r in records))} patients.")
    print("Test ground-truth distribution:")
    for c_idx, cname in enumerate(CLASS_NAMES):
        count = int(np.sum(y_true == c_idx))
        print(f"  - Class {c_idx} ({cname}): {count} images ({count / len(y_true):.2%})")

    # 2. Check model checkpoints
    print("\n[Step 2/5] Loading model checkpoints and verifying preprocessing...")
    model_specs = [
        (
            "Custom CNN",
            phase5_dir / "custom_cnn" / "best_model.keras",
            lambda x: x.astype(np.float32) / 255.0,
            "custom_cnn",
        ),
        (
            "MobileNetV2",
            phase5_dir / "mobilenetv2" / "best_model.keras",
            lambda x: (x.astype(np.float32) / 127.5) - 1.0,
            "mobilenetv2",
        ),
        (
            "EfficientNetB0",
            phase5_dir / "efficientnetb0" / "best_model.keras",
            lambda x: x.astype(np.float32),
            "efficientnetb0",
        ),
    ]

    for name, p, _, _ in model_specs:
        if not p.exists():
            raise FileNotFoundError(f"Required model checkpoint does not exist: {p}")
        print(f"  Verified checkpoint: {name} -> {p}")

    # 3. Predict probabilities on test split
    print("\n[Step 3/5] Generating prediction probability matrices...")
    eval_dict: Dict[str, Dict[str, Any]] = {}
    probs_dict: Dict[str, np.ndarray] = {}

    for name, model_path, prep_fn, key in model_specs:
        print(f"  Evaluating {name}...")
        model = tf.keras.models.load_model(model_path)
        x_input = prep_fn(X_raw)
        probs = model.predict(x_input, batch_size=32, verbose=0)
        probs_dict[name] = probs

        res = evaluate_model_roc(name, y_true, probs)
        eval_dict[name] = res

        print(f"    Macro AUC:    {res['macro_auc']:.4f}")
        print(f"    Weighted AUC: {res['weighted_auc']:.4f}")
        print(f"    Micro AUC:    {res['micro_auc']:.4f}")
        for cname in CLASS_NAMES:
            cdata = res["per_class"][cname]
            ci = cdata["ci_95"]
            print(f"      {cname:25s} AUC: {cdata['auc']:.4f} [95% CI: {ci[0]:.3f} - {ci[1]:.3f}]")

    # 4. Generate Publication-Quality Plots
    print("\n[Step 4/5] Generating publication-quality ROC plots (300 DPI)...")

    # A. Individual model ROC curves
    for name, _, _, key in model_specs:
        plot_single_model_roc(eval_dict[name], output_dir / f"{key}_roc.png", dpi=dpi)

    # B. MobileNetV2 vs EfficientNetB0 comparison plot
    plot_comparison_mobilenet_vs_efficientnet(
        eval_dict["MobileNetV2"],
        eval_dict["EfficientNetB0"],
        output_dir / "mobilenetv2_vs_efficientnetb0_roc.png",
        dpi=dpi,
    )

    # C. All 3 models 2x2 comparison plot
    plot_all_models_per_class_comparison(
        eval_dict,
        output_dir / "all_models_comparison_roc.png",
        dpi=dpi,
    )

    # 5. Save Numerical Results and Predictions
    print("\n[Step 5/5] Exporting numerical results and prediction records...")
    csv_path = output_dir / "roc_auc_metrics.csv"
    save_numerical_results_csv(eval_dict, csv_path)

    probs_csv_path = output_dir / "test_prediction_probabilities.csv"
    save_test_prediction_probabilities_csv(records, y_true, probs_dict, probs_csv_path)

    # Save summary JSON
    summary_json_path = output_dir / "roc_auc_summary.json"
    clean_json_data = {}
    for m_name, res in eval_dict.items():
        clean_json_data[m_name] = {
            "macro_auc": res["macro_auc"],
            "weighted_auc": res["weighted_auc"],
            "micro_auc": res["micro_auc"],
            "per_class": {
                cname: {
                    "auc": cdata["auc"],
                    "se": cdata["se"],
                    "ci_95": cdata["ci_95"],
                    "n_pos": cdata["n_pos"],
                    "n_neg": cdata["n_neg"],
                    "is_low_power": cdata.get("is_low_power", False),
                }
                for cname, cdata in res["per_class"].items()
            },
        }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(clean_json_data, f, indent=2)
    print(f"Saved: {summary_json_path}")

    print("\nPipeline execution complete! All artifacts saved to:", output_dir)
    return eval_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiclass ROC & AUC evaluation pipeline.")
    parser.add_argument("--splits-path", type=Path, default=None, help="Path to test_split.json")
    parser.add_argument("--phase5-dir", type=Path, default=None, help="Path to results/phase5")
    parser.add_argument("--output-dir", type=Path, default=None, help="Path to results/roc_analysis")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for plots (default: 300)")
    args = parser.parse_args()

    run_roc_auc_pipeline(
        splits_path=args.splits_path,
        phase5_dir=args.phase5_dir,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )
