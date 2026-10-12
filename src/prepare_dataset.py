"""Dataset preparation utilities for the oral lesion project.

This module is intentionally kept dataset-agnostic and safe. It is designed to be used
only after the actual AI4OralHealth dataset has been inspected and validated.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def ensure_dataset_present(dataset_root: str | Path) -> Path:
    """Validate that the dataset root exists before using any split logic."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not any(root.rglob("*")):
        raise FileNotFoundError(f"Dataset root is empty: {root}")
    return root


def load_dataset_summary(summary_path: str | Path) -> Dict[str, Any]:
    """Load a JSON summary generated during dataset inspection."""
    path = Path(summary_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset summary not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_patient_mapping(paths: Iterable[str]) -> Dict[str, List[str]]:
    """Map patient identifiers to image paths.

    This is intentionally conservative and should be adapted to the real dataset metadata
    once patient IDs are confirmed from CSV / JSON / folder naming.
    """
    mapping: Dict[str, List[str]] = {}
    for path in paths:
        patient_id = Path(path).stem.split("_")[0]
        mapping.setdefault(patient_id, []).append(path)
    return mapping


import random
import re


def extract_patient_id(path: str | Path) -> str:
    """Extract consistent patient/case identifier from filename."""
    stem = Path(path).stem.strip()
    # Check for SMITA pattern
    m = re.search(r"(SMITA\d+)", stem, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Check for Cancer 'Ca X' or 'CaX'
    m = re.search(r"^(Ca\s*\d+)", stem, re.IGNORECASE)
    if m:
        return m.group(1).upper().replace(" ", "")
    # Check numeric prefix before ' - '
    m = re.match(r"^(\d+)\s*-", stem)
    if m:
        return f"NUM_{m.group(1)}"
    # Check 4-digit case number e.g. 6026, 6765
    m = re.match(r"^(\d{4})", stem)
    if m:
        return f"CASE_{m.group(1)}"
    return f"OTHER_{stem.split()[0]}"


def patient_level_stratified_split(
    records: List[Dict[str, Any]],
    patient_key: str = "patient_id",
    class_key: str = "source_folder",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Create patient-aware, class-stratified train/val/test splits with 0 patient leakage.

    Uses a deterministic greedy multi-label patient allocation strategy:
    1. Groups all image records by patient ID.
    2. Builds a multi-class presence and count profile for each patient.
    3. Orders classes from rarest (e.g. Oral Cancer) to most frequent (Normal).
    4. Allocates patients to partitions (Train, Val, Test) to match target class
       ratios while ensuring every patient's images remain strictly in one partition.
    5. Resolves ties using a deterministic pseudo-random sequence seeded with `seed`.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train/validation/test ratios must sum to 1.0")

    if not records:
        return [], [], [], {}

    patient_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        patient_records[r[patient_key]].append(r)

    all_pids = sorted(patient_records.keys())

    # Calculate global class distribution and sort classes by rarity
    global_totals = Counter(r[class_key] for r in records)
    classes_by_rarity = sorted(global_totals.keys(), key=lambda c: global_totals[c])

    # Build patient-level class count vector
    patient_class_counts: Dict[str, Dict[str, int]] = {}
    for pid in all_pids:
        counts = Counter(r[class_key] for r in patient_records[pid])
        patient_class_counts[pid] = {c: counts.get(c, 0) for c in classes_by_rarity}

    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    targets = {s: {c: ratios[s] * global_totals[c] for c in classes_by_rarity} for s in ratios}
    current = {s: {c: 0 for c in classes_by_rarity} for s in ratios}
    patient_alloc: Dict[str, str] = {}

    rng = random.Random(seed)
    patient_tie_breaker = {pid: rng.random() for pid in all_pids}
    unassigned = set(all_pids)

    # 1. Greedy allocation starting with the rarest classes
    for c in classes_by_rarity:
        c_pids = [p for p in unassigned if patient_class_counts[p][c] > 0]
        # Sort by count of c desc, total images desc, then tie-breaker
        c_pids.sort(key=lambda p: (
            -patient_class_counts[p][c],
            -sum(patient_class_counts[p].values()),
            patient_tie_breaker[p]
        ))

        for pid in c_pids:
            def split_key(s: str):
                c_ratio = current[s][c] / max(targets[s][c], 1e-6)
                overall_ratio = sum(current[s][k] for k in classes_by_rarity) / (ratios[s] * len(records))
                deficit = targets[s][c] - current[s][c]
                return (c_ratio, overall_ratio, -deficit)

            best_s = min(ratios.keys(), key=split_key)
            patient_alloc[pid] = best_s
            unassigned.remove(pid)
            for k in classes_by_rarity:
                current[best_s][k] += patient_class_counts[pid][k]

    # 2. Remaining unassigned patients (predominantly single/majority class)
    rem_pids = sorted(unassigned, key=lambda p: (-sum(patient_class_counts[p].values()), patient_tie_breaker[p]))
    majority_class = classes_by_rarity[-1]
    for pid in rem_pids:
        def rem_key(s: str):
            overall_ratio = sum(current[s][k] for k in classes_by_rarity) / (ratios[s] * len(records))
            patient_ratio = sum(1 for p, sp in patient_alloc.items() if sp == s) / (ratios[s] * len(all_pids))
            norm_deficit = targets[s][majority_class] - current[s][majority_class]
            return (overall_ratio, patient_ratio, -norm_deficit)

        best_s = min(ratios.keys(), key=rem_key)
        patient_alloc[pid] = best_s
        for k in classes_by_rarity:
            current[best_s][k] += patient_class_counts[pid][k]

    train_pids = sorted([p for p, sp in patient_alloc.items() if sp == "train"])
    val_pids = sorted([p for p, sp in patient_alloc.items() if sp == "val"])
    test_pids = sorted([p for p, sp in patient_alloc.items() if sp == "test"])

    # Strict leakage validation
    assert len(set(train_pids) & set(val_pids)) == 0, "Train-Val Leakage!"
    assert len(set(train_pids) & set(test_pids)) == 0, "Train-Test Leakage!"
    assert len(set(val_pids) & set(test_pids)) == 0, "Val-Test Leakage!"

    train_recs = [r for pid in train_pids for r in sorted(patient_records[pid], key=lambda x: x["image_path"])]
    val_recs = [r for pid in val_pids for r in sorted(patient_records[pid], key=lambda x: x["image_path"])]
    test_recs = [r for pid in test_pids for r in sorted(patient_records[pid], key=lambda x: x["image_path"])]

    stats = {
        "total_images": len(records),
        "total_patients": len(all_pids),
        "seed": seed,
        "classes": classes_by_rarity,
        "splits": {
            "train": {"images": len(train_recs), "patients": len(train_pids), "classes": dict(current["train"])},
            "val": {"images": len(val_recs), "patients": len(val_pids), "classes": dict(current["val"])},
            "test": {"images": len(test_recs), "patients": len(test_pids), "classes": dict(current["test"])},
        }
    }
    return train_recs, val_recs, test_recs, stats


def patient_level_split(
    records: List[Dict[str, Any]],
    patient_key: str = "patient_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Backward-compatible entry point delegating to stratified patient splitting."""
    train_r, val_r, test_r, _ = patient_level_stratified_split(
        records=records,
        patient_key=patient_key,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    return train_r, val_r, test_r


def build_smart_om_manifest(dataset_root: Path) -> List[Dict[str, Any]]:
    """Index raw unannotated photographic images with class labels and patient IDs."""
    class_map = {
        "01. Normal": ("Healthy", 0),
        "02. Variation from normal": ("Healthy_Variation", 0),
        "03. OPMD": ("OPMD", 1),
        "04. Oral Cancer": ("OSCC", 2),
    }
    records = []
    for folder_name, (label_name, label_idx) in class_map.items():
        raw_dir = dataset_root / folder_name / "01. Unannotated"
        if not raw_dir.exists():
            continue
        for img in sorted(raw_dir.rglob("*")):
            if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append({
                    "image_path": str(img.resolve()),
                    "relative_path": str(img.relative_to(dataset_root)),
                    "patient_id": extract_patient_id(img),
                    "class_name": label_name,
                    "label": label_idx,
                    "source_folder": folder_name,
                })
    return records


if __name__ == "__main__":
    candidates = [
        Path("dataset/oral-lesion-dataset-org/SMART-OM"),
        Path("dataset"),
        Path("../dataset/oral-lesion-dataset-org/SMART-OM"),
        Path("../dataset"),
    ]
    dataset_root = next((c.resolve() for c in candidates if c.exists() and any(c.iterdir())), None)
    if dataset_root is None:
        raise FileNotFoundError("SMART-OM dataset directory not found.")

    records = build_smart_om_manifest(dataset_root)

    # 1. Run stratified patient-level split (Run 1)
    train_recs, val_recs, test_recs, stats = patient_level_stratified_split(
        records, patient_key="patient_id", class_key="source_folder", seed=42
    )

    # 2. Run again for reproducibility verification (Run 2)
    t2, v2, te2, stats2 = patient_level_stratified_split(
        records, patient_key="patient_id", class_key="source_folder", seed=42
    )
    assert [r["image_path"] for r in train_recs] == [r["image_path"] for r in t2], "Reproducibility failure in Train!"
    assert [r["image_path"] for r in val_recs] == [r["image_path"] for r in v2], "Reproducibility failure in Val!"
    assert [r["image_path"] for r in test_recs] == [r["image_path"] for r in te2], "Reproducibility failure in Test!"

    repo_root = dataset_root.parent
    while repo_root != repo_root.parent and not (repo_root / "results").exists() and not (repo_root / "src").exists():
        repo_root = repo_root.parent

    out_dir = repo_root / "results" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Overwrite split files
    with (out_dir / "train_split.json").open("w", encoding="utf-8") as f:
        json.dump(train_recs, f, indent=2)
    with (out_dir / "val_split.json").open("w", encoding="utf-8") as f:
        json.dump(val_recs, f, indent=2)
    with (out_dir / "test_split.json").open("w", encoding="utf-8") as f:
        json.dump(test_recs, f, indent=2)

    print(f"Total raw images indexed: {len(records)}")
    print(f"Train set: {len(train_recs)} images ({len(set(r['patient_id'] for r in train_recs))} patients)")
    print(f"Val set:   {len(val_recs)} images ({len(set(r['patient_id'] for r in val_recs))} patients)")
    print(f"Test set:  {len(test_recs)} images ({len(set(r['patient_id'] for r in test_recs))} patients)")
    print(f"Splits saved successfully to: {out_dir}")


