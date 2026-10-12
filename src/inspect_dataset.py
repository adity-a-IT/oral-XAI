"""Inspect the AI4OralHealth dataset structure and generate a dataset summary report.

This script is intentionally conservative: it does not assign labels or drop data silently.
It reports the actual directory structure, image files, patient IDs, annotation files, and
missing/corrupt images before any train/test split is attempted.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def find_image_files(root: Path) -> List[Path]:
    """Find common image file extensions under a root directory."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions])


def detect_corrupt_images(paths: List[Path]) -> List[Tuple[Path, str]]:
    """Return image paths that fail to open or are unreadable."""
    bad: List[Tuple[Path, str]] = []
    for path in paths:
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as exc:
            bad.append((path, str(exc)))
    return bad


def summarize_dataset(root: Path, output_json: Path | None = None) -> Dict:
    """Summarize the dataset with counts, class hints, patient hints, and annotation files."""
    root = root.resolve()
    image_files = find_image_files(root)
    annotations = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json", ".txt", ".xml", ".yaml", ".yml"}])

    # Class-like directory names are inferred from folder structure; these are not silently assumed labels.
    class_dirs = defaultdict(list)
    for path in image_files:
        rel_parts = path.relative_to(root).parts
        if len(rel_parts) >= 2:
            class_dirs[rel_parts[0]].append(path)

    patient_hints = defaultdict(list)
    for path in image_files:
        rel_parts = path.relative_to(root).parts
        for part in rel_parts:
            if part.lower().startswith("patient") or part.lower().startswith("pid") or part.lower().startswith("subject"):
                patient_hints[part].append(path)

    summary = {
        "dataset_root": str(root),
        "total_images": len(image_files),
        "image_files": [str(p.relative_to(root)) for p in image_files[:10]],
        "unique_top_level_dirs": sorted(class_dirs.keys()),
        "class_like_dir_counts": {k: len(v) for k, v in sorted(class_dirs.items())},
        "annotation_files": [str(p.relative_to(root)) for p in annotations[:20]],
        "patient_hint_counts": {k: len(v) for k, v in sorted(patient_hints.items())},
        "corrupt_images": [],
    }

    corrupt = detect_corrupt_images(image_files)
    summary["corrupt_images"] = [
        {"path": str(path.relative_to(root)), "error": error} for path, error in corrupt
    ]

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def resolve_dataset_root() -> Path:
    candidates = [
        Path("dataset/oral-lesion-dataset-org/SMART-OM"),
        Path("dataset"),
        Path("../dataset/oral-lesion-dataset-org/SMART-OM"),
        Path("../dataset"),
    ]
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c.resolve()
    return Path("dataset").resolve()


if __name__ == "__main__":
    dataset_root = resolve_dataset_root()
    repo_root = dataset_root.parent
    while repo_root != repo_root.parent and not (repo_root / "results").exists() and not (repo_root / "src").exists():
        repo_root = repo_root.parent
    report_path = (repo_root / "results" / "dataset_summary.json").resolve()
    summary = summarize_dataset(dataset_root, report_path)
    print(json.dumps({
        "dataset_root": summary["dataset_root"],
        "total_images": summary["total_images"],
        "unique_top_level_dirs": summary["unique_top_level_dirs"],
        "class_like_dir_counts": summary["class_like_dir_counts"],
        "annotation_files_count": len(summary["annotation_files"]),
        "corrupt_images_count": len(summary["corrupt_images"]),
    }, indent=2))
    print(f"\nDataset summary saved to: {report_path}")

