"""Dataset analysis helper for deep inspection.

This is a second analysis script focused on extracting structural information for the dataset,
including file counts, patient IDs, and class distributions when directory naming provides them.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ANNOTATION_SUFFIXES = {".csv", ".json", ".txt", ".xml", ".yaml", ".yml"}


def collect_files(root: Path) -> Dict[str, List[Path]]:
    """Collect images and annotations under root."""
    images: List[Path] = []
    annotations: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        elif path.suffix.lower() in ANNOTATION_SUFFIXES:
            annotations.append(path)
    return {"images": sorted(images), "annotations": sorted(annotations)}


def infer_label_from_path(path: Path, root: Path) -> str | None:
    """Attempt to infer label from a folder name, but do not assume it is correct without validation."""
    rel_parts = path.relative_to(root).parts
    if len(rel_parts) >= 2:
        return rel_parts[0]
    return None


def analyze(root: Path) -> Dict:
    files = collect_files(root)
    images = files["images"]
    annotations = files["annotations"]

    by_label = defaultdict(list)
    for image in images:
        label = infer_label_from_path(image, root)
        if label is not None:
            by_label[label].append(str(image.relative_to(root)))

    patient_ids = Counter()
    for image in images:
        parts = image.relative_to(root).parts
        for part in parts:
            low = part.lower()
            if "patient" in low or "pid" in low or "subject" in low or "case" in low:
                patient_ids[part] += 1

    result = {
        "root": str(root),
        "num_images": len(images),
        "num_annotations": len(annotations),
        "class_like_counts": {k: len(v) for k, v in sorted(by_label.items())},
        "patient_like_counts": dict(sorted(patient_ids.items())),
        "sample_image_paths": [str(p.relative_to(root)) for p in images[:10]],
        "sample_annotation_paths": [str(p.relative_to(root)) for p in annotations[:20]],
    }
    return result


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
    root = resolve_dataset_root()
    repo_root = root.parent
    while repo_root != repo_root.parent and not (repo_root / "results").exists() and not (repo_root / "src").exists():
        repo_root = repo_root.parent
    report = analyze(root)
    print(json.dumps(report, indent=2))
    output = (repo_root / "results" / "dataset_analysis.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved to {output}")

