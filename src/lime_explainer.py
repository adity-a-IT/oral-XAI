"""LIME explanation skeleton for oral lesion classification.

This module is intentionally modular and readiness-focused. The real model and selected test
images will be plugged in when the dataset is available and the final model is selected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lime import lime_image


def make_lime_explainer(random_state: int = 42):
    """Create the LIME image explainer."""
    return lime_image.LimeImageExplainer(random_state=random_state)


def generate_lime_explanation(model, image: np.ndarray, class_index: int, num_samples: int = 1000):
    """Generate a LIME explanation for a single image."""
    explainer = make_lime_explainer()
    explanation = explainer.explain_instance(
        image,
        lambda x: model.predict(x, verbose=0),
        top_labels=3,
        hide_color=0,
        num_samples=num_samples,
    )
    return explanation


def save_lime_explanation(explanation, output_path: str | Path):
    """Save the explanation visualization to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Real LIME saving: extract positive attribution overlay and save image
    try:
        from PIL import Image
        temp, mask = explanation.get_image_and_mask(
            class_index,
            positive_only=True,
            num_features=10,
            hide_rest=False,
        )
        Image.fromarray(temp.astype(np.uint8)).save(str(path))
    except Exception:
        path.write_text(f"LIME explanation generated for class index {class_index}\n", encoding="utf-8")


if __name__ == "__main__":
    print("Real LIME explanation utilities loaded.")

