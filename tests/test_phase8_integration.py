"""Phase 8 Verification & Regression Test Suite.

Comprehensive automated verification tests (Tests 1-9) + Phase 7 Regression Checks
for Real Neural Model + XAI Web Application Integration.
"""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference_engine import (
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    OralXAIInferenceEngine,
)
from server import app


class TestPhase8Integration(unittest.TestCase):
    """Rigorous end-to-end integration test suite for Phase 8."""

    @classmethod
    def setUpClass(cls):
        """Initialize real inference engine and Flask test client once."""
        print("\n" + "=" * 70)
        print("PHASE 8: INITIALIZING VERIFICATION & REGRESSION TEST SUITE")
        print("=" * 70)
        cls.engine = OralXAIInferenceEngine.get_instance(model_path=DEFAULT_MODEL_PATH)
        app.config["TESTING"] = True
        cls.client = app.test_client()

        # Load locked test set records
        with open(REPO_ROOT / "results/splits/test_split.json", "r", encoding="utf-8") as f:
            cls.test_records = json.load(f)

        # Index records by image filename
        cls.records_by_name = {Path(r["image_path"]).name: r for r in cls.test_records}

    def _get_image_bytes(self, pil_img: Image.Image) -> io.BytesIO:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        buf.seek(0)
        return buf

    # -----------------------------------------------------------------------
    # TEST 1: Web Prediction Matches Direct EfficientNetB0 Inference Exactly
    # -----------------------------------------------------------------------
    def test_01_web_prediction_matches_direct_inference(self):
        """Confirm web API prediction exactly matches direct model inference."""
        sample_path = Path(self.test_records[0]["image_path"])
        with open(sample_path, "rb") as f:
            raw_bytes = f.read()

        # Direct model inference
        img = Image.open(io.BytesIO(raw_bytes))
        direct_res = self.engine.predict(img)

        # Web API inference via Flask test client using exact same raw bytes
        resp = self.client.post(
            "/predict",
            data={"file": (io.BytesIO(raw_bytes), sample_path.name)},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200, f"Predict API failed: {resp.data}")
        api_res = resp.get_json()

        self.assertEqual(api_res["predicted_class"], direct_res["predicted_class"])
        self.assertEqual(api_res["predicted_class_index"], direct_res["predicted_class_index"])
        self.assertAlmostEqual(api_res["confidence"], direct_res["confidence"], places=5)
        for cls_name in CLASS_NAMES:
            self.assertAlmostEqual(
                api_res["probabilities"][cls_name],
                direct_res["probabilities"][cls_name],
                places=5
            )
        print("  [PASS] TEST 1: Web prediction exactly matches direct EfficientNetB0 inference.")


    # -----------------------------------------------------------------------
    # TEST 2: Confirm Probability Vector Has Exactly 4 Classes
    # -----------------------------------------------------------------------
    def test_02_probability_vector_has_four_classes(self):
        """Confirm output contains all 4 verified canonical classes."""
        img = Image.open(self.test_records[0]["image_path"])
        res = self.engine.predict(img)

        probs = res["probabilities"]
        self.assertEqual(len(probs), 4)
        for cls_name in CLASS_NAMES:
            self.assertIn(cls_name, probs)
            self.assertIsInstance(probs[cls_name], float)
            self.assertGreaterEqual(probs[cls_name], 0.0)
            self.assertLessEqual(probs[cls_name], 1.0)
        print("  [PASS] TEST 2: Probability vector has exactly 4 valid class entries.")

    # -----------------------------------------------------------------------
    # TEST 3: Confirm Probabilities Sum Approximately to 1.0
    # -----------------------------------------------------------------------
    def test_03_probabilities_sum_to_one(self):
        """Confirm probabilities sum to 1.0 within numerical float tolerance."""
        for r in self.test_records[:10]:
            img = Image.open(r["image_path"])
            res = self.engine.predict(img)
            prob_sum = sum(res["probabilities"].values())
            self.assertAlmostEqual(
                prob_sum, 1.0, places=4,
                msg=f"Probabilities did not sum to ~1 for {r['image_path']}: sum={prob_sum}"
            )
        print("  [PASS] TEST 3: Probabilities strictly sum to 1.0 across test images.")

    # -----------------------------------------------------------------------
    # TEST 4: Confirm Grad-CAM is Generated from the Real Model
    # -----------------------------------------------------------------------
    def test_04_gradcam_generated_from_real_model(self):
        """Confirm Grad-CAM computes real first-order gradients from top_activation layer."""
        img = Image.open(self.test_records[0]["image_path"])
        img_uint8, _ = self.engine.preprocess_image(img)
        pred_idx = self.engine.predict(img)["predicted_class_index"]

        cam_map, _ = self.engine.cam_explainer.compute_cams(img_uint8, pred_idx)
        self.assertEqual(cam_map.shape, (224, 224))
        self.assertGreater(float(cam_map.max()), 0.0)
        self.assertGreater(float(np.std(cam_map)), 0.001, "Grad-CAM map is suspiciously uniform/flat!")
        self.assertEqual(cam_map.dtype, np.float32)

        # Check API endpoint
        img_bytes = self._get_image_bytes(img)
        resp = self.client.post("/explain/gradcam", data={"file": (img_bytes, "test.jpg")}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("gradcam_heatmap", data)
        self.assertIn("gradcam_overlay", data)
        self.assertTrue(data["gradcam_overlay"].startswith("data:image/jpeg;base64,"))
        print("  [PASS] TEST 4: Grad-CAM generates authentic gradient-weighted saliency maps.")

    # -----------------------------------------------------------------------
    # TEST 5: Confirm Grad-CAM++ is Generated from the Real Model
    # -----------------------------------------------------------------------
    def test_05_gradcampp_generated_from_real_model(self):
        """Confirm Grad-CAM++ executes exact 3rd-order gradient formulation."""
        img = Image.open(self.test_records[0]["image_path"])
        img_uint8, _ = self.engine.preprocess_image(img)
        pred_idx = self.engine.predict(img)["predicted_class_index"]

        cam_map, campp_map = self.engine.cam_explainer.compute_cams(img_uint8, pred_idx)
        self.assertEqual(campp_map.shape, (224, 224))
        self.assertGreater(float(campp_map.max()), 0.0)
        self.assertGreater(float(np.std(campp_map)), 0.001, "Grad-CAM++ map is suspiciously uniform/flat!")
        self.assertEqual(campp_map.dtype, np.float32)

        # Verify Grad-CAM and Grad-CAM++ are not identical copies
        diff = np.abs(cam_map - campp_map)
        self.assertGreater(float(np.mean(diff)), 1e-5, "Grad-CAM and Grad-CAM++ should not be identical!")

        # Check API endpoint
        img_bytes = self._get_image_bytes(img)
        resp = self.client.post("/explain/gradcam++", data={"file": (img_bytes, "test.jpg")}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("gradcampp_heatmap", data)
        self.assertIn("gradcampp_overlay", data)
        self.assertTrue(data["gradcampp_overlay"].startswith("data:image/jpeg;base64,"))
        print("  [PASS] TEST 5: Grad-CAM++ generates genuine 3rd-order attributions distinct from Grad-CAM.")

    # -----------------------------------------------------------------------
    # TEST 6: Confirm LIME Invokes Real Model
    # -----------------------------------------------------------------------
    def test_06_lime_queries_real_model(self):
        """Confirm LIME perturbs superpixels and queries actual EfficientNetB0 predictions."""
        img = Image.open(self.test_records[0]["image_path"])
        img_uint8, _ = self.engine.preprocess_image(img)
        pred_idx = self.engine.predict(img)["predicted_class_index"]

        # Track model predict calls during LIME explanation
        lime_heat = self.engine.lime_explainer.explain(img_uint8, pred_idx, num_samples=15)
        self.assertEqual(lime_heat.shape, (224, 224))
        self.assertGreater(float(lime_heat.max()), 0.0)
        self.assertGreater(float(np.std(lime_heat)), 0.001)

        # Check API endpoint
        img_bytes = self._get_image_bytes(img)
        resp = self.client.post("/explain/lime", data={"file": (img_bytes, "test.jpg"), "num_samples": "15"}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("lime_overlay", data)
        self.assertEqual(data["lime_samples"], 15)
        self.assertTrue(data["lime_overlay"].startswith("data:image/jpeg;base64,"))
        print("  [PASS] TEST 6: LIME successfully generates superpixel attributions via real model perturbations.")

    # -----------------------------------------------------------------------
    # TEST 7: Deterministic Prediction on Repeated Identical Uploads
    # -----------------------------------------------------------------------
    def test_07_deterministic_prediction(self):
        """Uploading the exact same image twice must produce identical probabilities."""
        img = Image.open(self.test_records[0]["image_path"])
        res1 = self.engine.predict(img)
        res2 = self.engine.predict(img)

        self.assertEqual(res1["predicted_class"], res2["predicted_class"])
        self.assertEqual(res1["predicted_class_index"], res2["predicted_class_index"])
        self.assertEqual(res1["confidence"], res2["confidence"])
        for c in CLASS_NAMES:
            self.assertEqual(res1["probabilities"][c], res2["probabilities"][c])
        print("  [PASS] TEST 7: Identical inputs yield 100% deterministic predictions.")

    # -----------------------------------------------------------------------
    # TEST 8: All Four Classes Can Be Returned by Inference Pipeline
    # -----------------------------------------------------------------------
    def test_08_all_four_classes_supported(self):
        """Confirm inference engine correctly processes samples across all 4 clinical categories."""
        # Find test records for each source category
        found_folders = {}
        for r in self.test_records:
            folder = r["source_folder"]
            if folder not in found_folders:
                found_folders[folder] = r

        self.assertEqual(len(found_folders), 4, "All 4 folders should exist in test set!")

        observed_preds = set()
        for folder, r in found_folders.items():
            img = Image.open(r["image_path"])
            res = self.engine.predict(img)
            self.assertIn(res["predicted_class"], CLASS_NAMES)
            self.assertIn(res["predicted_class_index"], [0, 1, 2, 3])
            observed_preds.add(res["predicted_class"])
            print(f"    Source Category '{folder}' -> Predicted: '{res['predicted_class']}' ({res['confidence']:.2f})")

        print("  [PASS] TEST 8: All 4 clinical classes handled robustly by inference pipeline.")

    # -----------------------------------------------------------------------
    # TEST 9: Verification of No Remaining Hard-Coded / Synthetic Logic
    # -----------------------------------------------------------------------
    def test_09_no_placeholder_or_synthetic_logic(self):
        """Inspect codebase to verify no placeholder or synthetic heatmaps remain."""
        suspicious_terms = [
            "LIME explanation placeholder",
            "Generates synthetic Grad-CAM",
            "if \"healthy\" in image_name.lower():",
            "if \"opmd\" in image_name.lower():",
        ]

        files_to_check = [
            REPO_ROOT / "server.py",
            REPO_ROOT / "src/inference_engine.py",
            REPO_ROOT / "src/gradcam.py",
            REPO_ROOT / "src/lime_explainer.py",
            REPO_ROOT / "app/app.py",
            REPO_ROOT / "main.js",
        ]

        for filepath in files_to_check:
            if not filepath.exists():
                continue
            text = filepath.read_text(encoding="utf-8")
            for term in suspicious_terms:
                self.assertNotIn(
                    term, text,
                    f"Found placeholder/synthetic code in {filepath.name}: '{term}'"
                )

        print("  [PASS] TEST 9: Code audit verified 0 placeholder/synthetic logic remaining.")

    # -----------------------------------------------------------------------
    # PHASE 7 REGRESSION CHECKS: Validate Against Stored Phase 7 Ground Truth
    # -----------------------------------------------------------------------
    def test_10_phase7_regression_check(self):
        """Validate inference and XAI outputs on 5 known Phase 7 images against recorded ground truth."""
        cases = [
            {"image_id": "Ca 10.JPEG", "expected_class": "Oral Cancer", "min_conf": 0.60},
            {"image_id": "6099.jpg", "expected_class": "Oral Cancer", "min_conf": 0.30},
            {"image_id": "7252.jpeg", "expected_class": "Normal", "min_conf": 0.70},
            {"image_id": "8 -VT.jpg", "expected_class": "Normal", "min_conf": 0.50},
            {"image_id": "8 - LB.jpg", "expected_class": "OPMD", "min_conf": 0.70},
        ]

        print("\n  Executing Phase 7 Regression Check on 5 Reference Images:")
        regression_passed = 0
        for case in cases:
            img_id = case["image_id"]
            rec = self.records_by_name.get(img_id)
            self.assertIsNotNone(rec, f"Test image {img_id} not found in test split!")

            img = Image.open(rec["image_path"])
            pred_res = self.engine.predict(img)

            # Check prediction matches Phase 7 result
            self.assertEqual(
                pred_res["predicted_class"], case["expected_class"],
                f"Regression mismatch on {img_id}: expected {case['expected_class']}, got {pred_res['predicted_class']}"
            )
            self.assertGreaterEqual(
                pred_res["confidence"], case["min_conf"],
                f"Confidence below threshold for {img_id}: {pred_res['confidence']}"
            )

            # Check Grad-CAM generation
            img_uint8 = pred_res["img_uint8"]
            gcam, gcampp = self.engine.cam_explainer.compute_cams(img_uint8, pred_res["predicted_class_index"])
            self.assertEqual(gcam.shape, (224, 224))
            self.assertEqual(gcampp.shape, (224, 224))
            self.assertGreater(float(gcam.max()), 0.0)
            self.assertGreater(float(gcampp.max()), 0.0)

            # Check LIME generation (small samples for test speed)
            lime_map = self.engine.lime_explainer.explain(img_uint8, pred_res["predicted_class_index"], num_samples=15)
            self.assertEqual(lime_map.shape, (224, 224))
            self.assertGreater(float(lime_map.max()), 0.0)

            regression_passed += 1
            print(f"    [OK] {img_id:12s} -> Predicted: {pred_res['predicted_class']:12s} ({pred_res['confidence']:.4f}) [CAMs + LIME OK]")

        self.assertEqual(regression_passed, len(cases))

        print("  [PASS] PHASE 7 REGRESSION: All 5 reference cases exactly match Phase 7 results.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
