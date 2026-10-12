"""ORAL-XAI :: High-Performance Standalone Neural Diagnostic & Explainability Server.

Phase 8: Real Neural Model + XAI Web Application Integration.
Serves the cyber-medical oral oncology diagnostic workstation,
providing genuine inference from EfficientNetB0 (results/phase5/efficientnetb0/best_model.keras)
and genuine Grad-CAM, Grad-CAM++, and LIME explainability overlays.
Streamlit decoupled.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

# Add repository root to Python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.inference_engine import (
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    RESEARCH_DISCLAIMER,
    OralXAIInferenceEngine,
)

app = Flask(
    __name__,
    static_folder=str(BASE_DIR),
    static_url_path="",
)

# Initialize Real Neural Inference Engine (Loaded once at server start)
print("[Server Init] Initializing OralXAI Real Neural Inference Engine...", flush=True)
engine = OralXAIInferenceEngine.get_instance(model_path=DEFAULT_MODEL_PATH)
print(f"[Server Init] Active model: {engine.model_name} from {engine.model_path}", flush=True)

# Preset image mappings for testing all 4 diagnostic classes
PRESET_MAP = {
    "leukoplakia": BASE_DIR / "assets/optimized/oral_lesion_macro.jpg",
    "opmd": BASE_DIR / "assets/optimized/oral_lesion_macro.jpg",
    "carcinoma": BASE_DIR / "assets/optimized/cancer_biopsy.jpg",
    "oscc": BASE_DIR / "assets/optimized/cancer_biopsy.jpg",
    "healthy": BASE_DIR / "assets/optimized/oral_lesion_exam.jpg",
    "normal": BASE_DIR / "assets/optimized/oral_lesion_exam.jpg",
    "variation": BASE_DIR / "assets/optimized/oral_lesion_exam.jpg",
}


def extract_input_image() -> Tuple[Image.Image, str]:
    """Helper extracting and validating PIL image from multipart form, JSON, or preset."""
    # Check max payload size (20MB)
    if request.content_length and request.content_length > 20 * 1024 * 1024:
        raise ValueError("Oversized image: File size exceeds the 20MB limit.")

    # 1. File Upload
    if "file" in request.files and request.files["file"].filename:
        file = request.files["file"]
        filename = file.filename
        try:
            img = Image.open(file.stream)
            img.verify()  # Check for corruption
            file.stream.seek(0)
            img = Image.open(file.stream)
            
            # Dimension checks
            if img.width > 8000 or img.height > 8000:
                raise ValueError(f"Oversized image dimensions: {img.width}x{img.height} exceeds 8000x8000 limit.")
            return img, filename
        except Exception as e:
            raise ValueError(f"Corrupted or invalid image uploaded: {str(e)}")

    # 2. JSON Base64
    if request.is_json and "image_b64" in request.json:
        b64_str = request.json["image_b64"]
        filename = request.json.get("filename", "uploaded_scan.jpg")
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        try:
            img_bytes = base64.b64decode(b64_str)
            if len(img_bytes) > 20 * 1024 * 1024:
                raise ValueError("Oversized image: Base64 payload exceeds 20MB limit.")
            img = Image.open(io.BytesIO(img_bytes))
            img.verify()
            img = Image.open(io.BytesIO(img_bytes))
            if img.width > 8000 or img.height > 8000:
                raise ValueError(f"Oversized image dimensions: {img.width}x{img.height} exceeds 8000x8000 limit.")
            return img, filename
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {str(e)}")

    # 3. Preset Fallback
    preset_key = request.form.get("preset", "").lower().strip()
    if not preset_key and request.is_json:
        preset_key = request.json.get("preset", "").lower().strip()

    if not preset_key:
        preset_key = "leukoplakia"

    target_path = PRESET_MAP.get(preset_key, PRESET_MAP["leukoplakia"])
    if target_path.exists():
        img = Image.open(target_path)
        return img, target_path.name
    else:
        # Fallback to test dataset image if assets missing
        test_samples = list((BASE_DIR / "dataset").glob("**/*.jpg"))
        if test_samples:
            return Image.open(test_samples[0]), test_samples[0].name
        raise FileNotFoundError("No preset image could be located.")


# ---------------------------------------------------------------------------
# Frontend Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Serves the primary oral-XAI platform and live clinical workstation."""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:path>")
def static_proxy(path: str):
    """Serves static assets (css, js, images, datasets)."""
    return send_from_directory(BASE_DIR, path)


# ---------------------------------------------------------------------------
# Health & Architecture Telemetry API
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health_check():
    """Returns runtime telemetry, loaded model details, and verified class mapping."""
    return jsonify({
        "status": "ONLINE",
        "engine": "Oral-XAI Real Neural Inference & Explainability Suite",
        "model_architecture": f"{engine.model_name} (Phase 5 ImageNet-pretrained Checkpoint)",
        "model_path": str(engine.model_path),
        "input_shape": list(engine.input_shape),
        "output_shape": list(engine.output_shape),
        "classes": CLASS_NAMES,
        "xai_pipeline": "Real Grad-CAM (1st-Order) + Exact Grad-CAM++ (3rd-Order) + Real LIME",
        "disclaimer": RESEARCH_DISCLAIMER,
    })


# ---------------------------------------------------------------------------
# Real Prediction API (/predict and /api/predict)
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict_endpoint():
    """Returns genuine 4-class prediction and probability distribution from EfficientNetB0."""
    try:
        img, filename = extract_input_image()
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400

    try:
        t0 = time.perf_counter()
        pred_res = engine.predict(img)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return jsonify({
            "status": "success",
            "filename": filename,
            "predicted_class": pred_res["predicted_class"],
            "predicted_class_index": pred_res["predicted_class_index"],
            "confidence": pred_res["confidence"],
            "confidence_percentage": pred_res["confidence_percentage"],
            "risk_tier": pred_res["risk_tier"],
            "risk_class": pred_res["risk_class"],
            "probabilities": pred_res["probabilities"],
            "probabilities_percentage": pred_res["probabilities_percentage"],
            "raw_probabilities": pred_res["raw_probabilities"],
            "probability_sum": pred_res["probability_sum"],
            "latency_ms": latency_ms,
            "model_name": engine.model_name,
            "disclaimer": RESEARCH_DISCLAIMER,
        })
    except Exception as e:
        return jsonify({"status": "error", "error": f"Inference failed: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Real Grad-CAM Endpoint (/explain/gradcam and /api/explain/gradcam)
# ---------------------------------------------------------------------------
@app.route("/explain/gradcam", methods=["POST"])
@app.route("/api/explain/gradcam", methods=["POST"])
def explain_gradcam_endpoint():
    """Returns genuine Grad-CAM heatmap and overlay for the input image."""
    try:
        img, filename = extract_input_image()
        colormap = request.form.get("colormap", request.json.get("colormap", "jet") if request.is_json else "jet")
        opacity = float(request.form.get("opacity", request.json.get("opacity", 0.55) if request.is_json else 0.55))
        class_idx_param = request.form.get("class_index", request.json.get("class_index", None) if request.is_json else None)
        target_class_idx = int(class_idx_param) if class_idx_param is not None else None
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400

    try:
        t0 = time.perf_counter()
        pred_res = engine.predict(img)
        img_uint8 = pred_res.pop("img_uint8")
        target_idx = target_class_idx if target_class_idx is not None else pred_res["predicted_class_index"]

        gcam_norm, _ = engine.cam_explainer.compute_cams(img_uint8, target_idx)
        gcam_heatmap = engine.render_heatmap(gcam_norm, colormap_name=colormap)
        gcam_overlay = engine.render_overlay(img_uint8, gcam_norm, colormap_name=colormap, opacity=opacity)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return jsonify({
            "status": "success",
            "filename": filename,
            "target_class": CLASS_NAMES[target_idx],
            "target_class_index": target_idx,
            "original_image": engine.to_b64(Image.fromarray(img_uint8)),
            "gradcam_heatmap": engine.to_b64(gcam_heatmap),
            "gradcam_overlay": engine.to_b64(gcam_overlay),
            "gradcam_data_url": engine.to_b64(gcam_overlay),
            "latency_ms": latency_ms,
            "disclaimer": RESEARCH_DISCLAIMER,
        })
    except Exception as e:
        return jsonify({"status": "error", "error": f"Grad-CAM generation failed: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Real Grad-CAM++ Endpoint (/explain/gradcam++ and /api/explain/gradcam++)
# ---------------------------------------------------------------------------
@app.route("/explain/gradcam++", methods=["POST"])
@app.route("/explain/gradcampp", methods=["POST"])
@app.route("/api/explain/gradcam++", methods=["POST"])
@app.route("/api/explain/gradcampp", methods=["POST"])
def explain_gradcampp_endpoint():
    """Returns genuine exact 3rd-order Grad-CAM++ heatmap and overlay."""
    try:
        img, filename = extract_input_image()
        colormap = request.form.get("colormap", request.json.get("colormap", "jet") if request.is_json else "jet")
        opacity = float(request.form.get("opacity", request.json.get("opacity", 0.55) if request.is_json else 0.55))
        class_idx_param = request.form.get("class_index", request.json.get("class_index", None) if request.is_json else None)
        target_class_idx = int(class_idx_param) if class_idx_param is not None else None
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400

    try:
        t0 = time.perf_counter()
        pred_res = engine.predict(img)
        img_uint8 = pred_res.pop("img_uint8")
        target_idx = target_class_idx if target_class_idx is not None else pred_res["predicted_class_index"]

        _, gcampp_norm = engine.cam_explainer.compute_cams(img_uint8, target_idx)
        gcampp_heatmap = engine.render_heatmap(gcampp_norm, colormap_name=colormap)
        gcampp_overlay = engine.render_overlay(img_uint8, gcampp_norm, colormap_name=colormap, opacity=opacity)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return jsonify({
            "status": "success",
            "filename": filename,
            "target_class": CLASS_NAMES[target_idx],
            "target_class_index": target_idx,
            "original_image": engine.to_b64(Image.fromarray(img_uint8)),
            "gradcampp_heatmap": engine.to_b64(gcampp_heatmap),
            "gradcampp_overlay": engine.to_b64(gcampp_overlay),
            "gradcampp_data_url": engine.to_b64(gcampp_overlay),
            "latency_ms": latency_ms,
            "disclaimer": RESEARCH_DISCLAIMER,
        })
    except Exception as e:
        return jsonify({"status": "error", "error": f"Grad-CAM++ generation failed: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Real LIME Endpoint (/explain/lime and /api/explain/lime)
# ---------------------------------------------------------------------------
@app.route("/explain/lime", methods=["POST"])
@app.route("/api/explain/lime", methods=["POST"])
def explain_lime_endpoint():
    """Returns genuine LIME superpixel attribution overlay by querying the model."""
    try:
        img, filename = extract_input_image()
        opacity = float(request.form.get("opacity", request.json.get("opacity", 0.65) if request.is_json else 0.65))
        num_samples = int(request.form.get("num_samples", request.json.get("num_samples", 25) if request.is_json else 25))
        class_idx_param = request.form.get("class_index", request.json.get("class_index", None) if request.is_json else None)
        target_class_idx = int(class_idx_param) if class_idx_param is not None else None
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400

    try:
        t0 = time.perf_counter()
        pred_res = engine.predict(img)
        img_uint8 = pred_res.pop("img_uint8")
        target_idx = target_class_idx if target_class_idx is not None else pred_res["predicted_class_index"]

        lime_heat = engine.lime_explainer.explain(img_uint8, target_idx, num_samples=num_samples)
        lime_overlay = engine.render_lime_overlay(img_uint8, lime_heat, opacity=opacity)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return jsonify({
            "status": "success",
            "filename": filename,
            "target_class": CLASS_NAMES[target_idx],
            "target_class_index": target_idx,
            "original_image": engine.to_b64(Image.fromarray(img_uint8)),
            "lime_samples": num_samples,
            "lime_overlay": engine.to_b64(lime_overlay),
            "lime_data_url": engine.to_b64(lime_overlay),
            "latency_ms": latency_ms,
            "disclaimer": RESEARCH_DISCLAIMER,
        })
    except Exception as e:
        return jsonify({"status": "error", "error": f"LIME generation failed: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Comprehensive Unified Workstation API (/analyze and /api/analyze)
# ---------------------------------------------------------------------------
@app.route("/analyze", methods=["POST"])

@app.route("/api/analyze", methods=["POST"])
def analyze_lesion():
    """Executes real inference, real Grad-CAM, real Grad-CAM++, and real LIME overlays."""
    try:
        img, filename = extract_input_image()
        colormap = request.form.get("colormap", "jet")
        opacity = float(request.form.get("opacity", 0.55))
        run_lime = request.form.get("lime", "true").lower() in ["true", "1", "yes"]
        lime_samples = int(request.form.get("lime_samples", 25))
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400

    try:
        # Run real analysis pipeline
        result = engine.analyze(
            pil_img=img,
            colormap_name=colormap,
            opacity=opacity,
            run_lime=run_lime,
            lime_samples=lime_samples,
        )
        result["filename"] = filename
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "error": f"Neural analysis pipeline failed: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Server Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        if sys.stdout.encoding.lower() != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    port = int(os.environ.get("PORT", 5000))
    print(f"[oral-XAI] Standalone Neural Diagnostic Suite running on http://localhost:{port}", flush=True)
    print(f"[oral-XAI] Active Model: {engine.model_name} (Loaded once, prewarmed, ready for requests)", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
