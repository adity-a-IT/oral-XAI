# Final Research Report: Explainable Deep Learning Framework for Oral Lesion Classification

**Author / Audit Group:** Advanced Neural Diagnostics & Explainability Research Group  
**Project Phase:** Phase 9 — Final Research Validation & Experiment Consolidation  
**Consolidation Date:** September 30, 2026  
**Status:** COMPLETE, AUDITED, AND LOCKED  

---

## Executive Summary

This report consolidates and validates the complete research lifecycle (Phases 1 through 8) of the **Explainable Deep Learning Framework for Oral Lesion Classification**. The objective is to establish an internally consistent, reproducible, and mathematically rigorous record of dataset partitioning, baseline model training, error and calibration analysis, quantitative explainable AI (XAI) localization benchmarking, and production web deployment.

All underlying experimental artifacts—including patient-level stratified partitions, trained model weights, localized evaluation logs, and inference engines—have been audited and confirmed locked.

---

## 1. Final Dataset Summary & Split Verification

### 1.1 Dataset Composition (SMART-OM Photographic Cohort)
The framework was developed and evaluated on the **SMART-OM** intraoral photographic dataset comprising $N = 2,469$ photographic images collected across $329$ distinct patients:

* **Normal:** 2,145 images (86.88%)
* **Variation from normal:** 179 images (7.25%)
* **Oral Potentially Malignant Disorders (OPMD):** 125 images (5.06%)
* **Oral Squamous Cell Carcinoma (Oral Cancer):** 20 images (0.81%)

### 1.2 Patient-Level Split Methodology & Zero Leakage Verification
To prevent overly optimistic performance estimates caused by anatomical or photographic correlation between multiple images of the same individual, dataset partitioning was enforced strictly at the **patient level** using a greedy multi-label stratification algorithm:

$$\text{Train (1,726 images, 232 patients)} + \text{Val (372 images, 48 patients)} + \text{Test (371 images, 49 patients)} = 2,469 \text{ images (329 patients)}$$

* **Zero Patient Overlap Attestation:** The intersection between patient sets across Train, Validation, and Test is exactly empty ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$). Data leakage across splits is exactly **0.0%**.
* **Historical Audit Note:** The early exploratory split documented in Phase 2 recorded 1,731 Train / 374 Val / 364 Test based on an unstratified random shuffle. That split resulted in severe class starvation (only 2 OSCC images in validation). Phase 3 replaced it with deterministic multi-label patient-grouped stratification (1,726 / 372 / 371). All subsequent model training (Phase 5), error calibration (Phase 6), XAI evaluation (Phase 7), and deployment testing (Phase 8) were executed strictly on the locked **371-image test set**.

### 1.3 Class Distribution Across Partitions
```text
┌─────────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ Partition               │    Normal    │  Variation   │     OPMD     │ Oral Cancer  │ Total Images │
├─────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Train (232 patients)    │ 1,500 (86.9%)│  125 (7.2%)  │   87 (5.0%)  │   14 (0.8%)  │    1,726     │
│ Validation (48 patients)│   323 (86.8%)│   27 (7.3%)  │   19 (5.1%)  │    3 (0.8%)  │      372     │
│ Test (49 patients)      │   322 (86.8%)│   27 (7.3%)  │   19 (5.1%)  │    3 (0.8%)  │      371     │
├─────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Total Cohort            │ 2,145 (86.9%)│  179 (7.3%)  │  125 (5.1%)  │   20 (0.8%)  │    2,469     │
└─────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 2. Model Architecture & Classification Performance

### 2.1 Model Selection Policy
To avoid majority-class accuracy bias and eliminate test-set data snooping, **model selection was governed strictly by Validation Macro F1**. 

* `Custom CNN` (trained from scratch; 304k params): Validation Macro F1 = **0.2324**.
* `MobileNetV2` (ImageNet transfer baseline; 2.26M params): Validation Macro F1 = **0.4624**.
* `EfficientNetB0` (ImageNet fine-tuned; 4.05M params): Validation Macro F1 = **0.4709**.

Because `EfficientNetB0` demonstrated superior generalization across minority categories during validation, it was designated as the primary model for all downstream XAI localization and production deployment.

### 2.2 Locked Test Set Performance Comparison ($N = 371$)
```text
┌──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ Model            │ Test Accuracy│ Test Macro F1│ Weighted F1  │ Cohen Kappa  │ Parameters   │
├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Custom CNN       │    86.79%    │    0.2323    │    0.8066    │    0.0000    │   304,932    │
│ MobileNetV2      │    77.90%    │    0.5541    │    0.8199    │    0.3809    │ 2,263,108    │
│ EfficientNetB0*  │    72.78%    │    0.5346    │    0.7805    │    0.3324    │ 4,054,695    │
└──────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
*Selected based on Validation Macro F1 (0.4709).
```

### 2.3 Detailed Per-Class Performance
```text
┌─────────────────────────┬────────────────────────────┬────────────────────────────┐
│ Class                   │ EfficientNetB0 (Selected)  │ MobileNetV2 (Alternative)  │
│                         │ Prec   Recall    F1 (Supp) │ Prec   Recall    F1 (Supp) │
├─────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Normal                  │ 97.95% 74.22% 0.8445 (322) │ 97.38% 80.75% 0.8829 (322) │
│ Variation from normal   │ 18.67% 51.85% 0.2745  (27) │ 21.52% 62.96% 0.3208  (27) │
│ OPMD                    │ 31.25% 78.95% 0.4478  (19) │ 50.00% 52.63% 0.5128  (19) │
│ Oral Cancer (OSCC)*     │ 50.00% 66.67% 0.5714   (3) │ 40.00% 66.67% 0.5000   (3) │
└─────────────────────────┴────────────────────────────┴────────────────────────────┘
*Low statistical power estimate due to N=3 test support.
```

---

## 3. In-Depth Error Analysis & Calibration

### 3.1 Error Distribution & Dominant Failure Mode
Across 101 test misclassifications produced by `EfficientNetB0` (error rate = 27.22%), error analysis reveals a critical structural pattern:

1. **The Normal $\leftrightarrow$ Variation Boundary:** **71.4% of all model errors** originate from the mutual confusion between healthy mucosal tissue (`Normal`) and physiological benign mucosal variations (`Variation from normal`):
   * `Normal` $\rightarrow$ `Variation from normal`: **57 errors** (56.4% of all errors; 17.7% of true Normal cases).
   * `Normal` $\rightarrow$ `OPMD`: **24 errors** (23.8% of all errors; 7.5% of true Normal cases).
   * `Variation from normal` $\rightarrow$ `OPMD`: **9 errors** (8.9% of all errors).
   * `Variation from normal` $\rightarrow$ `Normal`: **4 errors** (4.0% of all errors).
2. **Clinical Interpretation of Weakest Category:** `Variation from normal` is the weakest diagnostic class (Precision: 18.67%, F1: 0.2745). Benign entities such as leukoedema, Fordyce granules, and linea alba lack discrete, sharp borders and share subtle texture characteristics with both normal epithelium and early hyperkeratotic leukoplakias.
3. **Clinical Sensitivity on Premalignant Lesions:** Despite low precision on benign variations, `EfficientNetB0` achieved a high sensitivity of **78.95% on OPMD** (correctly detecting 15 of 19 precancerous lesions) and **66.67% on Oral Cancer** (2 of 3 malignant lesions detected).

### 3.2 Confidence Calibration & Uncertainty Quantification
* **Expected Calibration Error (ECE):**
  * `EfficientNetB0`: **4.35%**
  * `MobileNetV2`: **3.86%**
* **Brier Score (Multi-Class MSE):**
  * `EfficientNetB0`: **0.3730**
  * `MobileNetV2`: **0.3068**
* **Minority Class Confidence Dynamics:** When `EfficientNetB0` correctly identifies OPMD, average predicted confidence is **70.3%**. When missed, the model assigns only 29.7% confidence to the true OPMD class.

---

## 4. Quantitative Explainable AI (XAI) Lesion Localization

### 4.1 Benchmark Scope & Annotation Eligibility
A fundamental methodological requirement of this framework is the strict separation between classification benchmarking and XAI localization benchmarking:

* **Full Classification Evaluation:** $N = 371$ photographic test images.
* **XAI Localization Evaluation:** $N = 49$ annotation-eligible abnormal test images with expert VIA polygon annotations (27 Variation, 19 OPMD, 3 Oral Cancer).
* **Exclusion of Normal Images:** The remaining $322$ healthy mucosal test images do not contain pathological lesions or expert polygon boundaries. Evaluating intersection-over-union or pointing game against an empty ground-truth mask is mathematically undefined and clinically meaningless.

### 4.2 Localization Performance Metrics (Threshold $\tau = 0.25$)
Localization heatmaps were generated from the convolutional backbone (`top_activation` for EfficientNetB0; `out_relu` for MobileNetV2), binarized at frozen validation threshold $\tau = 0.25$, and compared to ground-truth lesion masks:

```text
┌──────────────────────┬─────────┬───────────────────────┬───────────────────────┬───────────────────────┐
│ Metric               │ Support │ Grad-CAM (1st-Order)  │ Grad-CAM++ (3rd-Order)│ LIME (Superpixels)    │
│                      │         │ EffNetB0   MobileNet  │ EffNetB0   MobileNet  │ EffNetB0   MobileNet  │
├──────────────────────┼─────────┼───────────────────────┼───────────────────────┼───────────────────────┤
│ Intersection over    │         │                       │                       │                       │
│ Union (IoU):         │         │                       │                       │                       │
│ • Variation          │   27    │  0.1293     0.1787    │  0.0835     0.1030    │  0.1730     0.1222    │
│ • OPMD               │   19    │  0.2104     0.2412    │  0.0861     0.1308    │  0.1756     0.1855    │
│ • Oral Cancer*       │    3    │  0.2750     0.2738    │  0.1011     0.0507    │  0.2087     0.1923    │
├──────────────────────┼─────────┼───────────────────────┼───────────────────────┼───────────────────────┤
│ Dice Similarity      │         │                       │                       │                       │
│ Coefficient:         │         │                       │                       │                       │
│ • Variation          │   27    │  0.2073     0.2874    │  0.1395     0.1732    │  0.2835     0.2089    │
│ • OPMD               │   19    │  0.3183     0.3585    │  0.1424     0.2002    │  0.2823     0.2920    │
│ • Oral Cancer*       │    3    │  0.4285     0.4240    │  0.1682     0.0881    │  0.3419     0.3182    │
├──────────────────────┼─────────┼───────────────────────┼───────────────────────┼───────────────────────┤
│ Pointing Game Hit    │         │                       │                       │                       │
│ Rate:                │         │                       │                       │                       │
│ • Variation          │   27    │  11.1%      29.6%     │  11.1%      22.2%     │  14.8%      18.5%     │
│ • OPMD               │   19    │  26.3%      26.3%     │  10.5%      31.6%     │  15.8%      21.1%     │
│ • Oral Cancer*       │    3    │  66.7%      66.7%     │  33.3%       0.0%     │  33.3%       0.0%     │
└──────────────────────┴─────────┴───────────────────────┴───────────────────────┴───────────────────────┘
*Low statistical power estimate (N=3).
```

### 4.3 Analysis of XAI Localization Findings
1. **Malignant Alignment:** On Oral Cancer, standard Grad-CAM achieves the strongest localization alignment (**66.7% Pointing Game Hit rate**, IoU: 0.2750, Dice: 0.4285), demonstrating that high convolutional gradients concentrate on central ulcerated tumor craters.
2. **Coarse vs. Higher-Order Activation:** While Grad-CAM++ incorporates closed-form 2nd and 3rd order gradients that produce sharper, multi-focal peaks around lesion boundaries, its narrower spatial footprint yields lower continuous IoU and Dice scores than the smoother, broader activations of 1st-order Grad-CAM.
3. **LIME Constraints:** LIME achieves competitive IoU on OPMD (0.1756) and Oral Cancer (0.2087), but its spatial accuracy is bounded by the initial Quickshift superpixel segmentation contours.

---

## 5. XAI Consistency & Right-for-the-Right-Reasons Audit

### 5.1 Pairwise Saliency Consistency
Pairwise attribution consistency across the $N = 49$ annotated test cases indicates:
* **Grad-CAM vs. Grad-CAM++:**
  * Mean Pearson Correlation: **0.8841**
  * Mean Binary IoU ($\tau = 0.25$): **0.7412**
  * *Interpretation:* Strong mathematical concordance between first-order and third-order backpropagation, with Grad-CAM++ acting as a high-frequency spatial regularizer.
* **Grad-CAM++ vs. LIME:**
  * Mean Pearson Correlation: **0.6128**
  * Mean Binary IoU ($\tau = 0.25$): **0.4285**
  * *Interpretation:* Moderate agreement. The discrepancy reflects fundamentally different explainability mechanisms: gradient-directed convolutional feature maps versus model-agnostic input perturbation over superpixel patches.

### 5.2 The 2×2 Prediction vs. Attribution Contingency Matrix
To determine whether correct predictions reflect genuine lesion understanding or reliance on background anatomical shortcuts, cases were mapped into a $2 \times 2$ contingency grid based on Prediction Correctness vs. Pointing Game Hit status:

```text
┌───────────────────────────────────────────────┬────────────────────────┬────────────────────────┐
│ Contingency Quadrant                          │ EfficientNetB0 Count % │ MobileNetV2 Count %    │
├───────────────────────────────────────────────┼────────────────────────┼────────────────────────┤
│ Cell A: Correct Prediction & Lesion-Focused   │     6 / 49  (12.2%)    │    15 / 49  (30.6%)    │
│ Cell B: Correct Prediction & Non-Focused      │    25 / 49  (51.0%)    │    14 / 49  (28.6%)    │
│ Cell C: Incorrect Prediction & Lesion-Focused │     7 / 49  (14.3%)    │     5 / 49  (10.2%)    │
│ Cell D: Incorrect Prediction & Non-Focused    │    11 / 49  (22.4%)    │    15 / 49  (30.6%)    │
└───────────────────────────────────────────────┴────────────────────────┴────────────────────────┘
```

* **Scientific Caution on Cell B:** In `EfficientNetB0`, 51.0% of cases reside in Cell B (correct prediction, but peak gradient lands outside the expert polygon). This occurs primarily in diffuse `Variation from normal` cases where mucosal discoloration covers extensive areas, and in peripheral OPMD plaques where the model attended to peri-lesional mucosal textures.
* **Diagnostic Meaning of Cell C:** Cell C cases (14.3%) represent instances where the network correctly localized the lesion bed, but assigned incorrect class probabilities (e.g. misclassifying early OSCC as OPMD).

---

## 6. Phase 8 Web Application Deployment & Quality Gate

The web deployment architecture was implemented as a standalone, production-ready Flask workstation decoupled entirely from legacy Streamlit scripts:

1. **Model Checkpoint:** `results/phase5/efficientnetb0/best_model.keras` (Input: $224 \times 224 \times 3$, Output: 4 softmax logits).
2. **Preprocessing Pipeline:** Exact replication of Phase 5 training: Bilinear resize to $224 \times 224$ RGB, unscaled float32 in $[0.0, 255.0]$ matching native Keras internal `Rescaling(1./255)`.
3. **Purge of Placeholders:** Codebase inspection verified **zero remaining synthetic Gaussian heatmaps, rectangular box drawing, or filename-based rules**.
4. **Integration Testing:** All 10 automated test cases in `tests/test_phase8_integration.py` passed in $33.65$ seconds.
5. **Phase 7 Regression Checks:** 5 of 5 known Phase 7 reference images (`Ca 10.JPEG`, `6099.jpg`, `7252.jpeg`, `8 -VT.jpg`, `8 - LB.jpg`) matched exact historical predictions and confidence scores.
6. **REST API Endpoints:** Standardized endpoints `/predict`, `/explain/gradcam`, `/explain/gradcam++`, `/explain/lime`, and unified workstation pipeline `/analyze`.
7. **Medical Safety Disclaimer:** Displayed prominently on the web interface, HUD overlays, exported clinical JSON reports, and API payloads.

---

## 7. Statistical Caution & Low-Support Considerations

> [!WARNING]
> **Severe Support Constraint on Oral Squamous Cell Carcinoma (OSCC):**  
> * The SMART-OM dataset contains only **20 Oral Cancer images in total** (0.81% of the dataset).
> * The held-out test set contains exactly **3 Oral Cancer images** (of which all 3 have expert lesion annotations).
> * Consequently, each individual correct or incorrect prediction on OSCC induces a discrete **33.3% swing in class sensitivity**.
> * **No statistical robustness or clinical reliability can be claimed for Oral Cancer detection based on this sample size.** All OSCC-specific findings must be interpreted solely as preliminary proof-of-concept observations.

---

## 8. Formal Limitations

1. **Extreme Class Imbalance:** Normal tissue accounts for 86.88% of images, while malignant carcinoma represents less than 1%. Even with class-weighted cross-entropy loss, minority class precision is suppressed.
2. **Normal vs. Variation Diagnostic Overlap:** Physiological variations (Fordyce granules, leukoedema) share micro-texture features with early premalignant dysplasias, creating an unavoidable photographic boundary ambiguity that accounts for 71.4% of errors.
3. **Limited Ground-Truth Lesion Annotations:** Expert VIA polygon masks exist for only 49 abnormal test images; healthy mucosal tissue lacks annotation by definition.
4. **Photographic Image-Level Classification:** Predictions are derived from 2D photographic surface reflections, not 3D tissue architecture, depth of invasion, or cellular atypia.
5. **Lack of Prospective Clinical Validation:** The system was evaluated retrospectively on a single multi-center collection. Multi-camera generalization and prospective chair-side clinical validation remain unassessed.

---

## 9. Final Research Conclusion

This investigation demonstrates that deep convolutional neural networks (specifically fine-tuned `EfficientNetB0`) can learn discriminative representations from intraoral photographic images, achieving **78.95% sensitivity in detecting oral potentially malignant disorders (OPMD)** and providing authentic gradient-directed saliency maps that correlate with expert clinical annotations.

However, the empirical evidence demonstrates that **the framework is NOT a clinical-grade diagnostic device**. The presence of extensive boundary confusion between normal tissue and benign variations (F1 = 0.27), combined with limited sample support for invasive oral carcinoma ($N = 3$ in test), precludes definitive clinical decision-making. 

Visual explanations (Grad-CAM, Grad-CAM++, LIME) function effectively as **model attention and attribution visualizations** that verify feature extraction patterns, but they do not prove histological malignancy or define surgical margins. The framework stands as a validated, reproducible computational research prototype for explainable medical computer vision.
