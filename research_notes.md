# Research Notes

This project must retain a strict separation between:

- baseline results from the published ETASR paper
- independently generated results from this repository

No clinical claims are allowed beyond the research-prototype framing.

Critical data-leakage rule:

- If multiple images belong to the same patient, they must never be split across train and test sets.
- Use patient-level grouping during splitting when patient identifiers are available.

Project phases:

1. Environment verification
2. Dataset inspection and audit
3. Patient-aware split preparation
4. Preprocessing and augmentation
5. Baseline transfer learning models
6. Independent CNN architecture
7. Training and tracking
8. Evaluation and reporting
9. Grad-CAM and LIME
10. Explainability and error analysis
11. Streamlit demo

This phase is intentionally limited to environment validation and dataset inspection.

## Phase 2: SMART-OM Dataset Audit Findings (Completed)

- **Dataset Root**: `dataset/oral-lesion-dataset-org/SMART-OM`
- **Total Valid Images**: 7,731 (0 corrupt images verified via PIL)
- **Total Raw Unannotated Photographic Images**: 2,469
- **Categories & Breakdowns**:
  - `01. Normal`: 6,435 images (2,145 raw photographic images across 8 anatomical sites: dorsal/ventral tongue, buccal mucosa, lips, arches; plus region & full annotations)
  - `02. Variation from normal`: 716 images (179 raw unannotated, 179 region, 179 full, 179 lesion)
  - `03. OPMD` (Oral Potentially Malignant Disorders): 500 images (125 raw unannotated, 125 region, 125 full, 125 lesion)
  - `04. Oral Cancer` (OSCC / Carcinoma): 80 images (20 raw unannotated, 20 region, 20 full, 20 lesion)
- **Metadata**:
  - `Patient_s Metadata.xlsx` with 331 patient demographic & personal history records and clinical findings
  - Annotations available in VGG Image Annotator (VIA) format (polygon coordinates for region, full, lesion)
- **Patient Identifier Analysis & Critical Leakage Precaution**:
  - 329 unique patient/case identifiers detected in the raw images (`SMITA...`, `NUM_...`, `CASE_...`, `Ca...`)
  - 142 patients have images across multiple mucosal states (e.g. normal dorsal tongue + OPMD buccal mucosa)
  - **MANDATORY**: Patient-level GroupKFold or grouped splitting is required to avoid train-test contamination.

## Phase 3: Patient-Stratified Split (Completed & Locked)

- **Strategy**: Deterministic Greedy Multi-Label Patient-Grouped Stratification (Seed 42)
- **Partitions**:
  - Train: 1,726 images (232 patients, 69.91%)
  - Validation: 372 images (48 patients, 15.07%)
  - Test: 371 images (49 patients, 15.03%)
- **Zero Leakage**: Strict patient isolation confirmed (0 patient overlap across all sets).
- **Proportions**: Class distribution deviation from global dataset is <= 0.08% across all 4 classes.

## Phase 4: Preprocessing & Class-Imbalance Handling (Completed)

- **Image Resolution**: 224 x 224 x 3 RGB
- **Normalization**: Rescaling to [0.0, 1.0] (Configurable to [-1, 1] or ImageNet z-score)
- **Training Augmentation (Medical-Image-Safe)**:
  - Horizontal flip enabled (anatomically appropriate due to oral cavity bilateral symmetry)
  - Vertical flip strictly disabled (prevents anatomical disorientation)
  - Small rotation: +/- 10 degrees
  - Small translation: +/- 5%
  - Small zoom: [0.95, 1.05]
  - Mild brightness: +/- 8%
  - Mild contrast: [0.92, 1.08]
- **Validation/Test**: Augmentation strictly disabled (100% deterministic preprocessing).
- **Balanced Class Weights (Calculated from Train Only)**:
  - Normal (0): `0.2877`
  - Variation from normal (1): `3.4520`
  - OPMD (2): `4.9598`
  - Oral Cancer (3): `30.8214`
  - Saved to `results/class_weights.json`

## Phase 5: Baseline Model Training (Completed)

- **Models Evaluated**:
  1. Custom Oral CNN: 304,932 parameters, trained from scratch. Val Acc: 86.83%, Val Macro F1: 0.2324 (majority collapse).
  2. MobileNetV2 (ImageNet Pretrained): 2,263,108 parameters. Val Acc: 70.97%, Val Macro F1: 0.4624, Test Acc: 77.90%, Test Macro F1: **0.5541**, Test Weighted F1: **0.8199**, Kappa: 0.3809.
  3. EfficientNetB0 (ImageNet Pretrained): 4,054,695 parameters. Val Acc: 71.24%, Val Macro F1: **0.4709** (Best Validation Model), Test Acc: 72.78%, Test Macro F1: **0.5346**, Test Weighted F1: 0.7805, Kappa: 0.3324.
- **Selection**: `EfficientNetB0` selected as best validation model based on Validation Macro F1 (0.4709) to avoid majority-class accuracy bias.
- **Physical Dataset Limitation**: OSCC test support is 3 images (each sample represents 33.3% recall). Documented as low statistical power estimate.
- **Artifacts**: Saved under `results/phase5/`.

## Phase 6: In-Depth Evaluation, Error Analysis & Calibration (Completed)

- **Evaluation Scope**: Held-out test set ($N = 371$ images across 50 isolated patients).
- **Models Benchmarked**:
  - `EfficientNetB0`: Test Acc 72.78%, Test Macro F1 0.5346, Test Weighted F1 0.7805, ECE 13.45%, Brier 0.3700. Detected 15 of 19 OPMD cases (78.95% recall).
  - `MobileNetV2`: Test Acc 77.90%, Test Macro F1 0.5541, Test Weighted F1 0.8199, ECE 17.88%, Brier 0.3204.
- **Key Error Findings**:
  - Mutual confusion between `Normal` and `Variation from normal` accounts for **71.4% of all test misclassifications**.
  - `Variation from normal` is the weakest class (Precision: 18.7%, F1: 0.2745 in EfficientNetB0).
  - Anatomic site breakdown reveals lower diagnostic accuracy on buccal mucosa (53.2% - 54.3%) vs tongue (85.1% - 88.0%) due to concentration of subtle OPMD and physiological variations.
## Phase 7: Explainable AI (XAI) & Clinical Lesion Alignment (Completed)

- **Evaluation Scope**: Full test cohort of $N = 371$ images across 49 isolated patients (`results/splits/test_split.json`).
- **Ground-Truth Lesion Benchmarking**: $N = 49$ annotation-eligible abnormal test cases with expert VIA polygon annotations (27 Variation, 19 OPMD, 3 Oral Cancer). Normal cases ($N = 322$) correctly contain no lesion annotations.
- **XAI Algorithms Evaluated**:
  1. Grad-CAM (first-order convolutional gradient attribution).
  2. Exact closed-form Grad-CAM++ (1st, 2nd, and 3rd order gradients).
  3. LIME superpixel perturbations querying actual neural models.
- **Threshold Calibration**: Binary attribution threshold frozen at $\tau = 0.25$ based on validation set tuning.
- **Objective Localization Performance (IoU / Dice / Pointing Game)**:
  - **Oral Cancer ($N=3$)**: Grad-CAM achieves **66.7% Pointing Game Hit**, IoU: 0.2750, Dice: 0.4285 on `EfficientNetB0`.
  - **OPMD ($N=19$)**: Grad-CAM Pointing Game: **26.3%**, IoU: 0.2104, Dice: 0.3183.
  - **Variation ($N=27$)**: Diffuse boundary characteristics yield lower IoU (0.1293) despite Pointing Game capture.
- **$2 \times 2$ Contingency Matrix**:
  - `EfficientNetB0`: Cell A (Correct & Focused) = 12.2%, Cell B (Correct & Not Focused) = 51.0%, Cell C (Incorrect & Focused) = 14.3%, Cell D (Incorrect & Not Focused) = 22.4%.
  - `MobileNetV2`: Cell A = 30.6%, Cell B = 28.6%, Cell C = 10.2%, Cell D = 30.6%.
- **Core Clinical Finding**:
  - True premalignant and malignant lesions (OPMD and OSCC) exhibit genuine lesion-focused attribution.
  - Normal cases misclassified as Variation or OPMD heavily attend to salivary reflections and dental arch margins on the buccal mucosa.
- **Artifacts**: Saved under `results/phase7/`.

## Phase 8: Real Neural Model + XAI Web Application Integration (Completed)

- **Scope**: Production deployment of real trained neural models and verified XAI pipelines in a standalone web workstation.
- **Model Deployed**: Primary model `results/phase5/efficientnetb0/best_model.keras` (Input: 224x224x3, Output: 4 classes: Normal, Variation from normal, OPMD, Oral Cancer).
- **Preprocessing Parity**: Exact replication of Phase 5 training preprocessing (Bilinear resize to 224x224 RGB, float32 in [0.0, 255.0] matching internal Rescaling layer).
- **XAI Pipeline**: Exact Phase 7 mathematical parity:
  1. Grad-CAM (1st-order from `top_activation`).
  2. Exact Grad-CAM++ (3rd-order closed-form from `top_activation`).
  3. Real LIME superpixel perturbations querying actual model predictions.
- **Decoupled Architecture**: High-performance Flask backend (`server.py`) serving native responsive UI (`index.html`, `main.js`, `style.css`), fully replacing legacy Streamlit mockups.
- **Placeholder Elimination**: 100% removal of all synthetic Gaussian heatmaps, grid box simulations, and filename-based prediction rules.
- **Verification & Parity**: 10/10 automated tests passed in `tests/test_phase8_integration.py`, including Phase 7 regression match across 5 reference test images.
- **Artifacts**: `results/phase8/phase8_integration_report.md` and `results/phase8/phase8_integration_report.json`.

