# Phase 7: Dataset Split Consistency & Integrity Audit

**Audit Date:** 2026-09-30  
**Scope:** Verification of dataset split consistency across `results/splits/`, Phase 5 training, Phase 6 evaluation, and Phase 7 XAI evaluation.

---

## 1. Executive Summary: The Split Discrepancy Explained

The user flagged a numerical discrepancy between the split sizes:
- **Reference from early audit:** Train: 1,731 | Validation: 374 | Test: 364 (Total: 2,469)
- **Status in Phase 7:** Validation: 372 | Test: 371 | Train: 1,726 (Total: 2,469)

### Root Cause:
The discrepancy is caused by a **documented Phase 3 stratification upgrade** that occurred before Phase 5 training:
1. **Initial Split (Phase 2 / Early Phase 3):** An unstratified random patient shuffle was audited in [`results/splits/split_quality_report.md`](file:///c:/Users/Bcsm%20Computer/Downloads/Oral-lesion-xai/results/splits/split_quality_report.md), yielding **1,731 / 374 / 364**.
   - Under `problems_discovered`, that audit explicitly flagged: *"Split algorithm is patient-level but NOT class-aware... Extreme class imbalance... High validation instability risk for Oral Cancer (validation set contains only 2 images from 2 patients)."*
   - Recommendation: *"Before training, upgrade patient_level_split() to a patient-level stratified grouping..."*
2. **Locked Stratified Split (Phase 3):** `patient_level_stratified_split()` was implemented in `src/prepare_dataset.py` using **Deterministic Greedy Multi-Label Patient-Grouped Stratification (Seed 42)**.
   - This produced **1,726 Train / 372 Validation / 371 Test** images.
   - This was audited and approved in [`results/splits/stratified_split_quality_report.md`](file:///c:/Users/Bcsm%20Computer/Downloads/Oral-lesion-xai/results/splits/stratified_split_quality_report.md) and recorded in [`research_notes.md`](file:///c:/Users/Bcsm%20Computer/Downloads/Oral-lesion-xai/research_notes.md) under *Phase 3: Patient-Stratified Split (Completed & Locked)*.
   - The JSON partition files [`results/splits/train_split.json`](file:///c:/Users/Bcsm%20Computer/Downloads/Oral-lesion-xai/results/splits/train_split.json) (1,726), [`val_split.json`](file:///c:/Users/Bcsm%20Computer/Downloads/Oral-lesion-xai/results/splits/val_split.json) (372), and [`test_split.json`](file:///c:/Users/Bcsm%20Computer/Downloads/Oral-lesion-xai/results/splits/test_split.json) (371) were saved.
3. **Phase 5 & 6 Consistency:**
   - The models in `results/phase5/` (`EfficientNetB0` and `MobileNetV2`) were trained on `train_split.json` (1,726) and validated on `val_split.json` (372).
   - In Phase 6, [`results/phase6/phase6_report.md`](file:///c:/Users/Bcsm%20Computer/Downloads/Oral-lesion-xai/results/phase6/phase6_report.md) evaluated on `test_split.json` (371 images, 49 patients).
4. **Phase 7 Alignment:**
   - Phase 7 loaded `results/splits/val_split.json` (372) and `test_split.json` (371).
   - Phase 7 is therefore **100% mathematically and structurally aligned with Phase 5 and Phase 6**.

> [!WARNING]
> **Critical Data Leakage Hazard:**  
> If Phase 7 were retrofitted to evaluate the obsolete 364-image unstratified test split, **35 out of the 50 patients (70% of test patients) in the 364-image test set are present in the 1,726-image training set used to train the Phase 5 models**. Evaluating on the obsolete 364-image set would constitute massive train-test patient leakage.

---

## 2. Partition Comparison Table

| Metric | Obsolete Unstratified Split (`split_quality_report.md`) | Locked Stratified Split (`stratified_split_quality_report.md` & `*.json`) | Phase 5 Model Training | Phase 6 Comprehensive Eval | Phase 7 XAI Eval |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Images** | 1,731 | **1,726** | 1,726 | N/A | N/A |
| **Train Patients** | 230 | **232** | 232 | N/A | N/A |
| **Validation Images** | 374 | **372** | 372 | N/A | 372 (Threshold Calib) |
| **Validation Patients** | 49 | **48** | 48 | N/A | 48 |
| **Test Images** | 364 | **371** | 371 | 371 | **371 (Full Test)** |
| **Test Patients** | 50 | **49** | 49 | 49 | **49 (100% Match)** |
| **Total Images** | 2,469 | **2,469** | 2,469 | 2,469 | 2,469 |
| **Total Patients** | 329 | **329** | 329 | 329 | 329 |
| **Patient Leakage** | 0% | **0%** | 0% | 0% | **0%** |

---

## 3. Test Set Breakdown & Annotation Eligibility

The locked held-out test set (`results/splits/test_split.json`) comprises **371 images**:

| Diagnostic Class | Total Images in Test | Patients | Expert VIA Lesion Annotations Available | Annotation Eligibility |
| :--- | ---: | ---: | ---: | :--- |
| **Normal (`01. Normal`)** | 322 | 45 | 0 | **Ineligible for Lesion Localization** (Healthy tissue has no lesion boundary) |
| **Variation from normal (`02. Variation`)** | 27 | 14 | 27 | **Eligible** (100% annotated with VIA polygons) |
| **OPMD (`03. OPMD`)** | 19 | 10 | 19 | **Eligible** (100% annotated with VIA polygons) |
| **Oral Cancer (`04. Oral Cancer`)** | 3 | 3 | 3 | **Eligible** (100% annotated with VIA polygons) |
| **TOTAL** | **371** | **49** | **49** | **49 Annotation-Eligible Images** |

### Evaluation Protocol Distinction:
- **Full Test Set ($N = 371$ images, 49 patients):**
  - Used for classification accuracy, Macro F1, Brier score, Expected Calibration Error (ECE), and qualitative Grad-CAM/Grad-CAM++ heatmaps across all classes.
- **XAI Annotation-Eligible Subset ($N = 49$ images, 27 patients):**
  - Used for objective localization metrics against expert ground-truth polygon masks:
    - Intersection over Union (IoU)
    - Dice Similarity Coefficient
    - Pointing Game (maximum gradient activation hit/miss)
    - $2 \times 2$ Attribution/Prediction Contingency Matrix (Cell A, B, C, D)
- **Normal Tissue ($N = 322$ images):**
  - Normal mucosal images have no pathological lesions. Generating IoU against an empty lesion mask would be mathematically undefined ($\frac{0}{0}$); therefore, localization metrics are computed strictly on the 49 annotation-eligible lesion images, while confusion pair saliency (e.g. Normal attended to glare/saliva) is audited qualitatively across all 322 normal images.

---

## 4. Patient ID Verification

- **Locked Test Split Unique Patients:** 49 distinct patient IDs (`CA10`, `CASE_7252`, `SMITA00008`, `SMITA00009`, `SMITA00010`, `SMITA00021`, `SMITA00023`, `SMITA000252`, `SMITA00034`, ...).
- **Phase 7 Loaded Test Patients:** Exactly the same 49 distinct patient IDs.
- **Patient Overlap with Train:** **0 patients (0.00% leakage)**.
- **Patient Overlap with Validation:** **0 patients (0.00% leakage)**.
- **Patient Overlap between Phase 7 and Official `test_split.json`:** **100% (49 of 49 patients match identically)**.

---

## 5. Answers to Specific Audit Questions

1. **Exact number of images listed in each locked split:**
   - `results/splits/train_split.json`: **1,726 images**
   - `results/splits/val_split.json`: **372 images**
   - `results/splits/test_split.json`: **371 images**
   - Total: **2,469 images**
2. **Exact number actually loaded by Phase 7:**
   - Validation split: **372 images** (used for frozen threshold $\tau$ calibration in Step 4)
   - Test split: **371 images** (evaluated in Step 5)
3. **Which 2 validation images were excluded, if applicable:**
   - No individual images were dropped or excluded from a static list. The 374-image validation partition was an **obsolete unstratified allocation** (49 patients). Under the greedy multi-label stratification algorithm, 48 patients (372 images) were assigned to validation.
4. **Which 7 test images were added/used, if applicable:**
   - The 364-image test set was an **obsolete unstratified allocation** (50 patients). The stratified algorithm allocated 49 patients (371 images) to ensure rare classes (Oral Cancer and OPMD) were proportionally represented.
5. **Whether the difference is caused by missing files, unavailable annotations, filtering, filename normalization, duplicate handling, or a regenerated split:**
   - **Cause:** **A regenerated split** produced by the stratification upgrade in Phase 3 prior to training in Phase 5. No files are missing, no images were filtered, and no duplicates were dropped (all 2,469 raw photographic images and 329 patients are strictly preserved).

---

```text
SPLIT CONSISTENCY:
PASS WITH CONCERNS

OFFICIAL TEST IMAGES:
364 (Obsolete Unstratified Audit Reference) / 371 (Locked Stratified Test Split on Disk & in Phase 5/6)

PHASE 7 FULL TEST IMAGES:
371

ANNOTATION-ELIGIBLE IMAGES:
49 (3 Oral Cancer, 19 OPMD, 27 Variation from normal)

EXCLUDED IMAGES:
322 (Normal tissue images without lesion polygon annotations)

REASON:
Regenerated split during Phase 3 stratification upgrade. The 1,731/374/364 partition was an unstratified patient shuffle documented in split_quality_report.md. It was superseded in Phase 3 by deterministic greedy patient-stratification (stratified_split_quality_report.md) producing 1,726/372/371, which was locked and trained in Phase 5 and evaluated in Phase 6. Phase 7 evaluates the exact 371 locked test images from test_split.json.

PATIENT OVERLAP:
0 patient overlap between Train and Test (0% leakage). 100% patient match between Phase 7 and results/splits/test_split.json (49 of 49 patients).
```
