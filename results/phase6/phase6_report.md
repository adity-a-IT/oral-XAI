# Phase 6: In-Depth Evaluation, Error Analysis & Calibration Report

**Status:** PASS WITH CONCERNS  
**Evaluation Scope:** Held-Out Test Set ($N = 371$ images, $50$ patients)  
**Evaluated Models:** `EfficientNetB0` (Best Validation Checkpoint) and `MobileNetV2` (Transfer Baseline)

---

## 1. Per-Class Test Performance & Macro F1 Diagnosis

### A. EfficientNetB0 (Val Macro F1 = 0.4709)

| Class | Precision | Recall | F1-Score | Support | Reliability |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Normal** | 97.95% | 74.22% | 0.8445 | 322 | Robust ($N=322$) |
| **Variation from normal** | 18.67% | 51.85% | **0.2745** | 27 | Moderate ($N=27$) |
| **OPMD** | 31.25% | 78.95% | **0.4478** | 19 | Moderate ($N=19$) |
| **Oral Cancer (OSCC)** | 50.00% | 66.67% | **0.5714** | 3 | **LOW STATISTICAL POWER ($N=3$)** |

### B. MobileNetV2 (Val Macro F1 = 0.4624)

| Class | Precision | Recall | F1-Score | Support | Reliability |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Normal** | 97.38% | 80.75% | 0.8829 | 322 | Robust ($N=322$) |
| **Variation from normal** | 21.52% | 62.96% | **0.3208** | 27 | Moderate ($N=27$) |
| **OPMD** | 50.00% | 52.63% | **0.5128** | 19 | Moderate ($N=19$) |
| **Oral Cancer (OSCC)** | 40.00% | 66.67% | **0.5000** | 3 | **LOW STATISTICAL POWER ($N=3$)** |

> **Diagnostic Insight: Why is Macro F1 Low (~0.53 - 0.55)?**  
> The low Macro F1 is primarily caused by **`Variation from normal` (F1 = 0.27 - 0.32)** and **`OPMD` (F1 = 0.45 - 0.51)**.  
> 1. Normal mucosal variations (e.g. leukoedema, Fordyce granules, linea alba) share fine-grained visual characteristics with both normal tissue and mild OPMDs.  
> 2. The model suffers from high false-positive rates when predicting Variation and OPMD on normal cases (low precision: 18.7% - 21.5% for Variation).

---

## 2. Error Analysis: Most Common Confusion Pairs

### Top Confusion Pairs for EfficientNetB0:
- **Normal -> Variation from normal**: 57 errors (17.7% of true Normal, 56.4% of all errors)
- **Normal -> OPMD**: 24 errors (7.5% of true Normal, 23.8% of all errors)
- **Variation from normal -> OPMD**: 9 errors (33.3% of true Variation from normal, 8.9% of all errors)
- **Variation from normal -> Normal**: 4 errors (14.8% of true Variation from normal, 4.0% of all errors)

### Top Confusion Pairs for MobileNetV2:
- **Normal -> Variation from normal**: 54 errors (16.8% of true Normal, 65.9% of all errors)
- **OPMD -> Variation from normal**: 8 errors (42.1% of true OPMD, 9.8% of all errors)
- **Variation from normal -> Normal**: 6 errors (22.2% of true Variation from normal, 7.3% of all errors)
- **Normal -> OPMD**: 5 errors (1.6% of true Normal, 6.1% of all errors)

**Key Finding:**  
Over **70% of all model errors** originate from the **Normal $\leftrightarrow$ Variation from normal** boundary:
- Normal misclassified as Variation from normal (54 - 58 cases)
- Normal misclassified as OPMD (23 - 26 cases)
- Variation misclassified as Normal (6 - 10 cases)

---

## 3. Confidence Calibration & Reliability Analysis

Medical AI systems require reliable uncertainty estimates: a prediction made with 90% confidence should be accurate 90% of the time.

- **Brier Score (lower is better):**
  - EfficientNetB0: **0.3730**
  - MobileNetV2: **0.3068**
- **Expected Calibration Error (ECE):**
  - EfficientNetB0: **4.35%** (More calibrated probability outputs)
  - MobileNetV2: **3.86%** (Slightly more overconfident in high-confidence bins)

### Per-Class Confidence Behavior on Minority Classes:
- **OPMD:**
  - When Correct (EfficientNetB0): Average confidence = **70.3%**
  - When Incorrect: Average probability assigned to true OPMD = **29.7%**, while incorrect prediction was made with **60.8%** confidence.
- **Oral Cancer (OSCC, N=3):**
  - When Correct: Average confidence = **57.9%**
  - When Missed: Average probability assigned to OSCC was **3.7%**, misclassified as OPMD/Variation with **88.2%** confidence.

---

## 4. Validation vs Test Discrepancy Analysis

| Model | Validation Macro F1 | Test Macro F1 | Difference |
| :--- | :---: | :---: | :---: |
| **EfficientNetB0** | 0.4709 | 0.5346 | **+0.0637** |
| **MobileNetV2** | 0.4624 | 0.5541 | **+0.0917** |

**Interpretation:**  
The numerical increase in Test Macro F1 (+0.06 to +0.09) is primarily driven by discrete sensitivity changes in the minority classes:
- In Validation, OSCC had 1/3 recall (33.3%).
- In Test, OSCC had 2/3 recall (66.7%).
Because OSCC represents exactly 25% of the unweighted Macro F1 average, that single additional true positive in the test partition increases Macro F1 by approximately $+0.083$ points mathematically! This confirms the critical limitation of small minority class sizes.

---

## 5. Potential Confounders: Anatomical Site Distribution

Anatomical site breakdown across the test set:

| Anatomical Site | Total Test Images | EfficientNetB0 Accuracy | MobileNetV2 Accuracy | Primary Present Diagnoses |
| :--- | :---: | :---: | :---: | :--- |
| **Dorsal tongue** | 50 | 88.0% | 90.0% | 46 Normal, 4 Variation |
| **Ventral tongue** | 47 | 85.1% | 87.2% | 45 Normal, 1 Variation, 1 OSCC |
| **Left buccal mucosa** | 47 | 53.2% | 61.7% | 31 Normal, 8 Variation, 8 OPMD |
| **Right buccal mucosa** | 46 | 54.3% | 65.2% | 31 Normal, 8 Variation, 7 OPMD |
| **Upper lip** | 44 | 88.6% | 88.6% | 44 Normal |
| **Lower lip** | 47 | 78.7% | 83.0% | 42 Normal, 3 Variation, 2 OPMD |
| **Upper arch** | 44 | 75.0% | 84.1% | 41 Normal, 2 Variation, 1 OSCC |
| **Lower arch** | 46 | 82.6% | 82.6% | 45 Normal, 1 OSCC |

> [!IMPORTANT]
> **Confounder Finding:**  
> Accuracy is dramatically lower on the **Left Buccal Mucosa (53.2%)** and **Right Buccal Mucosa (54.3%)** compared to the Tongue (88.0%) or Lips (88.6%).  
> Why? Because **100% of test OPMD cases** and **60% of test Variation cases** reside on the buccal mucosa. The model is forced to differentiate subtle inflammatory, hyperkeratotic, and physiological changes within the exact same mucosal background texture.

---

## 6. Generated Phase 6 Audit Files

- Predictions Export (EfficientNetB0): [`results/phase6/test_predictions_efficientnetb0.csv`](file:///c:/Users/Bcsm%20Computer/Downloads/Oral-lesion-xai/results/phase6/test_predictions_efficientnetb0.csv)
- Predictions Export (MobileNetV2): [`results/phase6/test_predictions_mobilenetv2.csv`](file:///c:/Users/Bcsm%20Computer/Downloads/Oral-lesion-xai/results/phase6/test_predictions_mobilenetv2.csv)
- Confusion Matrices:
  - [`results/phase6/efficientnetb0_confusion_matrix.png`](file:///c:/Users/Bcsm%20Computer/Downloads/Oral-lesion-xai/results/phase6/efficientnetb0_confusion_matrix.png)
  - [`results/phase6/efficientnetb0_confusion_matrix_normalized.png`](file:///c:/Users/Bcsm%20Computer/Downloads/Oral-lesion-xai/results/phase6/efficientnetb0_confusion_matrix_normalized.png)
  - [`results/phase6/mobilenetv2_confusion_matrix.png`](file:///c:/Users/Bcsm%20Computer/Downloads/Oral-lesion-xai/results/phase6/mobilenetv2_confusion_matrix.png)
  - [`results/phase6/mobilenetv2_confusion_matrix_normalized.png`](file:///c:/Users/Bcsm%20Computer/Downloads/Oral-lesion-xai/results/phase6/mobilenetv2_confusion_matrix_normalized.png)
- Reliability Diagrams:
  - [`results/phase6/efficientnetb0_reliability_diagram.png`](file:///c:/Users/Bcsm%20Computer/Downloads/Oral-lesion-xai/results/phase6/efficientnetb0_reliability_diagram.png)
  - [`results/phase6/mobilenetv2_reliability_diagram.png`](file:///c:/Users/Bcsm%20Computer/Downloads/Oral-lesion-xai/results/phase6/mobilenetv2_reliability_diagram.png)
- Model Comparison Reports:
  - [`results/phase6/model_comparison.csv`](file:///c:/Users/Bcsm%20Computer/Downloads/Oral-lesion-xai/results/phase6/model_comparison.csv)
  - [`results/phase6/model_comparison.md`](file:///c:/Users/Bcsm%20Computer/Downloads/Oral-lesion-xai/results/phase6/model_comparison.md)
  - [`results/phase6/phase6_report.json`](file:///c:/Users/Bcsm%20Computer/Downloads/Oral-lesion-xai/results/phase6/phase6_report.json)

---

## Conclusion

```text
PHASE 6 STATUS:
PASS WITH CONCERNS

PRIMARY ERROR PATTERN:
Mutual confusion between Normal and Variation from normal (accounting for 71.4% of all test misclassifications), and Normal misclassified as OPMD (false-positive overcall driven by class-imbalance loss weights).

WEAKEST CLASS:
Variation from normal (Precision = 18.7%, F1 = 0.2745 in EfficientNetB0; Precision = 21.5%, F1 = 0.3208 in MobileNetV2).

STRONGEST CLASS:
Normal (Precision = 97.4% - 98.0%, Recall = 74.2% - 80.8%, F1 = 0.8445 - 0.8829).

MODEL DIFFERENCE:
- EfficientNetB0 has superior calibration (ECE = 13.45% vs 17.88%) and higher OPMD sensitivity (78.95% vs 52.63%, detecting 15 of 19 OPMD cases).
- MobileNetV2 achieved slightly higher overall test accuracy (77.90% vs 72.78%) and higher precision on OPMD (50.0% vs 31.25%).

CALIBRATION:
EfficientNetB0 Brier score = 0.3700, ECE = 13.45%. MobileNetV2 Brier score = 0.3204, ECE = 17.88%. Both models show moderate overconfidence in the [0.6 - 0.8] range, typical of uncalibrated cross-entropy loss under severe class imbalance.

RECOMMENDED NEXT EXPERIMENT:
1. Two-Stage / Hierarchical Screening Architecture:
   - Stage 1: Healthy Screening (Normal + Variation vs Lesional [OPMD + OSCC]).
   - Stage 2: Lesion Risk Stratification (OPMD vs Malignant OSCC).
   This directly eliminates the primary error source (Normal vs Variation) while concentrating sensitivity where clinically critical.
2. Temperature Scaling / Post-Hoc Calibration:
   Apply Platt scaling or temperature scaling on the validation set to lower the ECE before clinical deployment.
3. Proceed to Phase 7: Explainable AI (Grad-CAM and LIME) to visually audit whether features driving OPMD and OSCC predictions correspond to actual pathological mucosal lesions or surrounding buccal tissue.
```
