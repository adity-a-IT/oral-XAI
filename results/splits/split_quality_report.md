# Split Quality & Leakage Audit Report: SMART-OM Oral Lesion Dataset

**Date:** 2026-09-29  
**Audit Target:** `results/splits/train_split.json`, `results/splits/val_split.json`, `results/splits/test_split.json`  
**Dataset Root:** `dataset/oral-lesion-dataset-org/SMART-OM`  
**Objective:** Strict pre-training verification of patient isolation, class representation, statistical power, and split methodology.

---

## 1. Dataset Summary

The unannotated, raw photographic subset of the SMART-OM dataset comprises **2,469 photographic images** acquired from **329 unique patient / clinical cases**.

- **Normal (`01. Normal`)**: 2,145 images (86.88% of dataset) across 8 oral mucosal sites.
- **Variation from normal (`02. Variation from normal`)**: 179 images (7.25% of dataset).
- **Oral Potentially Malignant Disorders (`03. OPMD`)**: 125 images (5.06% of dataset).
- **Oral Cancer (`04. Oral Cancer`)**: 20 images (0.81% of dataset) from 18 distinct clinical cases.

---

## 2. Split Summary

Splits were generated using a patient-level partitioning algorithm targeting nominal 70% / 15% / 15% patient partitions.

| Split | Images | Image % | Patients | Patient % |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 1,731 | 70.11% | 230 | 69.91% |
| **Validation** | 374 | 15.15% | 49 | 14.89% |
| **Test** | 364 | 14.74% | 50 | 15.20% |
| **Total** | **2,469** | **100.0%** | **329** | **100.0%** |

---

## 3. Class Distribution

### Cross-Split Class Distribution Table

| Class | Train Images | Train Patients | Validation Images | Validation Patients | Test Images | Test Patients | Total Images |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 1,503 | 216 | 322 | 46 | 320 | 44 | 2,145 |
| Variation from normal | 134 | 83 | 19 | 12 | 26 | 16 | 179 |
| OPMD | 81 | 42 | 31 | 14 | 13 | 8 | 125 |
| Oral Cancer | 13 | 11 | 2 | 2 | 5 | 5 | 20 |
| **TOTAL** | **1,731** | **230** | **374** | **49** | **364** | **50** | **2,469** |

*Note on patient totals:* Individual patients frequently contribute multiple images spanning different anatomical areas and mucosal conditions (142 patients have multi-class image instances). Therefore, the sum of per-class patient counts exceeds the unique patient count of each split. The TOTAL row reflects the true distinct patient count per split.

### Within-Split Class Proportions

- **Train Set (1,731 images):**
  - Normal: 1,503 images (**86.83%**)
  - Variation from normal: 134 images (**7.74%**)
  - OPMD: 81 images (**4.68%**)
  - Oral Cancer: 13 images (**0.75%**)
- **Validation Set (374 images):**
  - Normal: 322 images (**86.10%**)
  - Variation from normal: 19 images (**5.08%**)
  - OPMD: 31 images (**8.29%**)
  - Oral Cancer: 2 images (**0.53%**)
- **Test Set (364 images):**
  - Normal: 320 images (**87.91%**)
  - Variation from normal: 26 images (**7.14%**)
  - OPMD: 13 images (**3.57%**)
  - Oral Cancer: 5 images (**1.37%**)

---

## 4. Patient Distribution & Case Allocation

Detailed patient/case tracking for the rarest and most critical class (**Oral Cancer**, 18 patients, 20 images):

- **Train (11 patients, 13 images):** `CA1`, `CA10`, `CA2`, `CA4`, `CA5`, `CA6`, `CA7`, `CA9`, `CASE_6026`, `CASE_6095`, `CASE_7252`
- **Validation (2 patients, 2 images):** `CA8`, `CASE_6765`
- **Test (5 patients, 5 images):** `CA3`, `CASE_6098`, `CASE_6099`, `CASE_6766`, `CASE_7059`

---

## 5. Leakage Test (Patient Overlap Verification)

Patient ID sets were extracted from each split JSON file and pairwise intersections were evaluated:

```text
Patient overlap:
Train ↔ Validation: 0 / expected 0
Train ↔ Test:       0 / expected 0
Validation ↔ Test:   0 / expected 0
```

- **Train ∩ Validation:** 0 patients (Empty set $\emptyset$)
- **Train ∩ Test:** 0 patients (Empty set $\emptyset$)
- **Validation ∩ Test:** 0 patients (Empty set $\emptyset$)

**Result:** Zero patient leakage confirmed. All images from any individual patient reside strictly within a single partition.

---

## 6. Class Imbalance Analysis

1. **Extreme Majority Dominance:**
   - Normal mucosal images make up 86.88% of all images.
   - A naive majority-class classifier predicting "Normal" for every test image would achieve **87.91% raw accuracy** on the test set while failing completely on all malignant lesions.
2. **Extreme Minority Scarcity:**
   - Oral Cancer constitutes merely 0.81% of the entire cohort (1:107 ratio relative to Normal).
   - In the validation set, Oral Cancer represents only 0.53% (2 images out of 374).

---

## 7. Problems Discovered

1. **Severe Statistical Instability in Validation Set:**
   - The validation set contains only **2 Oral Cancer images** (from 2 patients: `CA8` and `CASE_6765`).
   - If a model misclassifies a single Oral Cancer image during validation, the validation recall drops instantly by 50 percentage points (from 100% to 50% or 0%).
   - Monitoring `val_loss` or early stopping on validation accuracy will suffer extreme variance and noisy checkpoint saving.
2. **High Estimation Variance in Test Evaluation:**
   - The test set contains only **5 Oral Cancer images**.
   - A single misclassification shifts cancer sensitivity/recall by 20% ($1/5 = 0.20$).
   - The 95% binomial confidence interval for 4/5 correct predictions spans from 37.5% to 96.4%, severely limiting statistical significance for scientific publication.
3. **Split Algorithm is NOT Class-Aware (Unstratified Patient Shuffle):**
   - Inspection of `prepare_dataset.py` (`patient_level_split`) confirms that the algorithm groups records by `patient_id` and executes an unstratified random shuffle (`rng.shuffle(unique_patients)` with `seed=42`).
   - The fact that all four classes appeared in Train, Val, and Test was fortunate due to random seed 42. With another seed, the 18 Oral Cancer patients could easily have landed entirely in Train or resulted in 0 samples in Validation.
   - The split algorithm lacks patient-level multi-label stratification (`StratifiedGroupKFold` or greedy multi-class patient assignment).

---

## 8. Recommendation: Split Viability for Training

- **Can this split be used for preliminary prototype verification?**
  Yes, because it strictly enforces patient separation (0 leakage) and all four classes are present in every partition.
- **Should this split be used as-is for the final research benchmark?**
  No. Before production model training, the partitioning logic should be upgraded to **patient-level stratified splitting** to distribute rare clinical classes more systematically, and the 4-class vs 3-class framing (Normal/Variation vs OPMD vs OSCC) must be formalized.

---

## Conclusion

```text
SPLIT STATUS:
PASS WITH CONCERNS

REASON:
The split achieves 100% patient isolation with zero train-validation-test leakage, and all four diagnostic classes are represented across all three partitions. However, significant concerns exist:
1. Severe class imbalance (Normal = 86.88% vs Oral Cancer = 0.81%).
2. High validation instability: only 2 Oral Cancer images exist in the validation set, creating extreme volatility in validation metrics.
3. Low statistical power: only 5 Oral Cancer images exist in the test set, leading to excessively wide confidence intervals (+/-20% per image).
4. The split generator is purely patient-level and not class-stratified; representation of rare classes occurred by chance of seed=42 rather than by stratified design.

NEXT RECOMMENDED STEP:
1. Formalize the clinical task framing: determine whether the study uses 4 classes (Normal, Variation, OPMD, Cancer) or clinical 3-class consolidation (Healthy [Normal + Variation], OPMD, OSCC).
2. Upgrade patient_level_split() to a patient-level stratified algorithm (StratifiedGroupKFold or greedy patient-label stratification) to optimize minority class distribution.
3. Design a class-balanced loss weighting strategy (e.g. effective number of samples / focal loss / inverse frequency weights) to prevent the 86.9% Normal class from overwhelming network gradients during training.
```
