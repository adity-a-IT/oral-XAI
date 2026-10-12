# Stratified Patient-Level Split Quality Report: SMART-OM Oral Lesion Dataset

**Date:** 2026-09-29  
**Audit Target:** `results/splits/train_split.json`, `results/splits/val_split.json`, `results/splits/test_split.json`  
**Stratification Algorithm:** Deterministic Greedy Multi-Label Patient-Grouped Stratification  
**Random Seed:** `42` (Verified 100% reproducible across independent runs)  
**Core Constraint:** Images belonging to the same patient MUST NEVER appear in different partitions.

---

## 1. Dataset Summary

The unannotated raw photographic subset of the SMART-OM dataset contains **2,469 photographic images** from **329 unique patient/case identities**:

- **Normal (`01. Normal`)**: 2,145 images (**86.88%**)
- **Variation from normal (`02. Variation from normal`)**: 179 images (**7.25%**)
- **Oral Potentially Malignant Disorders (`03. OPMD`)**: 125 images (**5.06%**)
- **Oral Cancer (`04. Oral Cancer`)**: 20 images (**0.81%**) across 18 distinct clinical cases

---

## 2. Split Summary

Partitions were produced by grouping all images by `patient_id` and executing greedy multi-label stratification against nominal 70% / 15% / 15% targets.

| Split | Images | Image Ratio | Patients | Patient Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 1,726 | 69.91% | 232 | 70.52% |
| **Validation** | 372 | 15.07% | 48 | 14.59% |
| **Test** | 371 | 15.03% | 49 | 14.89% |
| **Total** | **2,469** | **100.0%** | **329** | **100.0%** |

---

## 3. Class Distribution & Deviation from Global Proportions

### Cross-Split Class Distribution Table

| Class | Train Images | Train Patients | Validation Images | Validation Patients | Test Images | Test Patients | Total Images |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 1,500 | 216 | 323 | 45 | 322 | 45 | 2,145 |
| Variation from normal | 125 | 81 | 27 | 16 | 27 | 14 | 179 |
| OPMD | 87 | 45 | 19 | 9 | 19 | 10 | 125 |
| Oral Cancer | 14 | 13 | 3 | 2 | 3 | 3 | 20 |
| **TOTAL** | **1,726** | **232** | **372** | **48** | **371** | **49** | **2,469** |

*(Note: 142 patients contribute multi-class photographic records across different oral sites. Consequently, per-class patient counts sum to more than the unique patient count of each split. The TOTAL row reflects true disjoint patient counts).*

### Class Proportions & Deviation from Global Dataset

- **Normal (Global: 86.88%):**
  - Train: 1,500 images (**86.91%**, deviation: **+0.03%**)
  - Validation: 323 images (**86.83%**, deviation: **-0.05%**)
  - Test: 322 images (**86.79%**, deviation: **-0.08%**)
- **Variation from normal (Global: 7.25%):**
  - Train: 125 images (**7.24%**, deviation: **-0.01%**)
  - Validation: 27 images (**7.26%**, deviation: **+0.01%**)
  - Test: 27 images (**7.28%**, deviation: **+0.03%**)
- **OPMD (Global: 5.06%):**
  - Train: 87 images (**5.04%**, deviation: **-0.02%**)
  - Validation: 19 images (**5.11%**, deviation: **+0.04%**)
  - Test: 19 images (**5.12%**, deviation: **+0.06%**)
- **Oral Cancer (Global: 0.81%):**
  - Train: 14 images (**0.81%**, deviation: **+0.00%**)
  - Validation: 3 images (**0.81%**, deviation: **-0.00%**)
  - Test: 3 images (**0.81%**, deviation: **-0.00%**)

**Max deviation across any class in any split is $\le 0.08\%$.**

---

## 4. Patient-Level Leakage Verification

Patient ID sets were extracted from `train_split.json`, `val_split.json`, and `test_split.json` and pairwise intersections evaluated:

```text
Patient overlap:
Train ↔ Validation: 0 / expected 0
Train ↔ Test:       0 / expected 0
Validation ↔ Test:   0 / expected 0
```

- **Train ∩ Validation:** 0 patients ($\emptyset$)
- **Train ∩ Test:** 0 patients ($\emptyset$)
- **Validation ∩ Test:** 0 patients ($\emptyset$)
- **Total Unique Patients Accounted For:** Exactly 329 patients.

**100% patient isolation confirmed. Zero patient leakage.**

---

## 5. Critical Minority-Class Audit: Oral Cancer (OSCC)

### Dataset Physical Reality
There are only **20 Oral Cancer images** from **18 patients/cases** in the entire SMART-OM repository. No patient grouping algorithm can fabricate statistical power when the raw sample count is 18 cases.

### Allocation Breakdown:
- **Total Patients:** 18
- **Total Images:** 20
- **Train Split:** **14 images** from **13 patients** (`CA1`, `CA2`, `CA3`, `CA5`, `CA6`, `CA7`, `CA8`, `CA9`, `CASE_6026`, `CASE_6095`, `CASE_6098`, `CASE_6765`, `CASE_7059`)
- **Validation Split:** **3 images** from **2 patients** (`CA4` [2 images], `CASE_6766` [1 image])
- **Test Split:** **3 images** from **3 patients** (`CA10` [1 image], `CASE_6099` [1 image], `CASE_7252` [1 image])

### Statistical Power & Stability Assessment:
1. **Mathematical Optimum:** The split achieves an exact 70.0% / 15.0% / 15.0% image partition (14 / 3 / 3 images) while maintaining complete patient integrity.
2. **Statistical Reality:** With only 3 test images and 3 validation images, each individual prediction carries a massive **33.3% swing** ($1/3 = 0.333$) on class recall.
3. **Conclusion on 4-Class OSCC Evaluation:** While this split represents the mathematical optimum possible under a 70/15/15 patient-level design, independent 4-class OSCC validation remains statistically noisy.

---

## 6. Verification Checklist Results

| Check # | Verification Requirement | Result | Details |
| :---: | :--- | :---: | :--- |
| **1** | Total image count = 2,469 | **PASS** | 1,726 + 372 + 371 = 2,469 |
| **2** | Every image appears exactly once | **PASS** | 2,469 unique filepaths |
| **3** | No patient appears in >1 split | **PASS** | Train ∩ Val = 0, Train ∩ Test = 0, Val ∩ Test = 0 |
| **4** | Every class appears in all splits | **PASS** | All 4 classes present in Train, Val, and Test |
| **5** | Class counts & percentages reported | **PASS** | Documented above |
| **6** | Patient counts reported | **PASS** | 232 Train, 48 Val, 49 Test (Total 329) |
| **7** | Class deviation from global dataset | **PASS** | Maximum deviation across all splits is 0.08% |
| **8** | Fixed seed = 42 confirmed | **PASS** | Deterministic tie-breaking verified |
| **9** | Deterministic reproducibility (2 runs) | **PASS** | Run 1 and Run 2 produced 100% identical outputs |

---

## Conclusion & Recommendations

```text
SPLIT STATUS:
PASS WITH CONCERNS

PATIENT LEAKAGE:
Zero patient overlap verified across all partitions:
Train ↔ Validation: 0
Train ↔ Test:       0
Validation ↔ Test:   0
All 2,469 images belonging to all 329 patients are strictly confined to a single partition.

CLASS DISTRIBUTION:
Proportions match the global dataset distribution almost perfectly:
- Normal: 86.91% Train, 86.83% Val, 86.79% Test (Global: 86.88%)
- Variation from normal: 7.24% Train, 7.26% Val, 7.28% Test (Global: 7.25%)
- OPMD: 5.04% Train, 5.11% Val, 5.12% Test (Global: 5.06%)
- Oral Cancer: 0.81% Train, 0.81% Val, 0.81% Test (Global: 0.81%)
Maximum deviation across any class in any split is 0.08%.

ORAL CANCER DISTRIBUTION:
Total in dataset: 20 images from 18 patients/cases.
- Train: 14 images (13 patients)
- Validation: 3 images (2 patients)
- Test: 3 images (3 patients)
The allocation is mathematically optimal for 70/15/15, but having only 3 validation images and 3 test images cannot provide high statistical power (each prediction carries a 33.3% sensitivity weight).

RECOMMENDATION:
1. Accept this patient-stratified split as the gold-standard partition for the repository, replacing the previous unstratified split.
2. In model training (Phase 5), apply class-weighting (inverse class frequency or effective sample weighting) to prevent the 86.9% Normal class from drowning the gradient signal of OPMD and Oral Cancer.
3. For evaluation reporting (Phase 8), report both:
   a) 4-class per-class metrics (with explicit notation of sample sizes: N=322, N=27, N=19, N=3), and
   b) 3-class clinically consolidated metrics (Healthy [Normal + Variation] vs OPMD vs OSCC) or Binary screening metrics (Benign vs Potentially Malignant/Malignant) to ensure statistically robust, peer-reviewable performance claims.
```
