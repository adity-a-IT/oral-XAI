# Preprocessing & Class-Imbalance Handling Report (Phase 4)

**Status:** PASS  
**Random Seed:** `42`  
**Dataset Target:** SMART-OM Oral Lesion Dataset (Locked Patient-Stratified Split)

---

## 1. Dataset Partition Summary

- **Train Set:** 1,726 images (232 patients)
- **Validation Set:** 372 images (48 patients)
- **Test Set:** 371 images (49 patients)
- **Total Images:** 2,469 images
- **Patient Isolation:** **0 patient overlap** across all partitions.
- **Image Isolation:** **0 image overlap** across all partitions.
- **File Integrity:** 0 malformed or corrupt images detected.

---

## 2. Calculated Balanced Class Weights (Train Only)

Calculated strictly on the training partition ($N = 1726$) using the formula:
$$\text{weight}[c] = \frac{N_{\text{train}}}{C \times N_c}$$

| Class Index | Class Diagnostic Name | Source Folder | Train Samples | Train % | Balanced Weight |
| :---: | :--- | :--- | ---: | ---: | ---: |
| **0** | Normal | `01. Normal` | 1,500 | 86.91% | **0.2877** |
| **1** | Variation from normal | `02. Variation from normal` | 125 | 7.24% | **3.4520** |
| **2** | OPMD | `03. OPMD` | 87 | 5.04% | **4.9598** |
| **3** | Oral Cancer | `04. Oral Cancer` | 14 | 0.81% | **30.8214** |

*Verification:* Validation and test partitions were strictly excluded from class weight calculation.

---

## 3. Preprocessing Specifications

- **Target Input Resolution:** `224 x 224` pixels
- **Color Format:** 3-channel RGB
- **Normalization:** `rescaling` (scaled to `[0.0, 1.0]`)
- **CLAHE (Contrast Enhancement):** `Disabled (configurable)`

---

## 4. Medical-Image-Safe Training Augmentation

To preserve subtle lesion borders, mucosal texture, and diagnostic validity, training augmentation is kept strictly within conservative clinical thresholds:

- **Horizontal Flip:** `True` (clinically safe due to bilateral symmetry of the oral cavity)
- **Vertical Flip:** `False` (**strictly disabled** to prevent anatomical inversion of palate/floor of mouth)
- **Rotation:** `[-10.0°, +10.0°]` (mild rotation)
- **Translation:** `+/- 5.0%` (small shifts)
- **Zoom / Scaling:** `[0.95, 1.05]` (mild zoom +/- 5%)
- **Brightness Variation:** `+/- 8.0%`
- **Contrast Variation:** `[0.92, 1.08]`

---

## 5. Validation & Test Preprocessing Safeguards

- **Augmentation:** **Strictly Disabled (0% random perturbation)**
- **Pipeline:** Deterministic bilinear resize $\rightarrow$ RGB conversion $\rightarrow$ Normalization.
- **Evaluation Guarantee:** 100% deterministic reproducibility for model evaluation and XAI attribution.

---

## 6. Verification Checklist

| Verification Requirement | Result | Evidence |
| :--- | :---: | :--- |
| Every split contains only valid images | **PASS** | 0 corrupt files |
| Exactly one label per image | **PASS** | 100% records conform to 4 classes |
| Zero image leakage | **PASS** | Train $\cap$ Val = 0, Train $\cap$ Test = 0, Val $\cap$ Test = 0 |
| Zero patient leakage | **PASS** | Disjoint patient partitions verified |
| Validation / Test augmentation disabled | **PASS** | Strictly enforced in pipeline |
| Class weights from TRAIN only | **PASS** | 1,726 train records used exclusively |

---

## Conclusion

```text
PHASE 4 STATUS:
PASS

CLASS WEIGHTS:
- Normal (0): 0.2877
- Variation from normal (1): 3.4520
- OPMD (2): 4.9598
- Oral Cancer (3): 30.8214

AUGMENTATION:
- Horizontal flip: True (anatomically safe)
- Vertical flip: False (anatomically disallowed)
- Rotation: +/- 10.0 degrees
- Translation: +/- 5.0%
- Zoom: 0.95 to 1.05
- Brightness: +/- 8.0%
- Contrast: 0.92 to 1.08
- Validation/Test augmentation: Strictly Disabled

INPUT SIZE:
(224, 224, 3)

NORMALIZATION:
Rescaling [0.0, 1.0] (Configurable to standard [-1, 1] or ImageNet z-score)

LEAKAGE CHECK:
Zero patient leakage (0 patient overlap across all splits)
Zero image leakage (0 image overlap across all splits)

NEXT STEP:
Phase 5 — baseline model training
```
