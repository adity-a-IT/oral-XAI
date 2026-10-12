# Phase 5: Baseline Model Training Report

**Status:** PASS WITH CONCERNS  
**Best Validation Architecture:** `EfficientNetB0`  
**Selection Criterion:** Highest Validation Macro F1 (**0.4709**)  
**Random Seed:** `42`  

---

## 1. Experiment Overview & Architecture Design

Three distinct models were trained independently using identical dataset splits, class weights, and evaluation protocols:

1. **Custom Oral CNN:** Lightweight baseline (304,932 parameters) trained from scratch with medical-safe augmentation.
2. **MobileNetV2:** ImageNet transfer learning (2,263,108 parameters) with frozen backbone feature extraction + controlled fine-tuning.
3. **EfficientNetB0:** ImageNet transfer learning (4,054,695 parameters) with compound-scaled convolutional backbone + controlled fine-tuning.

---

## 2. Model Comparison Summary

# Phase 5: Baseline Model Comparison Table

| Model | Total Params | Trainable Params | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Test Weighted F1 | Test Cohen's Kappa |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Custom CNN** | 304,932 | 304,484 | 86.83% | **0.2324** | 86.79% | **0.2323** | 0.8066 | 0.0000 |
| **MobileNetV2** | 2,263,108 | 1,057,604 | 70.97% | **0.4624** | 77.90% | **0.5541** | 0.8199 | 0.3809 |
| **EfficientNetB0** | 4,054,695 | 1,134,900 | 71.24% | **0.4709** | 72.78% | **0.5346** | 0.7805 | 0.3324 |

**Selected Best Validation Model:** `EfficientNetB0` (Selected strictly by Validation Macro F1 to counteract class imbalance).


---

## 3. Detailed Final Test Evaluation: Best Model (EfficientNetB0)

Evaluated once on the locked 371-image test set:

- **Overall Test Accuracy:** 72.78%
- **Macro Precision:** 0.4947
- **Macro Recall:** 0.6792
- **Macro F1-Score:** 0.5346
- **Weighted F1-Score:** 0.7805
- **Cohen's Kappa:** 0.3324

### Per-Class Test Performance:

| Class Diagnostic Category | Support | Precision | Recall | F1-Score | Statistical Reliability |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Normal** | 322 | 0.9795 | 0.7422 | 0.8445 | Robust (N=322) |
| **Variation from normal** | 27 | 0.1867 | 0.5185 | 0.2745 | Moderate (N=27) |
| **OPMD** | 19 | 0.3125 | 0.7895 | 0.4478 | Moderate (N=19) |
| **Oral Cancer (OSCC)** | 3 | 0.5000 | 0.6667 | 0.5714 | **LOW STATISTICAL POWER (N=3)** |

---

## 4. Critical OSCC Sample Size Limitation

> [!WARNING]
> The test set contains only **3 OSCC images** (due to the physical dataset limitation of 20 total images in SMART-OM). A single error shifts OSCC test recall by **33.3 percentage points**. Consequently, while OSCC metrics are reported with full academic transparency, no claim of clinical statistical robustness can be asserted for independent 4-class OSCC sensitivity without external validation cohorts.

---

## Conclusion

```text
PHASE 5 STATUS:
PASS WITH CONCERNS

MODELS TRAINED:
1. Custom CNN (Params: 304,932)
2. MobileNetV2 (Params: 2,263,108)
3. EfficientNetB0 (Params: 4,054,695)

BEST VALIDATION MODEL:
EfficientNetB0

VALIDATION MACRO F1:
0.4709

FINAL TEST RESULTS:
- Accuracy: 72.78%
- Macro F1: 0.5346
- Weighted F1: 0.7805

OSCC TEST SUPPORT:
3 images

OSCC METRIC LIMITATION:
Each sample constitutes 33.3% of class recall; metrics are documented as low-statistical-power estimates.

NEXT STEP:
Phase 6 — final model selection + detailed evaluation
```
