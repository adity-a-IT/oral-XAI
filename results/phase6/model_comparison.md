# Phase 6: Model Comparison & Error Evaluation

| Metric / Dimension | EfficientNetB0 (Best Val Model) | MobileNetV2 (Alternative Baseline) |
| :--- | :---: | :---: |
| **Total Parameters** | 4,054,695 | 2,263,108 |
| **Validation Macro F1** | **0.4709** | 0.4624 |
| **Test Accuracy** | 72.78% | **77.90%** |
| **Test Macro F1** | 0.5346 | **0.5541** |
| **Test Weighted F1** | 0.7805 | **0.8199** |
| **Cohen's Kappa** | 0.3324 | **0.3809** |
| **Brier Score (Lower is better)** | **0.3700** | 0.3204 |
| **Expected Calibration Error (ECE)** | **0.1345** | 0.1788 |
| **Normal Recall (N=322)** | 74.22% | **80.75%** |
| **Variation from normal Recall (N=27)** | 51.85% | **62.96%** |
| **OPMD Recall (N=19)** | **78.95%** (15/19) | 52.63% (10/19) |
| **Oral Cancer Recall (N=3)\*** | **66.67%** (2/3) | **66.67%** (2/3) |
| **OPMD Precision** | 31.25% | **50.00%** |
| **Oral Cancer Precision\*** | **50.00%** | 40.00% |

*\*OSCC sample size is N=3 test images; metrics are low-statistical-power estimates.*
