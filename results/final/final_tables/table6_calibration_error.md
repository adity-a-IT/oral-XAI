# Table 6: Calibration & Error Distribution Analysis

| Metric | EfficientNetB0 | MobileNetV2 | Methodological Interpretation |
|:---|:---:|:---:|:---|
| **Expected Calibration Error (ECE)** | 4.35% | 3.86% | Probability estimates match empirical diagnostic accuracy within ~4%. |
| **Brier Score (MSE)** | 0.3730 | 0.3068 | Overall squared probability error across 4-class vectors (lower is superior). |
| **Total Test Misclassifications** | 101 / 371 (27.2%) | 82 / 371 (22.1%) | Overall classification error rate on held-out test cohort. |
| **Normal $\rightarrow$ Variation Errors** | 57 / 101 (56.4%) | 54 / 82 (65.9%) | **Dominant Failure Mode:** Healthy mucosa predicted as physiological variation. |
| **Normal $\rightarrow$ OPMD Errors** | 24 / 101 (23.8%) | 5 / 82 (6.1%) | False-positive precancerous alarms on normal mucosal reflections. |
| **Variation $\rightarrow$ OPMD Errors** | 9 / 101 (8.9%) | 0 / 82 (0.0%) | Benign variation misdiagnosed as precancerous leukoplakia. |
| **OPMD $\rightarrow$ Variation Errors** | 4 / 101 (4.0%) | 8 / 82 (9.8%) | Precancerous lesion under-triaged as benign variation. |
| **Normal $\leftrightarrow$ Variation Boundary Errors** | **71.4% of all errors** | **73.2% of all errors** | Primary cause of low macro precision across both architectures. |
