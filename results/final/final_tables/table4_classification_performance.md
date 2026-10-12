# Table 4: Overall Generalization Performance on Locked Test Cohort ($N = 371$)

| Model | Selection Metric (Val Macro F1) | Test Accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 | Test Weighted F1 | Test Cohen's Kappa |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Custom CNN** | 0.2324 | 86.79% | 21.70% | 25.00% | 0.2323 | 0.8066 | 0.0000 |
| **MobileNetV2** | 0.4624 | 77.90% | 52.22% | 65.75% | 0.5541 | 0.8199 | 0.3809 |
| **EfficientNetB0\*** | **0.4709** | 72.78% | 49.47% | 67.92% | 0.5346 | 0.7805 | 0.3324 |

*\*EfficientNetB0 was designated as the primary model solely based on Validation Macro F1 to eliminate test-set snooping bias.*
