# Phase 5: Baseline Model Comparison Table

| Model | Total Params | Trainable Params | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Test Weighted F1 | Test Cohen's Kappa |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Custom CNN** | 304,932 | 304,484 | 86.83% | **0.2324** | 86.79% | **0.2323** | 0.8066 | 0.0000 |
| **MobileNetV2** | 2,263,108 | 1,057,604 | 70.97% | **0.4624** | 77.90% | **0.5541** | 0.8199 | 0.3809 |
| **EfficientNetB0** | 4,054,695 | 1,134,900 | 71.24% | **0.4709** | 72.78% | **0.5346** | 0.7805 | 0.3324 |

**Selected Best Validation Model:** `EfficientNetB0` (Selected strictly by Validation Macro F1 to counteract class imbalance).
