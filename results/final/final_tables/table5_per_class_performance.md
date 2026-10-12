# Table 5: Per-Class Diagnostic Performance on Test Set ($N = 371$)

### EfficientNetB0 (Selected Validation Checkpoint)
| Diagnostic Category | Test Support ($N$) | Precision | Recall (Sensitivity) | F1-Score | Statistical Reliability Category |
|:---|:---:|---:|---:|---:|:---|
| **Normal** | 322 | 97.95% | 74.22% | 0.8445 | Robust Statistical Power ($N = 322$) |
| **Variation from normal** | 27 | 18.67% | 51.85% | 0.2745 | Moderate Statistical Power ($N = 27$) |
| **OPMD** | 19 | 31.25% | 78.95% | 0.4478 | Moderate Statistical Power ($N = 19$) |
| **Oral Cancer (OSCC)\*** | 3 | 50.00% | 66.67% | 0.5714 | **LOW STATISTICAL POWER ($N = 3$)** |

### MobileNetV2 (Alternative Baseline)
| Diagnostic Category | Test Support ($N$) | Precision | Recall (Sensitivity) | F1-Score | Statistical Reliability Category |
|:---|:---:|---:|---:|---:|:---|
| **Normal** | 322 | 97.38% | 80.75% | 0.8829 | Robust Statistical Power ($N = 322$) |
| **Variation from normal** | 27 | 21.52% | 62.96% | 0.3208 | Moderate Statistical Power ($N = 27$) |
| **OPMD** | 19 | 50.00% | 52.63% | 0.5128 | Moderate Statistical Power ($N = 19$) |
| **Oral Cancer (OSCC)\*** | 3 | 40.00% | 66.67% | 0.5000 | **LOW STATISTICAL POWER ($N = 3$)** |

*\*OSCC support is limited to N=3 test images; each individual sample represents a discrete 33.3% sensitivity jump.*
