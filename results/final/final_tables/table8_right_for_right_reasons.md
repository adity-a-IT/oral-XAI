# Table 8: 2×2 Right-for-the-Right-Reasons Attribution Contingency Table ($N = 49$)

| Cell Identification | Clinical & Algorithmic Meaning | EfficientNetB0 Count | EfficientNetB0 % | MobileNetV2 Count | MobileNetV2 % |
|:---|:---|:---:|:---:|:---:|:---:|
| **Cell A** | **Correct Prediction & Lesion-Focused Attribution** (Pointing Game Hit) | 6 / 49 | 12.2% | 15 / 49 | 30.6% |
| **Cell B** | **Correct Prediction & Non-Lesion Attribution** (Pointing Game Miss) | 25 / 49 | 51.0% | 14 / 49 | 28.6% |
| **Cell C** | **Incorrect Prediction & Lesion-Focused Attribution** (Pointing Game Hit) | 7 / 49 | 14.3% | 5 / 49 | 10.2% |
| **Cell D** | **Incorrect Prediction & Non-Lesion Attribution** (Pointing Game Miss) | 11 / 49 | 22.4% | 15 / 49 | 30.6% |
| **Total Cohort** | **Annotated Abnormal Test Images** | **49 / 49** | **100.0%** | **49 / 49** | **100.0%** |

*Methodological Insight:* Cell B cases frequently occur in diffuse physiological variations where model activations span broad mucosal fields outside tight central polygon coordinates. Cell C demonstrates cases where feature extraction successfully isolated the lesion bed, but class-boundary weighting favored adjacent categories.
