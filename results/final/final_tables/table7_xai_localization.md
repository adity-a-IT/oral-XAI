# Table 7: Objective XAI Lesion Localization Performance on Expert-Annotated Subset ($N = 49$)

### EfficientNetB0 (Primary Model)
| Lesion Category | Support ($N$) | Grad-CAM IoU | Grad-CAM Dice | Grad-CAM Pointing Game | Grad-CAM++ IoU | Grad-CAM++ Dice | Grad-CAM++ Pointing Game | LIME IoU | LIME Dice | LIME Pointing Game |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Variation from normal** | 27 | 0.1293 | 0.2073 | 11.1% | 0.0835 | 0.1395 | 11.1% | 0.1730 | 0.2835 | 14.8% |
| **OPMD** | 19 | 0.2104 | 0.3183 | **26.3%** | 0.0861 | 0.1424 | **10.5%** | 0.1756 | 0.2823 | **15.8%** |
| **Oral Cancer (OSCC)\*** | 3 | 0.2750 | 0.4285 | **66.7%** | 0.1011 | 0.1682 | **33.3%** | 0.2087 | 0.3419 | **33.3%** |

### MobileNetV2 (Alternative Baseline)
| Lesion Category | Support ($N$) | Grad-CAM IoU | Grad-CAM Dice | Grad-CAM Pointing Game | Grad-CAM++ IoU | Grad-CAM++ Dice | Grad-CAM++ Pointing Game | LIME IoU | LIME Dice | LIME Pointing Game |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Variation from normal** | 27 | 0.1787 | 0.2874 | 29.6% | 0.1030 | 0.1732 | 22.2% | 0.1222 | 0.2089 | 18.5% |
| **OPMD** | 19 | 0.2412 | 0.3585 | **26.3%** | 0.1308 | 0.2002 | **31.6%** | 0.1855 | 0.2920 | **21.1%** |
| **Oral Cancer (OSCC)\*** | 3 | 0.2738 | 0.4240 | **66.7%** | 0.0507 | 0.0881 | **0.0%** | 0.1923 | 0.3182 | **0.0%** |

*Important Clarification:* Evaluated strictly on the $N = 49$ abnormal test cases with expert VIA polygon annotations. The remaining 322 normal test images have no pathological lesions and are excluded from localization scoring.  
*\*OSCC support is limited to N=3 test images; metrics are low-statistical-power estimates.*
