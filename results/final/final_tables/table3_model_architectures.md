# Table 3: Model Architecture & Parameter Comparison

| Model | Backbone Pretraining | Input Resolution | Total Parameters | Trainable Parameters | Preprocessing Pipeline | Optimization Strategy |
|:---|:---|:---:|---:|---:|:---|:---|
| **Custom CNN** | Trained from scratch | $224 \times 224 \times 3$ | 304,932 | 304,484 | Direct Rescaling to $[0.0, 1.0]$ | Adam (lr=1e-3), Balanced Weights |
| **MobileNetV2** | ImageNet Transfer Learning | $224 \times 224 \times 3$ | 2,263,108 | 1,057,604 | Standard Scaling to $[-1.0, 1.0]$ | Head (1e-3) + Fine-Tuning (1e-5) |
| **EfficientNetB0\*** | ImageNet Transfer Learning | $224 \times 224 \times 3$ | 4,054,695 | 1,134,900 | Raw $[0.0, 255.0]$ Float32 (Internal Rescale) | Head (1e-3) + Fine-Tuning (1e-5) |

*\*Selected model for downstream explainability and deployment based strictly on Validation Macro F1.*
