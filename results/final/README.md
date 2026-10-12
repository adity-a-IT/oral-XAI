# Final Research Package — Oral Lesion Deep Learning & XAI Framework

Welcome to the finalized, audited research package for:  
**"Explainable Deep Learning Framework for Oral Lesion Classification"**

This repository contains the complete experimental record, locked evaluation splits, trained model checkpoints, explainable AI localization logs, publication-quality figures, research tables, and deployment server.

---

## 1. Directory Structure

```text
results/final/
├── README.md                          # This reproduction guide
├── final_research_report.md           # Comprehensive consolidated research report
├── final_results.json                 # Machine-readable evaluation metrics & audit logs
├── reproducibility_checklist.md       # Quality gate and experimental integrity checklist
├── final_tables/                      # Research tables in Markdown (.md) and CSV (.csv)
│   ├── table1_class_distribution.*    # Dataset distribution (N=2,469)
│   ├── table2_split_distribution.*    # Patient-stratified split breakdown (1,726 / 372 / 371)
│   ├── table3_model_architectures.*   # Parameter counts and training configurations
│   ├── table4_classification_performance.* # Overall test metrics (Acc, Macro F1, Kappa)
│   ├── table5_per_class_performance.* # Per-class Precision, Recall, F1
│   ├── table6_calibration_error.*     # ECE, Brier score, and confusion matrix distribution
│   ├── table7_xai_localization.*      # Objective IoU, Dice, Pointing Game on N=49 cases
│   ├── table8_right_for_right_reasons.* # 2x2 prediction vs attribution contingency table
│   └── table9_phase8_deployment.*     # Production web deployment quality audit
└── final_figures/                     # Publication-ready figures (300 DPI PNG)
    ├── fig1_system_architecture.png   # End-to-end framework workflow
    ├── fig2_class_distribution.png    # Dataset imbalance & split distributions
    ├── fig3_confusion_matrices.png    # EfficientNetB0 vs MobileNetV2 confusion matrices
    ├── fig4_model_performance.png     # Overall & per-class F1 metric comparisons
    ├── fig5_xai_method_comparison.png # Multi-modal XAI on representative case
    ├── fig6_mask_vs_explanations.png  # Expert VIA polygon mask vs model attributions
    ├── fig7_representative_correct_cases.png # Correct diagnostic cases across all 4 classes
    ├── fig8_representative_failure_cases.png # Diagnostic failure modes & confounders
    └── fig9_right_for_right_reasons.png # 2x2 Right-for-the-Right-Reasons contingency chart
```

---

## 2. Reproduction Instructions

### A. Environment Setup
The codebase requires Python 3.11 with TensorFlow 2.16+, OpenCV, LIME, Matplotlib, and Flask:

```bash
# Verify Python version
py -3.11 --version

# Install dependencies if necessary
py -3.11 -m pip install -r requirements.txt
py -3.11 -m pip install flask lime
```

### B. Regenerate Figures & Tables
To reproduce all 9 publication figures and 9 tables strictly from existing locked artifacts:

```bash
py -3.11 src/generate_final_figures.py
```

### C. Run the Phase 8 Verification Test Suite
To verify model loading, 4-class probability sums, real Grad-CAM/Grad-CAM++/LIME generation, deterministic inference, and Phase 7 regression parity:

```bash
py -3.11 tests/test_phase8_integration.py
```

### D. Launch the Standalone Web Workstation
To interact with the live clinical diagnostic and explainability suite:

```bash
py -3.11 server.py
```
Then navigate to `http://localhost:5000` in any web browser.

---

## 3. Key Research Findings Summary

1. **Model Selection:** `EfficientNetB0` was selected as the primary architecture based strictly on **Validation Macro F1 (0.4709)** to avoid majority-class accuracy bias.
2. **Generalization Performance:** On the held-out test cohort ($N = 371$ images across 49 isolated patients), `EfficientNetB0` achieved **72.78% Accuracy**, **0.5346 Macro F1**, **0.7805 Weighted F1**, and **0.3324 Cohen's Kappa**.
3. **Primary Diagnostic Limitation:** **71.4% of all misclassifications** originate at the subtle boundary between `Normal` mucosa and `Variation from normal`.
4. **Minority Disease Sensitivity:**
   * **OPMD ($N=19$):** EfficientNetB0 achieved **78.95% recall** (15 of 19 detected).
   * **Oral Cancer ($N=3$):** EfficientNetB0 achieved **66.67% recall** (2 of 3 detected; explicitly flagged as a low statistical power estimate).
5. **Explainability Parity:** On the $N = 49$ annotation-eligible abnormal test cases with expert VIA polygon boundaries, Grad-CAM achieved a **66.7% Pointing Game Hit rate on Oral Cancer** (IoU: 0.2750, Dice: 0.4285) and **26.3% on OPMD** (IoU: 0.2104, Dice: 0.3183).
6. **Contingency Audit:** In the $2 \times 2$ prediction vs. attribution contingency analysis, **12.2% of cases are confirmed "right for the right reasons"** (Cell A), while 51.0% exhibit correct predictions relying on broader mucosal contexts (Cell B).
