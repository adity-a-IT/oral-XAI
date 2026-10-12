# Explainable Deep Learning Framework for Oral Lesion Classification Using LIME and Grad-CAM

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-00f3ff?style=for-the-badge&logo=github)](https://adity-a-it.github.io/oral-XAI/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-oral--XAI-2ea44f?style=for-the-badge&logo=github)](https://github.com/adity-a-IT/oral-XAI)

🌐 **Live Web Application:** [https://adity-a-it.github.io/oral-XAI/](https://adity-a-it.github.io/oral-XAI/)

This project is a research-oriented, reproducible prototype for classifying oral lesion images into Healthy, OPMD, and OSCC, with explainability via Grad-CAM and LIME.

The project follows a staged development workflow and begins with dataset inspection and validation before any training pipeline is built.

## Scope

- Dataset inspection and audit
- Patient-level split prevention of leakage
- Preprocessing and augmentation pipeline
- Baseline transfer learning models
- Independent CNN architecture inspired by the literature
- Training, evaluation, explainability, and error analysis
- Standalone Cyber-Medical Web Diagnostic & Explainability Suite (Flask + Neural XAI)

## Multiclass ROC Curves & Clinical AUC Benchmarks

Evaluated on the patient-stratified held-out test split ($N = 371$ images across 49 isolated patients, zero train/validation data leakage).

| Model | Normal ($N=322$) | Variation ($N=27$) | OPMD ($N=19$) | OSCC ($N=3$) * | Macro AUC | Weighted AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNetB0** (Best Val) | **0.9152** | 0.7610 | **0.9486** | 0.9330 * | 0.8894 | **0.9058** |
| **MobileNetV2** | 0.8976 | **0.7884** | 0.9266 | **0.9674** * | **0.8950** | 0.8917 |
| **Custom CNN** | 0.8283 | 0.7852 | 0.8201 | 0.9366 * | 0.8426 | 0.8256 |

*\* **Statistical Power Note on OSCC ($N=3$)** : Under patient-isolated splitting, the test cohort holds only 3 carcinoma cases. True positive rate advances in discrete 33.3% increments, with 95% confidence intervals spanning [0.736 - 1.000]. Nominal high AUCs must be treated with appropriate clinical caution until validated against large multicenter cohorts.*

### Reproducing ROC & AUC Evaluation
```powershell
# Run standalone ROC pipeline (generates 300 DPI plots and CSV)
.\.venv\Scripts\python src/evaluate_roc_auc.py
```
Output artifacts are saved in `results/roc_analysis/` and `assets/roc_analysis/`.

## Running the Project

The application runs independently inside its own Python virtual environment (`.venv`) without any Streamlit dependency.

### Quick Start (Windows)
Double-click `run.bat` or run:
```powershell
.\run.bat
```
Or directly using PowerShell:
```powershell
.\run.ps1
```

### Manual Execution in Virtual Environment
```powershell
# 1. Activate dedicated environment
.\.venv\Scripts\activate

# 2. Run the standalone server
python server.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser.

## Important note

This project is a research prototype, not a medical diagnostic device.

