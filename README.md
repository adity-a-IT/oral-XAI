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
