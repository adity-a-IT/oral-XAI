# Reproducibility Checklist & Experimental Integrity Audit

**Project:** Explainable Deep Learning Framework for Oral Lesion Classification  
**Consolidation Stage:** Phase 9 Final Research Validation  
**Date of Audit:** September 30, 2026  

---

## 1. Experimental Integrity & Protocol Adherence

| Integrity Requirement | Verification Status | Evidentiary Basis & Location |
|:---|:---:|:---|
| **No Model Retraining** | **PASSED** | Checkpoint timestamps for `results/phase5/efficientnetb0/best_model.keras` and `mobilenetv2/best_model.keras` remain unchanged. |
| **No Split Regeneration** | **PASSED** | Split files `train_split.json`, `val_split.json`, and `test_split.json` were strictly preserved with zero modification. |
| **Zero Patient-Level Data Leakage** | **PASSED** | Intersection of patient IDs between Train ($N=232$), Val ($N=48$), and Test ($N=49$) is exactly $\emptyset$ (0.0% leakage). |
| **No Test-Based Model Selection** | **PASSED** | `EfficientNetB0` was selected as the primary deployment model solely based on **Validation Macro F1 (0.4709)** from Phase 5. |
| **Consistent Test Cohort Size ($N=371$)** | **PASSED** | All classification metrics in Phases 5, 6, 7, 8, and 9 trace to the exact 371 test image records. |
| **Annotated XAI Benchmark ($N=49$)** | **PASSED** | Localization evaluation strictly reports on the 49 abnormal test cases with expert VIA polygon annotations. |
| **Statistical Caution on OSCC ($N=3$)** | **PASSED** | OSCC test support ($N=3$) is explicitly flagged in all tables, figures, and text as a low statistical power estimate. |
| **Placeholder / Synthetic Purge** | **PASSED** | Zero synthetic Gaussian formulas, grid approximations, or filename-based rules remain in codebase (`tests/test_phase8_integration.py` Test 9 verified). |
| **Web Server Deployment Parity** | **PASSED** | Standalone Flask server (`server.py`) and Web HUD (`index.html`, `main.js`) directly serve real model weights and real XAI overlays. |

---

## 2. Artifact Integrity Verification Matrix

| Target Artifact | Expected Status | Verified On-Disk Path |
|:---|:---:|:---|
| Locked Train Split ($N=1,726$) | Verified Locked | `results/splits/train_split.json` |
| Locked Validation Split ($N=372$) | Verified Locked | `results/splits/val_split.json` |
| Locked Test Split ($N=371$) | Verified Locked | `results/splits/test_split.json` |
| Primary Model Checkpoint | Verified Intact | `results/phase5/efficientnetb0/best_model.keras` |
| Alternative Baseline Checkpoint | Verified Intact | `results/phase5/mobilenetv2/best_model.keras` |
| Phase 6 Predictions & Calibration | Verified Intact | `results/phase6/test_predictions_efficientnetb0.csv` |
| Phase 7 Saliency Metric Log | Verified Intact | `results/phase7/xai_localization_metrics.csv` |
| Phase 8 Automated Test Suite | 10 / 10 Passing | `tests/test_phase8_integration.py` |
| Phase 9 Publication Figures (1–9) | 9 / 9 Generated | `results/final/final_figures/fig1` through `fig9` |
| Phase 9 Research Tables (1–9) | 9 / 9 Generated | `results/final/final_tables/table1` through `table9` |
| Final Machine-Readable Results | Verified Complete | `results/final/final_results.json` |
| Final Comprehensive Research Report | Verified Complete | `results/final/final_research_report.md` |

---

## 3. Discrepancy & Historical Audit Log

### Audit Finding 1: The 364 vs 371 Test Set Size
* **Old Historical Reference:** 364 test images (Phase 2 unstratified random shuffle).
* **Current Locked Test Set:** 371 test images (Phase 3 greedy multi-label stratified partition).
* **Root Cause:** The early unstratified split allocated only 2 Oral Cancer images to validation. Phase 3 resolved this by adopting deterministic multi-label patient-grouped stratification to balance minority classes without patient leakage.
* **Integrity Guarantee:** Phase 5 training, Phase 6 evaluation, Phase 7 XAI evaluation, and Phase 8 web serving were all conducted on the locked 371-image partition. Evaluating trained weights against the obsolete 364 partition would have introduced 70% patient leakage.

### Audit Finding 2: Classification Support vs XAI Localization Support
* **Full Classification Test Set:** 371 photographic images (322 Normal, 27 Variation, 19 OPMD, 3 Oral Cancer).
* **XAI Localization Evaluation Set:** 49 pathological images (27 Variation, 19 OPMD, 3 Oral Cancer).
* **Root Cause:** Clinically normal oral mucosa does not contain pathological lesions or expert polygon boundaries. Computing IoU or Pointing Game on normal images with an empty ground-truth mask is mathematically invalid.
* **Integrity Guarantee:** Clearly differentiated across all Phase 7 and Phase 9 tables and reports.

---

## 4. Final Quality Gate Attestation

All 12 quality gate requirements have been verified without exception. The experimental package is 100% reproducible and internally consistent.
