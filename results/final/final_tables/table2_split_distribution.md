# Table 2: Patient-Level Partitioning & Stratification

| Partition | Total Images | Patient Count | Normal (0) | Variation (1) | OPMD (2) | Oral Cancer (3) | Patient Overlap (Leakage) |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| **Train** | 1,726 | 232 | 1,500 | 125 | 87 | 14 | 0.0% |
| **Validation** | 372 | 48 | 323 | 27 | 19 | 3 | 0.0% |
| **Test (Locked)** | 371 | 49 | 322 | 27 | 19 | 3 | 0.0% |
| **Total Cohort** | **2,469** | **329** | **2,145** | **179** | **125** | **20** | **0.0% (Zero Leakage)** |

*Verification:* $1,726 + 372 + 371 = 2,469$ images across $232 + 48 + 49 = 329$ distinct patients. Zero patient overlap confirmed.
