# COMPREHENSIVE ERROR ANALYSIS REPORT

**Evaluated Model:** `efficientnet_b0`  
**Test Set Samples:** 1502  
**Total Misclassifications:** 486 (32.36% error rate)  

--- 

## 1. Top Confused Class Pairs

| True Class | Predicted Class | Count | Description |
|---|---|---|---|
| `nv` | `mel` | **140** | NV misclassified as MEL |
| `nv` | `bkl` | **94** | NV misclassified as BKL |
| `nv` | `akiec` | **32** | NV misclassified as AKIEC |
| `bkl` | `mel` | **31** | BKL misclassified as MEL |
| `mel` | `nv` | **28** | MEL misclassified as NV |
| `nv` | `bcc` | **23** | NV misclassified as BCC |
| `mel` | `bkl` | **21** | MEL misclassified as BKL |
| `mel` | `akiec` | **20** | MEL misclassified as AKIEC |
| `bkl` | `akiec` | **12** | BKL misclassified as AKIEC |
| `bkl` | `nv` | **12** | BKL misclassified as NV |

---

## 2. High-Confidence Misclassifications (>70% Confidence)

Found **31** cases where the model was highly confident but incorrect.

| Image ID | True Class | Predicted Class | Confidence |
|---|---|---|---|
| `ISIC_0031404` | `nv` | `bcc` | **96.46%** |
| `ISIC_0028098` | `nv` | `bkl` | **94.12%** |
| `ISIC_0031158` | `nv` | `df` | **91.85%** |
| `ISIC_0027142` | `bkl` | `nv` | **90.03%** |
| `ISIC_0033659` | `bkl` | `akiec` | **89.51%** |
| `ISIC_0025804` | `bkl` | `nv` | **89.27%** |
| `ISIC_0028760` | `mel` | `nv` | **84.44%** |
| `ISIC_0033992` | `nv` | `bkl` | **82.05%** |
| `ISIC_0031186` | `mel` | `akiec` | **81.89%** |
| `ISIC_0031339` | `mel` | `akiec` | **80.79%** |

---

## 3. Clinical Failure Diagnostics & Insights

- **Melanoma (`mel`) vs Benign Keratosis (`bkl`):** Visual overlap in pigment networks and irregular borders often causes misclassifications between Melanoma and solar lentigines.
- **Minority Classes (`df`, `vasc`):** Rare lesion types require targeted Focal Loss scaling to ensure high recall despite low sample representations in HAM10000.
