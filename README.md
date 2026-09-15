# TB Mortality Prediction Study

Machine-learning prediction of **death during tuberculosis (TB) treatment** using
the TB Information System (TBIS) registry, Wilayah Persekutuan, Malaysia, 2022
(N = 3,743 registered patients). One of several studies derived from the shared
TB dataset; this folder is self-contained.

This repository contains the experiment code, aggregate outputs, and figures only.
Patient-level datasets, trained model binaries, and document-generation files are
not included.

## Ethics

The Malaysian National Medical Research Registry has registered this study, which
has received ethical approval from two human ethics committees: the Medical
Research and Ethics Committee, Ministry of Health Malaysia (NMRR-19-2818-50518-IIR),
and the Human Research Ethics Committee of Universiti Sains Malaysia
(USM/JEPeM/19090558).

> Sensitive de-identified patient data. All processing is local. The raw dataset is
> **not** copied into this folder; scripts read it from `../Dataset/TB_data_2022.xlsx`.

## Objective

Binary classification: predict **mortality (died = 1)** vs **survived / other
concluded outcome (= 0)** during TB treatment, and identify the clinical and
socioeconomic predictors of death.

## Cohort definition (target = `Hasil Rawatan`, whitespace-normalised)

| | Outcome | n | Label |
|---|---|---|---|
| Event | Mati (died) | 405 | **1** |
| Non-event | Sembuh, Sempurna Rawatan, Terhenti Rawatan, Pindah Keluar & Hilang, Gagal Rawatan | 1,073 | **0** |
| Excluded | Masih Dalam Rawatan (still on treatment) | 1,799 | — |
| Excluded | Tukar Diagnosis (diagnosis revised away from TB) | 37 | — |
| Excluded | missing outcome | 429 | — |

**Analytic cohort = 1,478; in-hospital/treatment death rate = 27.4%.**

### Confirmed design decisions
1. **Tukar Diagnosis excluded** — these patients are no longer TB cases (diagnosis
   revised) and contribute 0 deaths; standard WHO/national TB cohort practice.
2. **HIV = pre-diagnosis status only** (`HIV Pra-Diagnosa`; 100% complete in-cohort).
3. **Strict baseline-only features** — only variables knowable at diagnosis /
   treatment start. All during/post-treatment fields (treatment duration, monthly
   sputum, HAART/CPT, special events, **and all cause-of-death fields**) are
   excluded from the model to prevent target leakage.

## Feature set (21 predictors)

**Tier 1 (≈100% complete in cohort, no/trivial imputation):** age, sex, citizenship,
education, income status, diabetes, smoking, HIV (pre-Dx), BCG scar, chest X-ray
severity, TB anatomical site, TB case category, baseline sputum smear, case-detection
method, healthcare-worker status.

**Tier 2 (clinically important, documented imputation):** number of dependents
(median), diagnosis-to-treatment interval (implausible >365 d / <0 d → median),
ethnicity (explicit *Unknown* category), residence location (mode), **TB meningitis**
and **miliary TB** (NaN → *No*: the organ-specific item is only raised for
extrapulmonary work-ups, so a pulmonary case definitionally lacks these).

Full per-feature imputation log: [`results/data_dictionary.csv`](results/data_dictionary.csv).

## Pipeline

| Step | Script | Key outputs |
|---|---|---|
| 0 | `config.py` | shared paths, feature maps, thread guards |
| 1 | `01_preprocessing.py` | `data/mortality_cohort_processed.csv`, `cohort_construction.csv`, `data_dictionary.csv` |
| 2 | `02_eda.py` | figs 01–05, `table1_baseline_characteristics.csv`, `eda_association_ranking.csv` |
| 3 | `03_modelling.py` | figs 06–08, `model_comparison_{cv,test}.csv`, confusion matrix, classification report, model `.pkl` |
| 4 | `04_interpretability.py` | figs 09–11, `logistic_regression_odds_ratios.csv`, `shap_feature_importance.csv` |
| 5 | `05_cause_of_death.py` | fig 12, `cause_of_death_summary.csv` |

### Modelling specifics
- 6 algorithms: Logistic Regression, Random Forest, XGBoost, Gradient Boosting, SVM, KNN.
- imbalanced-learn `Pipeline`: `ColumnTransformer` (standardise numeric + one-hot
  categorical) → **SMOTE** → classifier. SMOTE is fit on **training folds only**
  (never the validation fold or the test set), so there is no resampling leakage.
- **80/20 stratified hold-out** + **5-fold stratified CV** on the training set
  (`random_state=42`).
- Metrics: accuracy, weighted F1, ROC-AUC, **sensitivity**, **specificity**, plus
  confusion matrix and per-class report (sensitivity prioritised for mortality).
- macOS OpenMP safety: thread env vars set to 1 in `config.py` before any numeric
  import; every estimator uses `n_jobs=1`.

## Headline results

**Best model: Logistic Regression** (selected by mean CV ROC-AUC).

| Model | CV ROC-AUC | Test ROC-AUC | Test Sens. | Test Spec. |
|---|---|---|---|---|
| **Logistic Regression** | **0.760** | **0.736** | **0.741** | 0.670 |
| Gradient Boosting | 0.752 | 0.675 | 0.358 | 0.758 |
| SVM | 0.750 | 0.674 | 0.617 | 0.647 |
| XGBoost | 0.749 | 0.673 | 0.370 | 0.795 |
| Random Forest | 0.731 | 0.668 | 0.284 | 0.837 |
| KNN | 0.695 | 0.674 | 0.630 | 0.619 |

Logistic Regression gives the best discrimination **and** the most balanced
sensitivity/specificity; the tree ensembles are conservative (high specificity, low
sensitivity) at the default 0.5 threshold.

**Adjusted odds ratios (best model)** — strongest risk factors: TB meningitis
OR 12.3 (95% CI 3.7–40.9), miliary TB 6.8 (2.2–21.7), chest X-ray "not done" 5.3
(1.9–14.8), HIV-positive 3.5 (2.1–5.7), age 1.58 per decade (1.44–1.73). Protective:
female sex 0.43, having an income 0.53, extrapulmonary site 0.57. SHAP (TreeExplainer
on Gradient Boosting, the best tree model) independently ranks age → sex → income →
chest X-ray → HIV as the top contributors.

**Secondary (cause of death):** of 405 deaths, cause was recorded for 152 (37.5%);
of those, 62 (40.8%) were TB-related and 90 (59.2%) non-TB-related. Leading
contributors: late diagnosis (13.8%), secondary infection (13.2%).

## Limitation to report

At data extraction, **48% of registered patients were still on treatment** and were
excluded (no definitive endpoint). Because deaths conclude early while cures/completions
accrue later, the concluded-outcome cohort over-represents deaths, so the observed
**27.4% death rate is higher than the true cohort case-fatality ratio**. The model
therefore predicts mortality *conditional on having a concluded outcome at extraction*;
this incomplete-follow-up / immortal-time consideration should be stated explicitly.
A smoking OR < 1 (apparent protective effect) is most plausibly confounding (smokers
are younger) rather than causal and should be interpreted cautiously.

## Reproduce

```bash
PY=~/opt/anaconda3/bin/python3   # pandas 1.4.2, scikit-learn 1.6.1, xgboost 1.5.0,
                                 # imbalanced-learn 0.12.4, shap 0.49.1, statsmodels 0.13.2
cd Mortality_Prediction/scripts
$PY 01_preprocessing.py && $PY 02_eda.py && $PY 03_modelling.py && \
$PY 04_interpretability.py && $PY 05_cause_of_death.py
```

All randomness is seeded (`random_state=42`); reruns are deterministic.
