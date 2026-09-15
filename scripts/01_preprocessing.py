"""
01 - Preprocessing & cohort construction for TB mortality prediction.

Outputs:
  data/mortality_cohort_processed.csv  - analysis-ready cohort (English columns)
  results/cohort_construction.csv      - STROBE-style patient-flow counts
  results/data_dictionary.csv          - feature roles, tiers, imputation log
"""
import config  # sets thread env vars first
import pandas as pd
import numpy as np

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

print("Loading:", config.RAW_XLSX)
df = pd.read_excel(config.RAW_XLSX)
N0 = len(df)
print(f"Raw records: {N0}, columns: {df.shape[1]}")

# ----------------------------------------------------------------------------
# 1. Target: derive mortality from 'Hasil Rawatan' (strip whitespace first)
# ----------------------------------------------------------------------------
outcome = df[config.OUTCOME_COL].astype("string").str.strip()

flow = []
flow.append(("Registered TB records (2022, Wilayah Persekutuan)", N0))
n_missing = outcome.isna().sum()
n_intrt = (outcome == "Masih Dalam Rawatan").sum()
n_tukar = (outcome == "Tukar Diagnosis").sum()
flow.append(("Excluded: outcome missing", int(n_missing)))
flow.append(("Excluded: 'Masih Dalam Rawatan' (still on treatment)", int(n_intrt)))
flow.append(("Excluded: 'Tukar Diagnosis' (diagnosis revised away from TB)", int(n_tukar)))

is_died = outcome.isin(config.DIED_LABELS)
is_surv = outcome.isin(config.SURVIVED_LABELS)
in_cohort = (is_died | is_surv).fillna(False)

flow.append(("ANALYTIC COHORT (concluded TB outcomes)", int(in_cohort.sum())))
flow.append(("  Died (mortality = 1)", int(is_died.sum())))
flow.append(("  Survived/other concluded (mortality = 0)", int(is_surv.sum())))

cohort = df[in_cohort].copy()
cohort[config.TARGET] = is_died[in_cohort].astype(int).values
cohort["outcome_raw"] = outcome[in_cohort].values

print("\n--- Cohort construction ---")
for label, n in flow:
    print(f"  {label:<60} {n:>6}")
death_rate = 100 * cohort[config.TARGET].mean()
print(f"  Death rate: {death_rate:.1f}%")

pd.DataFrame(flow, columns=["step", "n"]).to_csv(
    config.RESULTS_DIR / "cohort_construction.csv", index=False)

# ----------------------------------------------------------------------------
# 2. Derived feature: diagnosis -> treatment-start interval (days)
#    >365 or <0 treated as data-entry error -> NaN (then median imputed).
# ----------------------------------------------------------------------------
diag = pd.to_datetime(cohort[config.DATE_DIAG], errors="coerce")
trt = pd.to_datetime(cohort[config.DATE_TRT], errors="coerce")
interval = (trt - diag).dt.days
n_bad = int(((interval < 0) | (interval > 365)).sum())
interval = interval.where((interval >= 0) & (interval <= 365), np.nan)
cohort["dx_to_treatment_days"] = interval.values
print(f"\nDiagnosis-to-treatment interval: {n_bad} implausible values (<0 or >365 days) set to NaN")

# ----------------------------------------------------------------------------
# 3. Select & rename features
# ----------------------------------------------------------------------------
work = cohort.rename(columns=config.RENAME)
keep = config.FEATURES + [config.TARGET, "outcome_raw"]
# age band for EDA only
work["age_band"] = cohort["(Recode) Umur (Tahun)"].values
keep.append("age_band")
work = work[keep].copy()

# ----------------------------------------------------------------------------
# 4. Imputation (documented). Tier-1 categoricals are ~100% complete in cohort.
# ----------------------------------------------------------------------------
imp_log = []  # feature, n_missing_before, strategy, fill_value

def log_impute(col, before, strategy, fill):
    imp_log.append({"feature": col, "n_missing_before": int(before),
                    "strategy": strategy, "fill_value": str(fill)})

# 4a. TB meningitis / miliary: NaN means the organ-specific item was not raised
#     (pulmonary-only cases) -> clinically 'Tidak' (No).
for col in ["tb_meningitis", "tb_miliary"]:
    before = work[col].isna().sum()
    work[col] = work[col].fillna("Tidak")
    log_impute(col, before, "clinical: NaN->'Tidak' (pulmonary case lacks this site)", "Tidak")

# 4b. Ethnicity: explicit Unknown category (avoids distorting distribution).
before = work["ethnicity"].isna().sum()
work["ethnicity"] = work["ethnicity"].fillna("Tidak Diketahui")
log_impute("ethnicity", before, "explicit category 'Tidak Diketahui' (Unknown)", "Tidak Diketahui")

# 4c. Numeric: median imputation.
for col in ["age", "num_dependents", "dx_to_treatment_days"]:
    before = work[col].isna().sum()
    med = work[col].median()
    work[col] = work[col].fillna(med)
    log_impute(col, before, "median", med)

# 4d. Remaining categoricals: mode imputation (only residence has a few missing).
for col in config.CATEGORICAL_FEATURES:
    if col in ("tb_meningitis", "tb_miliary", "ethnicity"):
        continue
    before = work[col].isna().sum()
    if before > 0:
        mode = work[col].mode(dropna=True).iloc[0]
        work[col] = work[col].fillna(mode)
        log_impute(col, before, "mode", mode)
    else:
        log_impute(col, 0, "none (complete in cohort)", "-")

assert work[config.FEATURES].isna().sum().sum() == 0, "Residual missing after imputation!"
print("\nAll features fully populated after imputation.")

# ----------------------------------------------------------------------------
# 5. Inspect categorical values (to validate English figure translations)
# ----------------------------------------------------------------------------
print("\n--- Categorical feature values (for label translation) ---")
for col in config.CATEGORICAL_FEATURES:
    vals = sorted(work[col].dropna().unique().tolist(), key=str)
    print(f"  {col:<22}: {vals}")

# ----------------------------------------------------------------------------
# 6. Data dictionary
# ----------------------------------------------------------------------------
inv_rename = {v: k for k, v in config.RENAME.items()}
dd = []
for col in config.FEATURES:
    role = "numeric" if col in config.NUMERIC_FEATURES else "categorical"
    tier = "Tier-1" if col in config.TIER1 else "Tier-2"
    orig = inv_rename.get(col, "derived: Tarikh Rawatan Dimulakan - Tarikh Diagnosa")
    li = next((x for x in imp_log if x["feature"] == col), {})
    dd.append({"feature": col, "original_column": orig, "role": role, "tier": tier,
               "n_missing_in_cohort": li.get("n_missing_before", 0),
               "imputation": li.get("strategy", "-"),
               "fill_value": li.get("fill_value", "-"),
               "label": config.FEATURE_LABELS.get(col, col)})
dd_df = pd.DataFrame(dd)
dd_df.to_csv(config.RESULTS_DIR / "data_dictionary.csv", index=False)
print("\n--- Data dictionary (imputation summary) ---")
print(dd_df[["feature", "tier", "n_missing_in_cohort", "imputation", "fill_value"]].to_string(index=False))

# ----------------------------------------------------------------------------
# 7. Save processed cohort
# ----------------------------------------------------------------------------
work.to_csv(config.PROCESSED_CSV, index=False)
print(f"\nSaved processed cohort: {config.PROCESSED_CSV}  shape={work.shape}")
print(f"Target balance: {work[config.TARGET].value_counts().to_dict()}  "
      f"(death rate {death_rate:.1f}%)")
