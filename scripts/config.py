"""
Shared configuration for the TB Mortality Prediction study.

IMPORTANT (macOS OpenMP): thread-limiting environment variables MUST be set
before numpy / scikit-learn / xgboost are imported anywhere. Every script in
this project imports `config` first, and `config` sets them at module load.
"""
import os
# --- macOS OpenMP / BLAS deadlock guard: set before any numeric import ---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
# scripts/ -> Mortality_Prediction/ -> project root
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent                 # Mortality_Prediction/
ROOT_DIR = PROJECT_DIR.parent                    # TB Study USM/
RAW_XLSX = ROOT_DIR / "Dataset" / "TB_data_2022.xlsx"

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"
for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PROCESSED_CSV = DATA_DIR / "mortality_cohort_processed.csv"

RANDOM_STATE = 42

# ----------------------------------------------------------------------------
# Target definition  (column: 'Hasil Rawatan', whitespace-stripped)
# ----------------------------------------------------------------------------
OUTCOME_COL = "Hasil Rawatan"
DIED_LABELS = {"Mati"}                                    # -> 1
SURVIVED_LABELS = {"Sembuh", "Sempurna Rawatan", "Terhenti Rawatan",
                   "Pindah Keluar & Hilang", "Gagal Rawatan"}   # -> 0
# Excluded entirely: "Masih Dalam Rawatan" (still on treatment),
#                    "Tukar Diagnosis" (diagnosis revised away from TB),
#                    missing.
TARGET = "mortality"

# ----------------------------------------------------------------------------
# Feature set (STRICT BASELINE-ONLY — known at diagnosis / treatment start).
# Maps original Bahasa Malaysia column -> English snake_case analysis name.
# ----------------------------------------------------------------------------
RENAME = {
    # numeric
    "Umur (Tahun)": "age",
    "Bilangan Tanggungan": "num_dependents",
    # categorical (demographic / socioeconomic)
    "Jantina": "sex",
    "Warganegara": "citizenship",
    "Status Pendidikan Pesakit": "education",
    "Status Pendapatan": "income_status",
    "Keturunan": "ethnicity",
    "Lokasi Tempat Tinggal Semasa": "residence_location",
    # categorical (clinical / comorbidity)
    "Diabetes Mellitus": "diabetes",
    "Merokok": "smoking",
    "HIV Pra-Diagnosa": "hiv_status",
    "Parut BCG": "bcg_scar",
    "Status X-ray Dada Semasa Diagnosa": "cxr_severity",
    "Lokasi Anatomi Tibi (Diagnosa)": "tb_site",
    "Kategori Kes Tibi": "tb_case_category",
    "Kahak Awal Rawatan": "baseline_sputum_smear",
    "Cara Pengesanan": "detection_method",
    "Petugas Perubatan/Kesihatan": "healthcare_worker",
    "TB Meningitis": "tb_meningitis",
    "TB Miliary": "tb_miliary",
}
# computed feature (not a direct rename)
DERIVED_NUMERIC = ["dx_to_treatment_days"]

NUMERIC_FEATURES = ["age", "num_dependents", "dx_to_treatment_days"]
CATEGORICAL_FEATURES = [
    "sex", "citizenship", "education", "income_status", "ethnicity",
    "residence_location", "diabetes", "smoking", "hiv_status", "bcg_scar",
    "cxr_severity", "tb_site", "tb_case_category", "baseline_sputum_smear",
    "detection_method", "healthcare_worker", "tb_meningitis", "tb_miliary",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Date columns used to derive the diagnosis->treatment interval
DATE_DIAG = "Tarikh Diagnosa"
DATE_TRT = "Tarikh Rawatan Dimulakan"

# Tier label for the data dictionary
TIER1 = {  # ~100% complete in cohort, zero/trivial imputation
    "age", "sex", "citizenship", "education", "income_status", "diabetes",
    "smoking", "hiv_status", "bcg_scar", "cxr_severity", "tb_site",
    "tb_case_category", "baseline_sputum_smear", "detection_method",
    "healthcare_worker",
}
TIER2 = {  # clinically important, documented imputation
    "num_dependents", "dx_to_treatment_days", "ethnicity",
    "residence_location", "tb_meningitis", "tb_miliary",
}

# ----------------------------------------------------------------------------
# English label maps for publication figures (populated after value inspection)
# ----------------------------------------------------------------------------
FEATURE_LABELS = {
    "age": "Age (years)",
    "num_dependents": "No. of dependents",
    "dx_to_treatment_days": "Diagnosis-to-treatment (days)",
    "sex": "Sex",
    "citizenship": "Citizenship",
    "education": "Education level",
    "income_status": "Income status",
    "ethnicity": "Ethnicity",
    "residence_location": "Residence location",
    "diabetes": "Diabetes mellitus",
    "smoking": "Smoking",
    "hiv_status": "HIV status (pre-diagnosis)",
    "bcg_scar": "BCG scar",
    "cxr_severity": "Chest X-ray severity",
    "tb_site": "TB anatomical site",
    "tb_case_category": "TB case category",
    "baseline_sputum_smear": "Baseline sputum smear",
    "detection_method": "Case-detection method",
    "healthcare_worker": "Healthcare worker",
    "tb_meningitis": "TB meningitis",
    "tb_miliary": "Miliary TB",
    "age_band": "Age band",
}

# Bahasa Malaysia category value -> English (filled after inspecting values)
VALUE_TRANSLATE = {
    "Ya": "Yes", "Tidak": "No",
    "LELAKI": "Male", "PEREMPUAN": "Female",
    "Negatif": "Negative", "Positif": "Positive", "Tidak Dibuat": "Not Done",
    "Ada": "Present", "Tiada": "Absent",
    "Pulmonari": "Pulmonary", "Ekstrapulmonari": "Extrapulmonary",
    "Pulmonari dan Ekstrapulmonari": "Pulmonary & Extrapulmonary",
    "Sederhana (Minimal)": "Minimal", "Teruk (Moderately Advanced)": "Mod. advanced",
    "Sangat Teruk (Far Advanced)": "Far advanced", "No Lesion": "No lesion",
    "Kes Baru": "New", "Kes Berulang": "Relapse",
    "Kes Setelah Terhenti Rawatan": "After interruption",
    "Kes Setelah Gagal Rawatan": "After failure",
    "Tidak Diketahui": "Unknown",
    # citizenship
    "Warganegara": "Citizen", "Bukan Warganegara": "Non-citizen",
    # residence
    "Bandar": "Urban", "Luar Bandar": "Rural",
    # detection method
    "Aktif": "Active", "Pasif": "Passive", "Saringan": "Screening",
    # ethnicity
    "MELAYU": "Malay", "CINA": "Chinese", "INDIA": "Indian", "LAIN-LAIN": "Others",
    "ORANG ASLI SEMENANJUNG": "Orang Asli (Pen.)", "PERIBUMI SABAH": "Sabah native",
    "PERIBUMI SARAWAK": "Sarawak native",
    # education
    "Tiada": "None", "Sekolah Rendah": "Primary", "Tingkatan 1/2/3": "Lower sec.",
    "Tingkatan 4/5": "Upper sec.", "Tingkatan 6/Diploma/Sijil": "Form 6/Dip.",
    "Ijazah": "Degree", "Lain-lain (nyatakan)": "Others",
}

PLOT_DPI = 300


def load_processed():
    """Load the processed cohort produced by 01_preprocessing.py."""
    import pandas as pd
    return pd.read_csv(PROCESSED_CSV)
