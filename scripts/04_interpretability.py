"""
04 - Interpretability.

(a) Best model is Logistic Regression -> adjusted odds ratios (multivariable,
    statsmodels MLE, reference = first category) + forest plot. This is the
    clinically standard interpretation of the best model.
(b) SHAP (TreeExplainer) on the best tree-based model (gradient-boosted),
    as specified in the protocol, for complementary ML feature attribution.

Outputs:
  results/logistic_regression_odds_ratios.csv
  results/shap_feature_importance.csv
  figures/fig_09_lr_odds_ratios.png
  figures/fig_10_shap_beeswarm.png
  figures/fig_11_shap_bar.png
"""
import config  # sets thread env vars first
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import shap
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid", context="paper")
import plotstyle
plotstyle.use_style()
plt.rcParams.update({"savefig.dpi": config.PLOT_DPI, "savefig.bbox": "tight",
                     "axes.titleweight": "bold", "font.size": 10})
RS = config.RANDOM_STATE
RISK_C, PROT_C = "#C0392B", "#2E86C1"

df = config.load_processed()
X = df[config.FEATURES].copy()
y = df[config.TARGET].astype(int).values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RS)


def tr(x):
    return config.VALUE_TRANSLATE.get(str(x), str(x))


# ============================================================================
# (a) Multivariable logistic regression -> adjusted odds ratios
#     Sparse categories are collapsed and `healthcare_worker` is dropped (only
#     1 death among HCWs -> quasi-complete separation) so the MLE converges.
#     These collapses apply ONLY to the inferential OR model; the ML models in
#     03_modelling.py use the full feature encoding.  Age per 10-year increase.
# ============================================================================
ETH = {"MELAYU": "Malay", "CINA": "Chinese", "INDIA": "Indian",
       "ORANG ASLI SEMENANJUNG": "Bumiputera/Other", "PERIBUMI SABAH": "Bumiputera/Other",
       "PERIBUMI SARAWAK": "Bumiputera/Other", "LAIN-LAIN": "Bumiputera/Other",
       "Tidak Diketahui": "Unknown"}
EDU = {"Tiada": "None", "Sekolah Rendah": "Primary", "Tingkatan 1/2/3": "Secondary",
       "Tingkatan 4/5": "Secondary", "Tingkatan 6/Diploma/Sijil": "Tertiary",
       "Ijazah": "Tertiary", "Lain-lain (nyatakan)": "Other"}
DET = {"Pasif": "Passive", "Aktif": "Active/Screening", "Saringan": "Active/Screening"}
YN = {"Ya": "Yes", "Tidak": "No"}

# NOTE: `citizenship` is omitted from the inferential OR model: 100% of
# non-citizens have 'Unknown' ethnicity (and vice-versa 99%), so the two are
# near-perfectly collinear and their coefficients diverge. The 'Ethnicity:
# Unknown' level therefore stands in for the (largely foreign-born) group.
cat = pd.DataFrame(index=X.index)
cat["Sex"] = X["sex"].map({"LELAKI": "Male", "PEREMPUAN": "Female"})
cat["Education level"] = X["education"].map(EDU)
cat["Income status"] = X["income_status"].map(YN)
cat["Ethnicity"] = X["ethnicity"].map(ETH)
cat["Residence location"] = X["residence_location"].map(tr)
cat["Diabetes mellitus"] = X["diabetes"].map(YN)
cat["Smoking"] = X["smoking"].map(YN)
cat["HIV (pre-Dx)"] = X["hiv_status"].map(tr)
cat["BCG scar"] = X["bcg_scar"].map(tr)
cat["Chest X-ray severity"] = X["cxr_severity"].map(tr)
cat["TB anatomical site"] = X["tb_site"].map(tr)
cat["TB case category"] = X["tb_case_category"].map(tr)
cat["Baseline sputum smear"] = X["baseline_sputum_smear"].map(tr)
cat["Case-detection method"] = X["detection_method"].map(DET)
cat["TB meningitis"] = X["tb_meningitis"].map(YN)
cat["Miliary TB"] = X["tb_miliary"].map(YN)

REF = {"Sex": "Male", "Education level": "Tertiary",
       "Income status": "No", "Ethnicity": "Malay", "Residence location": "Urban",
       "Diabetes mellitus": "No", "Smoking": "No", "HIV (pre-Dx)": "Negative",
       "BCG scar": "Present", "Chest X-ray severity": "No lesion",
       "TB anatomical site": "Pulmonary", "TB case category": "New",
       "Baseline sputum smear": "Negative", "Case-detection method": "Passive",
       "TB meningitis": "No", "Miliary TB": "No"}

design = pd.DataFrame(index=X.index)
design["Age (per 10 yr)"] = X["age"].astype(float) / 10.0
design["No. of dependents"] = X["num_dependents"].astype(float)
design["Dx-to-treatment (days)"] = X["dx_to_treatment_days"].astype(float)
col_label = {c: c for c in design.columns}
for col in cat.columns:
    ref = REF[col]
    levels = [ref] + [v for v in sorted(cat[col].unique()) if v != ref]
    d = pd.get_dummies(pd.Categorical(cat[col], categories=levels),
                       prefix=col, prefix_sep=": ", drop_first=True)
    for dc in d.columns:
        col_label[dc] = f"{dc} (vs {ref})"
    design = pd.concat([design, d.astype(float)], axis=1)

design = sm.add_constant(design)
res = sm.Logit(y, design).fit(method="bfgs", maxiter=2000, disp=0)
if not res.mle_retvals.get("converged", False):
    res = sm.Logit(y, design).fit(method="newton", maxiter=200, disp=0)

params, cis, pvals = res.params, res.conf_int(), res.pvalues
or_tab = pd.DataFrame({
    "term": [col_label.get(i, i) for i in params.index],
    "odds_ratio": np.exp(params).round(3),
    "ci_low": np.exp(cis[0]).round(3),
    "ci_high": np.exp(cis[1]).round(3),
    "p_value": pvals.round(4),
}).reset_index(drop=True)
or_tab = or_tab[or_tab["term"] != "const"]
or_tab["significant"] = np.where(or_tab["p_value"] < 0.05, "Yes", "No")
or_tab = or_tab.sort_values("odds_ratio", ascending=False).reset_index(drop=True)
or_tab.to_csv(config.RESULTS_DIR / "logistic_regression_odds_ratios.csv", index=False)
print(f"LR(OR model) converged={res.mle_retvals.get('converged')}, pseudo-R2={res.prsquared:.3f}, "
      f"terms={len(or_tab)} (healthcare_worker dropped; ethnicity/education/detection collapsed)")
print("\n--- Significant adjusted odds ratios (p<0.05) ---")
print(or_tab[or_tab.significant == "Yes"][["term", "odds_ratio", "ci_low", "ci_high", "p_value"]].to_string(index=False))

# Forest plot: significant terms + clip extreme CI for display
sig = or_tab[or_tab.significant == "Yes"].copy().sort_values("odds_ratio")
sig["ci_high_disp"] = sig["ci_high"].clip(upper=12)
fig, ax = plt.subplots(figsize=(8.5, max(4, 0.45 * len(sig) + 1.5)))
ypos = np.arange(len(sig))
for i, (_, r) in enumerate(sig.iterrows()):
    c = RISK_C if r["odds_ratio"] > 1 else PROT_C
    ax.plot([r["ci_low"], r["ci_high_disp"]], [i, i], color=c, lw=1.6, zorder=1)
    ax.scatter(r["odds_ratio"], i, color=c, s=45, zorder=2, edgecolor="black", linewidth=0.4)
    txt = f"{r['odds_ratio']:.2f} ({r['ci_low']:.2f}-{r['ci_high']:.2f})"
    ax.text(ax.get_xlim()[1] if False else 13, i, txt, va="center", fontsize=7.5)
ax.axvline(1, color="grey", ls="--", lw=1)
ax.set_yticks(ypos); ax.set_yticklabels(sig["term"], fontsize=8)
ax.set_xscale("log"); ax.set_xlim(0.2, 18)
ax.set_xticks([0.25, 0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
ax.set_xlabel("Adjusted odds ratio (95% CI, log scale)")
# figure-level title removed (caption supplied outside the plot)
fig.savefig(config.FIGURES_DIR / "fig_09_lr_odds_ratios.png")
plt.close(fig)

# ============================================================================
# (b) SHAP (TreeExplainer) on the best tree-based model
# ============================================================================
cv = pd.read_csv(config.RESULTS_DIR / "model_comparison_cv.csv")
cv["auc"] = cv["cv_roc_auc"].str.split(" ").str[0].astype(float)
tree_models = ["Gradient Boosting", "XGBoost", "Random Forest"]
best_tree = cv[cv.model.isin(tree_models)].sort_values("auc", ascending=False).iloc[0]["model"]
print(f"\nBest tree-based model for SHAP: {best_tree}")
pipe = joblib.load(config.MODELS_DIR / f"pipeline_{best_tree.replace(' ', '_')}.pkl")
pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]

raw_names = pre.get_feature_names_out()


def clean_name(raw):
    if raw.startswith("num__"):
        return config.FEATURE_LABELS.get(raw[5:], raw[5:])
    s = raw[5:]
    for f in config.CATEGORICAL_FEATURES:
        if s.startswith(f + "_"):
            return f"{config.FEATURE_LABELS.get(f, f)} = {tr(s[len(f)+1:])}"
    return s


feat_names = [clean_name(r) for r in raw_names]
X_te_t = pre.transform(X_te)

explainer = shap.TreeExplainer(clf)
sv = explainer.shap_values(X_te_t)
if isinstance(sv, list):          # some versions return [class0, class1]
    sv = sv[1]
sv = np.asarray(sv)
if sv.ndim == 3:                  # (n, features, classes)
    sv = sv[:, :, 1]

mean_abs = np.abs(sv).mean(axis=0)
shap_imp = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs.round(4)}) \
    .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
shap_imp.to_csv(config.RESULTS_DIR / "shap_feature_importance.csv", index=False)
print("\n--- Top 12 SHAP features ---")
print(shap_imp.head(12).to_string(index=False))

# Beeswarm
plt.figure()
shap.summary_plot(sv, X_te_t, feature_names=feat_names, max_display=15, show=False)
# figure-level title removed (caption supplied outside the plot)
plt.gcf().set_size_inches(9, 7)
plt.savefig(config.FIGURES_DIR / "fig_10_shap_beeswarm.png", dpi=config.PLOT_DPI, bbox_inches="tight")
plt.close()

# Mean |SHAP| bar
plt.figure()
shap.summary_plot(sv, X_te_t, feature_names=feat_names, plot_type="bar", max_display=15, show=False)
# figure-level title removed (caption supplied outside the plot)
plt.gcf().set_size_inches(8.5, 7)
plt.savefig(config.FIGURES_DIR / "fig_11_shap_bar.png", dpi=config.PLOT_DPI, bbox_inches="tight")
plt.close()

print("\nSaved odds-ratio table + forest plot, SHAP importance CSV, and 2 SHAP figures.")
