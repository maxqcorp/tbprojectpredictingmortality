"""
02 - Exploratory data analysis (publication-quality, 300 DPI).

Outputs (figures/):
  fig_01_target_distribution.png
  fig_02_age_by_outcome.png
  fig_03_demographics_by_outcome.png
  fig_04_clinical_by_outcome.png
  fig_05_association_ranking.png
Outputs (results/):
  table1_baseline_characteristics.csv
  eda_association_ranking.csv
"""
import config  # sets thread env vars first
import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
import plotstyle
plotstyle.use_style()
plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold",
                     "figure.dpi": 110, "savefig.dpi": config.PLOT_DPI,
                     "savefig.bbox": "tight"})

SURV_C, DIED_C = "#4C9F70", "#C0392B"   # green = survived, red = died
df = config.load_processed()
N = len(df)
OVERALL = df[config.TARGET].mean()
print(f"Cohort n={N}, overall death rate={100*OVERALL:.1f}%")


def tr(x):
    """Translate a BM category value to English for display."""
    return config.VALUE_TRANSLATE.get(str(x), str(x))


def cramers_v(ct):
    chi2 = ss.chi2_contingency(ct)[0]
    n = ct.to_numpy().sum()
    phi2 = chi2 / n
    r, k = ct.shape
    phi2c = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rc = r - ((r - 1) ** 2) / (n - 1)
    kc = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kc - 1), (rc - 1))
    return np.sqrt(phi2c / denom) if denom > 0 else 0.0


# Fixed display order for ordinal-ish variables
ORDER = {
    "cxr_severity": ["No Lesion", "Sederhana (Minimal)", "Teruk (Moderately Advanced)",
                     "Sangat Teruk (Far Advanced)", "Tidak Dibuat"],
    "education": ["Tiada", "Sekolah Rendah", "Tingkatan 1/2/3", "Tingkatan 4/5",
                  "Tingkatan 6/Diploma/Sijil", "Ijazah", "Lain-lain (nyatakan)"],
    "age_band": ["Kurang 1 Tahun", "1-4 Tahun", "5-14 Tahun", "15-24 Tahun", "25-34 Tahun",
                 "35-44 Tahun", "45-54 Tahun", "55-64 Tahun", "Lebih 65 Tahun"],
}
AGE_BAND_EN = {"Kurang 1 Tahun": "<1", "1-4 Tahun": "1-4", "5-14 Tahun": "5-14",
               "15-24 Tahun": "15-24", "25-34 Tahun": "25-34", "35-44 Tahun": "35-44",
               "45-54 Tahun": "45-54", "55-64 Tahun": "55-64", "Lebih 65 Tahun": "65+"}


def cat_order(col):
    if col in ORDER:
        cats = [c for c in ORDER[col] if c in df[col].unique()]
    else:  # order by death rate descending
        cats = df.groupby(col)[config.TARGET].mean().sort_values(ascending=False).index.tolist()
    return cats


def death_rate_panel(ax, col, rotate=0):
    cats = cat_order(col)
    g = df.groupby(col)[config.TARGET].agg(["mean", "sum", "count"]).reindex(cats)
    rates = 100 * g["mean"].values
    if col == "age_band":
        labels = [AGE_BAND_EN.get(c, c) for c in cats]
    else:
        labels = [tr(c) for c in cats]
    bars = ax.bar(range(len(cats)), rates, color=DIED_C, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.axhline(100 * OVERALL, color="grey", ls="--", lw=1)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center", fontsize=8)
    ax.set_ylabel("Death rate (%)", fontsize=8)
    ax.set_title(config.FEATURE_LABELS.get(col, col), fontsize=9)
    ax.set_ylim(0, min(100, max(rates.max() * 1.25, 5)))
    for b, r, nd, nt in zip(bars, rates, g["sum"].values, g["count"].values):
        ax.text(b.get_x() + b.get_width() / 2, r + ax.get_ylim()[1] * 0.02,
                f"{r:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, ax.get_ylim()[1] * 0.02,
                f"n={int(nt)}", ha="center", va="bottom", fontsize=6, color="white")


# ----------------------------------------------------------------------------
# Figure 1 - Target distribution
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4.2))
counts = df[config.TARGET].value_counts().reindex([0, 1])
bars = ax.bar(["Non-death\nconcluded outcome", "Died"], counts.values,
              color=[SURV_C, DIED_C], edgecolor="black", linewidth=0.5)
for b, c in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, c + 12,
            f"{c}\n({100*c/N:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=10)
ax.set_ylabel("Number of patients")
# figure-level title removed (caption supplied outside the plot)
ax.set_ylim(0, counts.max() * 1.18)
fig.savefig(config.FIGURES_DIR / "fig_01_target_distribution.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# Figure 2 - Age by outcome
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
# (a) age distribution by outcome
for lab, c, name in [(0, SURV_C, "Survived"), (1, DIED_C, "Died")]:
    sns.kdeplot(df.loc[df[config.TARGET] == lab, "age"], ax=axes[0], fill=True,
                color=c, alpha=0.4, label=name, linewidth=1.5)
axes[0].set_xlabel("Age (years)"); axes[0].set_title("Age distribution by outcome")
axes[0].legend(frameon=False, fontsize=8)
m0, m1 = df.loc[df[config.TARGET] == 0, "age"].median(), df.loc[df[config.TARGET] == 1, "age"].median()
axes[0].axvline(m0, color=SURV_C, ls="--", lw=1); axes[0].axvline(m1, color=DIED_C, ls="--", lw=1)
axes[0].text(0.02, 0.95, f"Median age\nSurvived {m0:.0f} vs Died {m1:.0f}",
             transform=axes[0].transAxes, va="top", fontsize=8)
# (b) death rate by age band
death_rate_panel(axes[1], "age_band")
axes[1].set_title("Death rate by age band")
axes[1].set_xlabel("Age band (years)")
# figure-level title removed (caption supplied outside the plot)
fig.savefig(config.FIGURES_DIR / "fig_02_age_by_outcome.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# Figure 3 - Demographics / socioeconomic by outcome
# ----------------------------------------------------------------------------
demo = ["sex", "citizenship", "residence_location", "ethnicity", "education", "income_status"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
for ax, col in zip(axes.ravel(), demo):
    rot = 35 if col in ("ethnicity", "education") else 0
    death_rate_panel(ax, col, rotate=rot)
# figure-level title removed (caption supplied outside the plot)
fig.savefig(config.FIGURES_DIR / "fig_03_demographics_by_outcome.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# Figure 4 - Clinical / comorbidity by outcome
# ----------------------------------------------------------------------------
clin = ["diabetes", "smoking", "hiv_status", "bcg_scar", "cxr_severity", "tb_site",
        "tb_case_category", "baseline_sputum_smear", "detection_method",
        "tb_meningitis", "tb_miliary", "healthcare_worker"]
fig, axes = plt.subplots(3, 4, figsize=(17, 11.5), constrained_layout=True)
for ax, col in zip(axes.ravel(), clin):
    rot = 30 if col in ("cxr_severity", "tb_site", "tb_case_category") else 0
    death_rate_panel(ax, col, rotate=rot)
# figure-level title removed (caption supplied outside the plot)
fig.savefig(config.FIGURES_DIR / "fig_04_clinical_by_outcome.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# Association ranking (chi-square + Cramer's V) + Table 1
# ----------------------------------------------------------------------------
# Discretise numeric features for the association measure
disc = df.copy()
disc["age_grp"] = disc["age_band"]
disc["dep_grp"] = pd.cut(disc["num_dependents"], [-1, 0, 2, 100], labels=["0", "1-2", "3+"])
disc["dx_grp"] = pd.cut(disc["dx_to_treatment_days"], [-1, 0, 7, 30, 9999],
                        labels=["0", "1-7", "8-30", ">30"])
assoc_map = {c: c for c in config.CATEGORICAL_FEATURES}
assoc_map.update({"age": "age_grp", "num_dependents": "dep_grp", "dx_to_treatment_days": "dx_grp"})

rows = []
for feat, col in assoc_map.items():
    ct = pd.crosstab(disc[col], disc[config.TARGET])
    chi2, p, dof, _ = ss.chi2_contingency(ct)
    rows.append({"feature": feat, "label": config.FEATURE_LABELS.get(feat, feat),
                 "chi2": round(chi2, 2), "dof": int(dof), "p_value": p,
                 "cramers_v": round(cramers_v(ct), 3),
                 "significant": "Yes" if p < 0.05 else "No"})
assoc = pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)
assoc.to_csv(config.RESULTS_DIR / "eda_association_ranking.csv", index=False)
print("\n--- Association ranking (Cramer's V) ---")
print(assoc[["label", "chi2", "p_value", "cramers_v", "significant"]].to_string(index=False))

# Figure 5 - association ranking
fig, ax = plt.subplots(figsize=(8, 7))
a = assoc.iloc[::-1]
colors = [DIED_C if s == "Yes" else "#95a5a6" for s in a["significant"]]
bars = ax.barh(a["label"], a["cramers_v"], color=colors, edgecolor="black", linewidth=0.4)
for b, v, p in zip(bars, a["cramers_v"], a["p_value"]):
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax.text(v + 0.003, b.get_y() + b.get_height() / 2, f"{v:.3f} {star}",
            va="center", fontsize=7.5)
ax.set_xlabel("Cramér's V (association with mortality)")
# figure-level title removed (caption supplied outside the plot)
ax.set_xlim(0, a["cramers_v"].max() * 1.18)
fig.savefig(config.FIGURES_DIR / "fig_05_association_ranking.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# Table 1 - baseline characteristics by survival status
# ----------------------------------------------------------------------------
t1 = []
surv = df[df[config.TARGET] == 0]; died = df[df[config.TARGET] == 1]
for col in config.NUMERIC_FEATURES:
    p = ss.mannwhitneyu(surv[col], died[col]).pvalue
    t1.append({"variable": config.FEATURE_LABELS[col], "category": "median [IQR]",
               "overall": f"{df[col].median():.0f} [{df[col].quantile(.25):.0f}-{df[col].quantile(.75):.0f}]",
               "survived": f"{surv[col].median():.0f} [{surv[col].quantile(.25):.0f}-{surv[col].quantile(.75):.0f}]",
               "died": f"{died[col].median():.0f} [{died[col].quantile(.25):.0f}-{died[col].quantile(.75):.0f}]",
               "p_value": round(p, 4)})
for col in config.CATEGORICAL_FEATURES:
    ct = pd.crosstab(df[col], df[config.TARGET])
    p = ss.chi2_contingency(ct)[1]
    first = True
    for cat in cat_order(col):
        o = (df[col] == cat).sum(); s = (surv[col] == cat).sum(); d = (died[col] == cat).sum()
        t1.append({"variable": config.FEATURE_LABELS[col] if first else "",
                   "category": tr(cat),
                   "overall": f"{o} ({100*o/N:.1f}%)",
                   "survived": f"{s} ({100*s/len(surv):.1f}%)",
                   "died": f"{d} ({100*d/len(died):.1f}%)",
                   "p_value": round(p, 4) if first else ""})
        first = False
t1_df = pd.DataFrame(t1)
t1_df.to_csv(config.RESULTS_DIR / "table1_baseline_characteristics.csv", index=False)
print(f"\nSaved Table 1 ({len(t1_df)} rows), association ranking, and 5 figures.")
