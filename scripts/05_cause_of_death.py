"""
05 - Secondary descriptive analysis of CAUSE OF DEATH among cohort deaths.

These fields ('Sebab Kematian', 'Penyumbang kematian, ...') are death-only and
were deliberately excluded from the prediction features (they leak the outcome).
Here they are summarised descriptively for the 405 deaths in the analytic cohort.

Outputs:
  results/cause_of_death_summary.csv
  figures/fig_12_cause_of_death.png
"""
import config  # sets thread env vars first
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
import plotstyle
plotstyle.use_style()
plt.rcParams.update({"savefig.dpi": config.PLOT_DPI, "savefig.bbox": "tight",
                     "axes.titleweight": "bold", "font.size": 10})

df = pd.read_excel(config.RAW_XLSX)
outcome = df[config.OUTCOME_COL].astype("string").str.strip()
deaths = df[outcome == "Mati"].copy()
N_DEATH = len(deaths)
print(f"Cohort deaths (Mati): {N_DEATH}")

# ----------------------------------------------------------------------------
# 1. Cause-of-death recording completeness + TB-relatedness
#    'Sebab Kematian' (TBIS): Y = TB-related death, N = non-TB-related death.
# ----------------------------------------------------------------------------
sk = deaths["Sebab Kematian"].astype("string").str.strip()
recorded = sk.notna().sum()
y = (sk == "Y").sum()
n = (sk == "N").sum()
print(f"Cause-of-death recorded: {recorded}/{N_DEATH} ({100*recorded/N_DEATH:.1f}%)")
print(f"  TB-related (Y): {y}   non-TB-related (N): {n}")

summary = [
    {"category": "Deaths in cohort", "n": N_DEATH, "pct_of_deaths": 100.0},
    {"category": "Cause-of-death recorded", "n": int(recorded),
     "pct_of_deaths": round(100 * recorded / N_DEATH, 1)},
    {"category": "TB-related death (Y)", "n": int(y),
     "pct_of_recorded": round(100 * y / recorded, 1) if recorded else 0},
    {"category": "Non-TB-related death (N)", "n": int(n),
     "pct_of_recorded": round(100 * n / recorded, 1) if recorded else 0},
]

# ----------------------------------------------------------------------------
# 2. Contributing factors to death (multi-select flags among recorded deaths)
# ----------------------------------------------------------------------------
CONTRIB = {
    "Penyumbang kematian, Lewat Diagnosa": "Late diagnosis",
    "Penyumbang kematian, Kegagalan Multi-sistem": "Multi-system failure",
    "Penyumbang kematian, Jangkitan Sekunder": "Secondary infection",
    "Penyumbang kematian, Salah pengendalian": "Mismanagement",
    "Penyumbang kematian, Lain-lain": "Other",
}
contrib_rows = []
for col, lab in CONTRIB.items():
    c = (deaths[col] == 1).sum()
    contrib_rows.append({"contributor": lab, "n": int(c),
                         "pct_of_recorded": round(100 * c / recorded, 1) if recorded else 0})
    summary.append({"category": f"Contributor: {lab}", "n": int(c),
                    "pct_of_recorded": round(100 * c / recorded, 1) if recorded else 0})
contrib_df = pd.DataFrame(contrib_rows).sort_values("n", ascending=False)

pd.DataFrame(summary).to_csv(config.RESULTS_DIR / "cause_of_death_summary.csv", index=False)
print("\n--- Contributing factors (of recorded deaths) ---")
print(contrib_df.to_string(index=False))

# ----------------------------------------------------------------------------
# Figure 12 - cause of death (2 panels)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

# (a) TB-related vs non-TB vs unrecorded
labels = ["TB-related\n(Y)", "Non-TB-related\n(N)", "Not recorded"]
vals = [y, n, N_DEATH - recorded]
cols = ["#C0392B", "#E59866", "#BDC3C7"]
bars = axes[0].bar(labels, vals, color=cols, edgecolor="black", linewidth=0.5)
for b, v in zip(bars, vals):
    axes[0].text(b.get_x() + b.get_width() / 2, v + 3,
                 f"{v}\n({100*v/N_DEATH:.1f}%)", ha="center", va="bottom",
                 fontweight="bold", fontsize=9)
axes[0].set_ylabel("Number of deaths")
axes[0].set_title(f"Cause-of-death classification (all {N_DEATH} cohort deaths)")
axes[0].set_ylim(0, max(vals) * 1.25)

# (b) contributing factors
bars = axes[1].barh(contrib_df["contributor"][::-1], contrib_df["n"][::-1],
                    color="#922B21", edgecolor="black", linewidth=0.4)
for b, v, p in zip(bars, contrib_df["n"][::-1], contrib_df["pct_of_recorded"][::-1]):
    axes[1].text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v} ({p:.0f}%)",
                 va="center", fontsize=8.5)
axes[1].set_xlabel("Number of deaths")
axes[1].set_title(f"Contributing factors to death\n(of {recorded} deaths with cause recorded)")
axes[1].set_xlim(0, contrib_df["n"].max() * 1.25)
# figure-level title removed (caption supplied outside the plot)
fig.savefig(config.FIGURES_DIR / "fig_12_cause_of_death.png")
plt.close(fig)
print("\nSaved cause_of_death_summary.csv and fig_12_cause_of_death.png")
