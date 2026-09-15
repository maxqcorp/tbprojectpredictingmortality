"""06 - Cohort-selection (STROBE-style) flow diagram -> figures/fig_00_cohort_flow.png"""
import config
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"savefig.dpi": config.PLOT_DPI, "savefig.bbox": "tight", "font.size": 10})
import plotstyle
plotstyle.use_style()
fig, ax = plt.subplots(figsize=(8.5, 8))
ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")


def box(x, y, w, h, text, fc="#EAF2F8", ec="#2C3E50", fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.3, color="#2C3E50"))


box(2.5, 10.3, 5.0, 1.2, "TB patients registered in TBIS\nFederal Territories of Malaysia, 2022\n(N = 3,743)", bold=True)
arrow(5.0, 10.3, 5.0, 8.7)
box(5.7, 8.0, 4.2, 2.0,
    "Excluded (n = 2,265)\n• Still on treatment: 1,799\n• Outcome missing: 429\n• Diagnosis revised, non-TB: 37",
    fc="#FDEDEC", ec="#A93226", fs=9)
arrow(5.0, 8.7, 5.0, 7.2)
box(2.5, 6.0, 5.0, 1.2, "Analytic cohort: concluded TB outcome\n(n = 1,478)", bold=True, fc="#E8F8F5")
arrow(4.2, 6.0, 3.0, 4.6)
arrow(5.8, 6.0, 7.0, 4.6)
box(0.7, 3.2, 4.0, 1.4, "Died during treatment\n(n = 405, 27.4%)", fc="#FDEBD0", ec="#B9770E", bold=True)
box(5.3, 3.2, 4.0, 1.4, "Non-death concluded\noutcome (n = 1,073, 72.6%)", fc="#EAF2F8", bold=True)
arrow(2.7, 3.2, 4.3, 1.9)
arrow(7.3, 3.2, 5.7, 1.9)
box(2.5, 0.6, 5.0, 1.3,
    "80/20 stratified split\nTrain n = 1,182 (324 deaths)\nTest n = 296 (81 deaths)",
    fc="#F4ECF7", ec="#6C3483", fs=9)
# figure-level title removed (caption supplied outside the plot)
fig.savefig(config.FIGURES_DIR / "fig_00_cohort_flow.png")
plt.close(fig)
print("Saved fig_00_cohort_flow.png")
