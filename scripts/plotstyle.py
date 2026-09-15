"""
Shared MATLAB-style figure formatting for every chart in this project.

Import and call once near the top of a plotting script:

    import plotstyle
    plotstyle.use_style()

It (1) sets global rcParams for a clean white, black-bordered, outward-tick look,
and (2) patches Figure.savefig so that, at export time, EVERY visible axes in the
figure (including those produced by seaborn or shap, which strip spines) is given
a complete rectangular black border on all four sides, outward ticks, a white
background, and no grid. Axes turned off with axis('off') (e.g. a flow diagram)
are left untouched. Figures are written at 300 dpi with a tight bounding box.
"""
import matplotlib
from matplotlib.figure import Figure

BORDER_W = 1.0          # consistent border / tick line width across all figures
TICK_LEN = 4.0

_orig_savefig = Figure.savefig


def _style_ax(ax):
    if not getattr(ax, "axison", True):
        return                                   # skip axis('off') axes
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(BORDER_W)
    # direction/colour/width only -- do NOT toggle sides, so colorbar ticks survive
    ax.tick_params(which="both", direction="out", color="black",
                   width=BORDER_W, length=TICK_LEN, labelcolor="black")
    ax.grid(False)


def _patched_savefig(self, *args, **kwargs):
    self.patch.set_facecolor("white")
    for ax in self.axes:
        _style_ax(ax)
    kwargs.setdefault("dpi", 300)
    kwargs.setdefault("facecolor", "white")
    kwargs.setdefault("edgecolor", "white")
    if "bbox_inches" not in kwargs:
        kwargs["bbox_inches"] = "tight"
    kwargs.setdefault("pad_inches", 0.08)
    return _orig_savefig(self, *args, **kwargs)


def use_style():
    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": BORDER_W,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.grid": False,
        "axes.axisbelow": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "xtick.major.size": TICK_LEN,
        "ytick.major.size": TICK_LEN,
        "xtick.major.width": BORDER_W,
        "ytick.major.width": BORDER_W,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.labelcolor": "black",
        "text.color": "black",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.facecolor": "white",
        "legend.framealpha": 1.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    if not getattr(Figure, "_matlab_patched", False):
        Figure.savefig = _patched_savefig
        Figure._matlab_patched = True
