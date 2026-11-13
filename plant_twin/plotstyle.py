"""Print style for the report figures: Helvetica-metric sans serif, 7 pt, Okabe-Ito colours."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties, findfont  # noqa: E402

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
SANS_CHAIN = ["Helvetica", "Nimbus Sans", "TeX Gyre Heros", "Arimo", "Liberation Sans", "Arial", "DejaVu Sans"]
MAX_HEIGHT_MM = 170.0


def resolve_family():
    for fam in SANS_CHAIN:
        try:
            findfont(FontProperties(family=fam), fallback_to_default=False)
            return fam
        except ValueError:
            continue
    raise RuntimeError("no sans-serif font found")


def use_print_style():
    fam = resolve_family()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [fam] + SANS_CHAIN, "font.size": 7,
        "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "custom", "mathtext.rm": fam,
        "axes.prop_cycle": plt.cycler(color=OKABE_ITO), "savefig.dpi": 600,
    })
    return fam


def panel_labels(axes, letters=None, *, dx=-0.16, dy=1.04, size=8.0):
    letters = letters or [chr(65 + i) for i in range(len(axes))]
    for ax, letter in zip(axes, letters):
        ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size, fontweight="bold", va="bottom", ha="left")
