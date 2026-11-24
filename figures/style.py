"""Shared figure style for the paper: ACM-matching sans font (Linux Biolinum), 7.5 pt text, thin marks,
recessive axes and grid, one validated categorical palette (blue, orange; aqua only with direct labels)."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm

TEX = Path.home() / ".TinyTeX/texmf-dist/fonts/opentype/public/libertine"
for f in ("LinBiolinum_R.otf", "LinBiolinum_RB.otf", "LinBiolinum_RI.otf"):
    if (TEX / f).exists():
        fm.fontManager.addfont(str(TEX / f))

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # validated: CVD dE >= 9.2, normal-vision >= 27.6
VIOLET, YELLOW = "#6250d6", "#eda100"                    # stage pair: CVD dE 37.9, normal 42.1 (yellow needs labels)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL_W, TEXT_W = 3.33, 7.0                                   # ACM sigconf column / text width (in)

mpl.rcParams.update({
    "font.family": "Linux Biolinum O", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def panel_label(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, ha="right", va="bottom", fontweight="bold", fontsize=8)
