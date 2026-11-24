"""Figure: precision against coverage of the stopping rules on the 37 held-out real products (Table 3), with the
OCR-only variants and the PP-OCRv6 tiers of the OCR cross-check, annotated with median serial processing time.

Numbers: quest_sim/real/replay_time.py (PP-OCRv5 rows and times), eval_v6.py (PP-OCRv6 tiers),
OCR repeat with 3 / 4 agreeing OCR dates from the same frame records. Coverage: emitted dates on the 111 dated
clips; precision: correct over all emitted dates (dated clips and no-date sides).
"""
from pathlib import Path

import matplotlib.pyplot as plt

from style import BLUE, INK, INK2, ORANGE, VIOLET, COL_W

OUT = Path(__file__).resolve().parent.parent
# (label, coverage %, precision %, median time s or None)
VLM = [("VLM repeat", 79.3, 60.6, 26.2), ("Gated VLM repeat", 78.4, 64.3, 22.2)]
OCR = [("OCR repeat, 2", 46.8, 94.2, 18.9), ("3", 43.2, 97.9, 19.4), ("4", 39.6, 100.0, 20.4)]
XCHK = ("OCR cross-check", 47.7, 100.0, 19.2)
XV6 = [("v6 tiny", 41.4, 95.7), ("v6 small", 48.6, 94.4), ("v6 medium", 56.8, 93.8)]

fig, ax = plt.subplots(figsize=(COL_W, 1.95))
ax.scatter([c for _, c, _, _ in VLM], [p for _, _, p, _ in VLM], s=26, marker="o", color=BLUE, zorder=3,
           edgecolors="white", linewidths=0.8, label="VLM repeat")
ax.plot([c for _, c, _, _ in OCR], [p for _, _, p, _ in OCR], color=ORANGE, lw=0.8, zorder=2)
ax.scatter([c for _, c, _, _ in OCR], [p for _, _, p, _ in OCR], s=24, marker="s", color=ORANGE, zorder=3,
           edgecolors="white", linewidths=0.8, label="OCR repeat")
ax.scatter([c for _, c, _ in XV6], [p for _, _, p in XV6], s=24, marker="D", facecolors="white", edgecolors=VIOLET,
           linewidths=1.0, zorder=3, label="Cross-check, PP-OCRv6")
ax.scatter([XCHK[1]], [XCHK[2]], s=40, marker="D", color=VIOLET, zorder=4, edgecolors="white", linewidths=0.8,
           label="Cross-check, PP-OCRv5")

kw = dict(fontsize=6.5, color=INK2)
ax.annotate("VLM repeat, 26.2 s", (79.3, 60.6), xytext=(-4, -9), textcoords="offset points", ha="right", **kw)
ax.annotate("gated VLM repeat, 22.2 s", (78.4, 64.3), xytext=(2, 5), textcoords="offset points", ha="right", **kw)
ax.annotate("OCR repeat (2 / 3 / 4 frames):\n18.9 / 19.4 / 20.4 s", (43.2, 97.9), xytext=(-40, -30),
            textcoords="offset points", ha="left", **kw)
ax.annotate("cross-check, 19.2 s", (47.7, 100.0), xytext=(6, 2), textcoords="offset points", ha="left",
            fontsize=6.5, color=INK, fontweight="bold")
ax.annotate("v6 tiers", (56.8, 93.8), xytext=(5, -3), textcoords="offset points", ha="left", **kw)

ax.set_xlim(30, 90)
ax.set_ylim(55, 103)
ax.set_xlabel("Coverage of dated clips (%)")
ax.set_ylabel("Precision of emitted dates (%)")
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.6), handletextpad=0.3, borderaxespad=0.2, labelspacing=0.35, fontsize=6.5)
fig.savefig(OUT / "real_tradeoff.pdf")
fig.savefig(OUT / "real_tradeoff.png", dpi=300)
print("saved", OUT / "real_tradeoff.pdf")
