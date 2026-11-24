"""Figure: (a) outcomes of four stopping rules on the 111 dated real clips, by holding posture; (b) where a clip's
processing time goes per rule (mean localisation and mean VLM-read time per clip, all 148 clips), with the
median total. Data: real_breakdown.json, computed from quest_sim/real/replay_time.py frame records.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from style import AQUA, BLUE, INK, INK2, ORANGE, TEXT_W, VIOLET, panel_label

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "real_breakdown.json"))
NONE = "#c9c8c3"
RULES = ["VLM repeat", "Gated VLM repeat", "OCR repeat", "Cross-check"]
POST = [("a", "bent arm"), ("b", "close"), ("d", "straight")]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(TEXT_W, 1.95), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.75))

# (a) outcomes by posture: grouped by rule, one stacked bar per posture
y, ticks, labels = 0.4, [], []
for r in RULES:
    ax.text(0.2, y - 0.95, r, ha="left", va="center", fontsize=6.4, color=INK, fontweight="bold")
    for c, name in POST:
        o = D[r]["post"][c]
        left = 0
        for key, col in (("correct", AQUA), ("wrong", VIOLET), ("none", NONE)):
            v = o.get(key, 0)
            ax.barh(y, v, left=left, color=col, height=0.78, edgecolor="white", linewidth=0.6)
            if v >= 4:
                ax.text(left + v / 2, y, str(v), ha="center", va="center", fontsize=5.6,
                        color="white" if key != "none" else INK2)
            left += v
        ticks.append(y)
        labels.append(name)
        y += 1
    y += 1.3
ax.set_yticks(ticks)
ax.set_yticklabels(labels, fontsize=6)
ax.invert_yaxis()
ax.set_xlim(0, 37)
ax.set_xlabel("Dated clips (of 37 per posture)")
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
for key, col, lab in (("correct", AQUA, "correct"), ("wrong", VIOLET, "wrong"), ("none", NONE, "no date or abstain")):
    ax.bar(0, 0, color=col, label=lab)
ax.legend(loc="upper center", bbox_to_anchor=(0.6, 1.2), ncol=3, fontsize=6.3, handlelength=1.0, columnspacing=1.0)
panel_label(ax, "a")

# (b) time per clip: mean localisation + mean VLM reads, with median total
rows = ["VLM repeat", "Gated VLM repeat", "OCR repeat", "Cross-check (as specified)", "Cross-check"]
names = ["VLM repeat", "Gated VLM repeat", "OCR repeat", "Cross-check, as specified", "Cross-check, OCR-date skip"]
for i, (r, n) in enumerate(zip(rows, names)):
    d = D[r]
    bx.barh(i, d["loc_mean"], color=ORANGE, height=0.62, edgecolor="white", linewidth=0.6)
    bx.barh(i, d["read_mean"], left=d["loc_mean"], color=BLUE, height=0.62, edgecolor="white", linewidth=0.6)
    tot = d["loc_mean"] + d["read_mean"]
    bx.text(tot + 0.8, i, f"median {d['med']:.1f} s", va="center", fontsize=6, color=INK2)
bx.set_yticks(range(len(names)))
bx.set_yticklabels(names, fontsize=6.3)
bx.invert_yaxis()
bx.set_xlim(0, 62)
bx.set_xlabel("Mean processing time per clip (s)")
bx.grid(axis="y", visible=False)
bx.tick_params(axis="y", length=0)
bx.bar(0, 0, color=ORANGE, label="localisation (CPU)")
bx.bar(0, 0, color=BLUE, label="VLM reads (GPU)")
bx.legend(loc="upper center", bbox_to_anchor=(0.6, 1.2), ncol=2, fontsize=6.3, handlelength=1.0, columnspacing=1.0)
panel_label(bx, "b")

fig.savefig(HERE.parent / "real_breakdown.pdf")
fig.savefig(HERE.parent / "real_breakdown.png", dpi=300)
print("saved")
