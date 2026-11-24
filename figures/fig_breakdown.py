"""Figure 2: where a read's time goes on the Quest 3S (revision latency runs: 40 seeded crops x 3 rounds).
(a) prefill time (vision encoder + prompt) against prompt tokens, vision encoder on the CPU vs. on the GPU with
    the tiled kernel (default clocks); one point per crop, mean of 3 rounds.
(b) mean time per read split into prefill and decode, per configuration (means add up; medians would not)."""
import json
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import BLUE, ORANGE, VIOLET, YELLOW, INK, INK2, TEXT_W, panel_label

R = Path.home() / "qwen/phone_bench/dvfs/revision"
OUT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for k in (1, 2, 3) for l in open(R / f"latency_r{k}.jsonl")]


def per_image(cfg):
    by = {}
    for r in rows:
        if r["config"] == cfg:
            by.setdefault(r["image"], []).append(r)
    return {i: dict(tok=v[0]["prompt_n"], pf=st.mean(x["prompt_ms"] for x in v) / 1e3,
                    dc=st.mean(x["decode_ms"] for x in v) / 1e3) for i, v in by.items()}


fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_W, 1.75), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.42})

# (a) prefill vs prompt tokens
for cfg, col, mk, lab in (("orig", ORANGE, "o", "Encoder on CPU (stock)"),
                          ("+gpu_vit", BLUE, "s", "Encoder on GPU, tiled kernel")):
    d = per_image(cfg)
    x = np.array([v["tok"] for v in d.values()]); y = np.array([v["pf"] for v in d.values()])
    a.scatter(x, y, s=11, marker=mk, color=col, edgecolors="white", linewidths=0.5, zorder=3, label=lab)
a.set_xlabel("Prompt tokens (visual + text)")
a.set_ylabel("Prefill incl. vision encoder (s)")
a.set_title("Prefill grows with prompt length (default clocks)", loc="left", fontsize=7.5, color=INK, pad=4)
a.set_xlim(200, 1220); a.set_ylim(0, None)
a.legend(loc="upper left", handletextpad=0.2, borderaxespad=0.2)
panel_label(a, "a")

# (b) mean time per read, prefill + decode; group headers as rows
from matplotlib.patches import Patch
C = [("H", "Default clocks"), ("orig", "Baseline"), ("+gpu_vit", "+ tiled ViT kernel"),
     ("+task_vocab", "+ task-vocab. head"), ("+stop_if", "+ early stop"),
     ("H", "Maximum clocks"), ("orig_max", "Baseline"), ("+stop_if_max", "+ all three"),
     ("+cap384_if_max", "+ 384-token cap")]
ys, labels = [], []
y = 0
for cfg, lab in C:
    if cfg == "H":
        if y:
            y += 0.35
        b.text(0, y, lab, va="center", ha="left", fontsize=7, fontstyle="italic", color=INK2)
        y += 1
        continue
    d = per_image(cfg)
    pf = st.mean(v["pf"] for v in d.values()); dc = st.mean(v["dc"] for v in d.values())
    b.barh(y, pf, height=0.62, color=VIOLET, zorder=3)
    b.barh(y, dc, left=pf + 0.06, height=0.62, color=YELLOW, zorder=3)        # 0.06 s gap separates segments
    if cfg == "orig":                                                          # direct stage labels on the first bar
        b.text(pf / 2, y, "prefill", ha="center", va="center", fontsize=6.5, color="white", zorder=4)
        b.text(pf + 0.06 + dc / 2, y, "decode", ha="center", va="center", fontsize=6.5, color=INK, zorder=4)
    b.text(pf + dc + 0.25, y, f"{pf + dc:.1f}", va="center", ha="left", fontsize=6.5, color=INK2)
    ys.append(y); labels.append(lab)
    y += 1
b.set_yticks(ys); b.set_yticklabels(labels)
b.set_ylim(y - 0.4, -0.6)
b.tick_params(axis="y", length=0)
b.grid(axis="y", visible=False)
b.set_xlabel("Mean time per read (s)")
b.set_xlim(0, 12)
b.set_title("Mean time per read as the changes are applied", loc="left", fontsize=7.5, color=INK, pad=4)
b.legend(handles=[Patch(color=VIOLET, label="Prefill (vision encoder + prompt)"), Patch(color=YELLOW, label="Decode")],
         loc="lower right", handlelength=1.0, handletextpad=0.3, borderaxespad=0.1)
panel_label(b, "b")

fig.savefig(OUT / "breakdown.pdf")
fig.savefig(OUT / "breakdown.png", dpi=220)
print("saved", OUT / "breakdown.pdf")
