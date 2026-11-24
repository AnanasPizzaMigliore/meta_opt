"""Figure: latency per product vs exact-date accuracy for single-frame and multi-frame reading on the Quest 3S.

Accuracy: reads of 7 simulated frames per product (frames_m, on-device text detector), 545 test products.
Latency per product = sum over the frames a policy reads of
    localisation (text detection + recognition of every detected line)  +  one VLM read,
with each term from models fitted to measurements on the headset:
    read(prompt tokens)   quadratic fit to 78 headset reads (e2e_if40.jsonl, max clocks; R^2 0.875, MAE 0.48 s)
    detection(pixels)     0.0254 s + 5.31e-7 s/pixel        (ONNX Runtime, 4 big cores)
    recognition(line)     5.6 ms + 0.21 ms per pixel of width at height 48
Visual tokens per crop follow llama.cpp's resize to the token cap; the text part of the prompt is 185 tokens.

  python fig_latency_accuracy.py  ->  <paper>/figures/latency_accuracy.{pdf,png} and .csv (table view)
"""
import csv
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
OUT = Path("/home/user/NOSSDAV 2027/paper/figures")
K = 7
sys.argv = sys.argv[:1]
spec = importlib.util.spec_from_file_location("ev", "/home/user/qwen/eval_gguf.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
files = [e["filename"] for e in json.load(open(ev.TEST_JSON))]
gt = {e["filename"]: e for e in json.load(open(ev.GT_FILE))}
F = ("year", "month", "day")
dated = [f for f in files if any(gt[f].get(k) is not None for k in F)]

# ---- cost models (fitted on headset measurements) ----
rows = [json.loads(l) for l in open("/home/user/qwen/phone_bench/dvfs/e2e_if40.jsonl")]
xy = [(r["prompt_n"], r["read_s"]) for r in rows if r["config"] in ("+stop_if_max", "+cap384_if_max")]
READ = np.polyfit([a for a, _ in xy], [b for _, b in xy], 2)
TEXT_TOKENS = 185
det_s = lambda w, h: 0.0254 + 5.31e-7 * w * h
rec_s = lambda b: (5.6 + 0.21 * 48 * (b[2] - b[0]) / max(b[3] - b[1], 1)) / 1e3


def vis_tokens(w, h, cap):
    t = math.ceil(w / 32) * math.ceil(h / 32)
    s = 1.0
    while t > cap:
        s = min(s, math.sqrt(cap / t)) * 0.99
        t = math.ceil(w * s / 32) * math.ceil(h * s / 32)
    return max(t, 64)


metas = [json.load(open(HERE / "frames_m" / f"f{k}_meta.json")) for k in range(K)]
crop_wh = [{f: Image.open(HERE / "frames_m" / f"f{k}" / f).size for f in files} for k in range(K)]


def frame_cost(f, k, cap):
    m = metas[k][f]
    loc = det_s(*m["frame_wh"]) + sum(rec_s(b) for _, b, _ in m["all_lines"])
    read = float(np.polyval(READ, vis_tokens(*crop_wh[k][f], cap) + TEXT_TOKENS))
    return loc + read


tri = lambda r: tuple(r.get(k) for k in F)


def plurality(ts):
    c = Counter(ts)
    top = max(c.values())
    return next(t for t in ts if c[t] == top)


def policies(reads):
    """policy -> (answers, frames read per product)"""
    out = {"1 frame": ({f: tri(reads[0][f]) for f in files}, {f: 1 for f in files})}
    for n in (3, 5, 7):
        out[f"vote {n}"] = ({f: plurality([tri(reads[k][f]) for k in range(n)]) for f in files}, {f: n for f in files})
    ans, used = {}, {}
    for f in files:
        seen = Counter()
        ans[f], used[f] = plurality([tri(reads[k][f]) for k in range(K)]), K
        for k in range(K):
            seen[tri(reads[k][f])] += 1
            if seen[tri(reads[k][f])] == 2:
                ans[f], used[f] = tri(reads[k][f]), k + 1
                break
    out["until 2 agree"] = (ans, used)
    return out


def exact(answers):
    res = {f: dict(zip(F, a)) for f, a in answers.items()}
    s = ev.compute_accuracy(res, gt, files)["ALL"]
    return 100 * s["full_correct"] / s["gt_has_date"]


table = []
for cap in (1024, 384):
    reads = [json.load(open(HERE / "reads" / f"m_cap{cap}_f{k}.json")) for k in range(K)]
    for name, (ans, used) in policies(reads).items():
        if name == "1 frame":   # average over the 7 frames, pooling their latencies
            acc = float(np.mean([exact({f: tri(reads[k][f]) for f in files}) for k in range(K)]))
            lat = sorted(frame_cost(f, k, cap) for f in files for k in range(K))
            table.append(dict(cap=cap, policy=name, exact=round(acc, 1), median_s=round(float(np.median(lat)), 2),
                              p90_s=round(lat[int(0.9 * (len(lat) - 1))], 2), mean_frames=1.0))
            continue
        lat = sorted(sum(frame_cost(f, k, cap) for k in range(used[f])) for f in files)
        table.append(dict(cap=cap, policy=name, exact=round(exact(ans), 1),
                          median_s=round(float(np.median(lat)), 2), p90_s=round(lat[int(0.9 * (len(lat) - 1))], 2),
                          mean_frames=round(float(np.mean(list(used.values()))), 2)))

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "latency_accuracy.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(table[0]))
    w.writeheader()
    w.writerows(table)
for r in table:
    print(r)

# ---- plot: one axis, two series, thin marks, direct labels, recessive grid ----
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {1024: ("#2a78d6", "o", "1024 visual-token cap"), 384: ("#eb6834", "s", "384 visual-token cap")}
import sys as _sys
_sys.path.insert(0, "/home/user/NOSSDAV 2027/paper/figures/src")
import style  # noqa: E402,F401  shared paper style (Linux Biolinum O, thin axes, recessive grid)
fig, ax = plt.subplots(figsize=(3.33, 2.35))
ORDER = ["1 frame", "until 2 agree", "vote 3", "vote 5", "vote 7"]
# label offsets (points): policies are discrete choices, so no connecting line; labels placed to avoid overlap
OFF = {1024: {"1 frame": (0, 6, "center"), "until 2 agree": (-5, 6, "right"), "vote 3": (0, -11, "center"),
              "vote 5": (0, 6, "center"), "vote 7": (4, 6, "left")},
       384: {"1 frame": (0, -11, "center"), "until 2 agree": (-5, -9, "right"), "vote 3": (0, -11, "center"),
             "vote 5": (0, 6, "center"), "vote 7": (4, -11, "left")}}
for cap, (col, mk, lab) in SERIES.items():
    pts = [r for r in table if r["cap"] == cap]
    for r in pts:   # p90 whisker, thin and light, with an end cap at p90
        ax.plot([r["median_s"], r["p90_s"]], [r["exact"]] * 2, color=col, lw=1, alpha=0.45, solid_capstyle="butt", zorder=1)
        ax.plot([r["p90_s"]] * 2, [r["exact"] - 0.35, r["exact"] + 0.35], color=col, lw=1, alpha=0.6, zorder=1)
    ax.scatter([r["median_s"] for r in pts], [r["exact"] for r in pts], s=28, marker=mk, color=col, label=lab,
               edgecolors="white", linewidths=1.0, zorder=3)
    for r in pts:
        if cap == 384 and r["policy"].startswith("vote"):
            continue                                   # fixed-vote points labelled once (1024 series) to cut clutter
        dx, dy, ha = OFF[cap][r["policy"]]
        ax.annotate(r["policy"], (r["median_s"], r["exact"]), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va="center", fontsize=6, color=INK2)
ax.set_xscale("log")
ax.set_xticks([2, 5, 10, 20, 50])
ax.set_xticklabels(["2", "5", "10", "20", "50"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlim(1.8, 90)
ax.set_xlabel("Modelled time per product on Quest 3S (s, log scale)")
ax.set_ylabel("Exact date accuracy (%)")
ax.grid(True, color=GRID, lw=0.6, which="major")
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
lo = min(r["exact"] for r in table)
ax.set_ylim(math.floor(lo - 2), math.ceil(max(r["exact"] for r in table) + 2))
p1 = next(r for r in table if r["cap"] == 1024 and r["policy"] == "1 frame")
p2 = next(r for r in table if r["cap"] == 1024 and r["policy"] == "until 2 agree")
ax.annotate("", xy=(p2["median_s"] * 0.93, p2["exact"] - 0.25), xytext=(p1["median_s"] * 1.08, p1["exact"] + 0.25),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=7), zorder=4)
ax.text(4.7, 61.3, f"{p2['median_s'] / p1['median_s']:.1f}$\\times$ time,\n+{p2['exact'] - p1['exact']:.1f} points",
        fontsize=6.5, color=INK, ha="left", va="center")
ax.legend(loc="lower right", frameon=False, fontsize=6.5, handletextpad=0.3)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "latency_accuracy.pdf")
fig.savefig(OUT / "latency_accuracy.png", dpi=220)
print("saved", OUT / "latency_accuracy.pdf")
