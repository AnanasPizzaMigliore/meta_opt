"""Cheaper on-device localisation: recognise fewer text lines.

The full rule (make_frames.py) recognises every detected line and keeps the date-shaped ones. On the Quest
the recogniser costs ~0.07 s per line and the median frame has ~19 lines. Here each frame's saved lines
(text, box, score from frames_m/f<k>_meta.json) are replayed under rules that recognise only a subset,
selected from the detection geometry alone (available before recognition):

  all            every line (reference)
  geom           height 12..120 px and width/height 1.5..20 (date-like line shapes)
  geom+topK      geom, then the K lines with the largest height (K = 3, 5, 8)
  topK           the K tallest lines, no shape filter

For each rule: lines recognised (median / p90), estimated recogniser time on the Quest, and how often the
crop box matches the full rule's (IoU >= 0.7) -- a matching box gives the same read. Crops for the chosen
rule are written to frames_m_<rule>/f<k>/ so they can be read and scored.

  python loc_policies.py [k] [write_rule]
"""
import json
import statistics as st
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import importlib.util

HERE = Path(__file__).parent
K = int(sys.argv[1]) if len(sys.argv) > 1 else 0
WRITE = sys.argv[2] if len(sys.argv) > 2 else None

# reuse the exact localisation rule from make_frames.py without running its OCR set-up
import re
src = (HERE / "make_frames.py").read_text()
ns = {"re": re}
exec(src[src.index("MONTH ="):src.index("def frame_of")] + src[src.index("def locate"):src.index("ocr = PaddleOCR(")], ns)
locate = ns["locate"]

REC_MS = lambda w, h: 7 + 0.1875 * (48 * w / max(h, 1))    # Quest: 48x160 -> 37 ms, 48x320 -> 67 ms


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u else 0.0


def geom(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return 12 <= h <= 120 and 1.5 <= w / max(h, 1) <= 20


RULES = {
    "all": lambda L: L,
    "geom": lambda L: [l for l in L if geom(l[1])],
    "geom+top3": lambda L: sorted([l for l in L if geom(l[1])], key=lambda l: -(l[1][3] - l[1][1]))[:3],
    "geom+top5": lambda L: sorted([l for l in L if geom(l[1])], key=lambda l: -(l[1][3] - l[1][1]))[:5],
    "geom+top8": lambda L: sorted([l for l in L if geom(l[1])], key=lambda l: -(l[1][3] - l[1][1]))[:8],
    "top5": lambda L: sorted(L, key=lambda l: -(l[1][3] - l[1][1]))[:5],
    "top8": lambda L: sorted(L, key=lambda l: -(l[1][3] - l[1][1]))[:8],
}

meta = json.load(open(HERE / "frames_m" / f"f{K}_meta.json"))
print(f"frame f{K}: {len(meta)} images")
print(f"{'rule':12s} {'lines med':>9s} {'p90':>5s} {'rec s med':>9s} {'p90':>6s} {'same box %':>10s} {'whole-frame %':>13s}")
for name, rule in RULES.items():
    n, cost, same, whole = [], [], 0, 0
    for fn, m in meta.items():
        L = [(t, b) for t, b, _ in m["all_lines"]]
        sub = rule(L)
        W, H = m["frame_wh"]
        box, level, _ = locate(sub, W, H)
        ref = m["box"]
        n.append(len(sub))
        cost.append(sum(REC_MS(b[2] - b[0], b[3] - b[1]) for _, b in sub) / 1e3)
        same += iou(box, ref) >= 0.7
        whole += level == "fallback_whole"
        if WRITE == name:
            out = HERE / f"frames_m_{name}" / f"f{K}"
            out.mkdir(parents=True, exist_ok=True)
            img = Image.open(HERE / "frames_raw" / f"f{K}" / fn).convert("RGB")
            c = img.crop(box)
            c.resize((c.width * 2, c.height * 2), Image.BICUBIC).save(out / fn, quality=95)
    n.sort(); cost.sort()
    p90 = lambda v: v[int(0.9 * (len(v) - 1))]
    print(f"{name:12s} {st.median(n):9.0f} {p90(n):5.0f} {st.median(cost):9.2f} {p90(cost):6.2f} "
          f"{100 * same / len(meta):10.1f} {100 * whole / len(meta):13.1f}")
