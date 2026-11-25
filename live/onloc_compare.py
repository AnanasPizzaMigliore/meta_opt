"""Compare the on-device localisation port (onloc_dump output) with the offline PaddleOCR pipeline (meta.json and
live_plan.txt): recognised lines, gate level, OCR date and crop box, per frame.
  CLIPS=<clip dir> python3 live/onloc_compare.py <dump.tsv> [...]
"""
import collections, json, os, sys
from pathlib import Path
L = Path(os.environ.get("CLIPS", Path(__file__).resolve().parent.parent / "data" / "real" / "clips"))  # per-clip meta.json
plan = {}
for l in open(Path(__file__).parent / "live_plan.txt"):
    t = l.split()
    if t[0] == "F": plan[t[1]] = (t[2], t[3])
metas = {}
def meta(name):
    clip, f = name.rsplit("_", 1)
    if clip not in metas: metas[clip] = json.load(open(L / clip / "meta.json"))
    return metas[clip][f]
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - ix*iy
    return ix*iy/u if u else 1.0
c = collections.Counter(); ious = []; diffs = []; td = []; tr = []
for p in sys.argv[1:]:
    for l in open(p, encoding="utf-8"):
        f = l.rstrip("\n").split("\t")
        name, level, date, box, n = f[0], f[1], f[2], [int(v) for v in f[3].split(",")], int(f[4])
        td.append(float(f[5])); tr.append(float(f[6]))
        lines = [x.split("|")[0] for x in f[8].split("\x1f")] if len(f) > 8 and f[8] else []
        m = meta(name); plevel, pdate = plan[name]
        c["frames"] += 1
        c["same lines"] += lines == m["all_lines"]
        c["same line count"] += len(lines) == len(m["all_lines"])
        c["same line set"] += sorted(lines) == sorted(m["all_lines"])
        c["same level"] += level == plevel
        c["same date"] += date == pdate
        c["both have date"] += date != "NONE" and pdate != "NONE"
        c["date offline only"] += date == "NONE" and pdate != "NONE"
        c["date port only"] += date != "NONE" and pdate == "NONE"
        c["dates differ"] += date != "NONE" and pdate != "NONE" and date != pdate
        ious.append(iou(box, m["box"]))
        if date != pdate: diffs.append((name, pdate, date))
for k, v in c.items(): print(f"{k:20s} {v}  ({100*v/c['frames']:.1f}%)")
import statistics as st
print("crop IoU median %.3f, >=0.9: %.1f%%, >=0.5: %.1f%%" % (st.median(ious), 100*sum(i>=.9 for i in ious)/len(ious), 100*sum(i>=.5 for i in ious)/len(ious)))
print("ms det median %.0f rec median %.0f" % (st.median(td), st.median(tr)))
for d in diffs[:15]: print("  date diff", d)
