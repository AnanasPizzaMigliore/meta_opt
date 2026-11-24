"""Contact sheet + summary of one processed clip:  summarise_clip.py <clip dir> <loc dir>"""
import collections
import glob
import json
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageDraw

clip, loc = sys.argv[1], Path(sys.argv[2])
fs = sorted(glob.glob(f"{clip}/f*.jpg"))
pick = fs[::max(1, len(fs) // 12)][:12]
T = 432
sheet = Image.new("RGB", (6 * T, 2 * T))
for i, f in enumerate(pick):
    im = Image.open(f).resize((T, T))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 90, 22], fill=(0, 0, 0))
    d.text((4, 4), Path(f).stem, fill=(255, 255, 0))
    sheet.paste(im, ((i % 6) * T, (i // 6) * T))
sheet.save(str(loc) + "_sheet.jpg", quality=85)

m = json.load(open(loc / "meta.json"))
lev = collections.Counter(v["level"] for v in m.values())
print(f"{loc.name}: {len(fs)} frames, {len(m)} sampled; localisation {dict(lev)}; "
      f"median sharpness {statistics.median(v['sharpness'] for v in m.values()):.0f}")
for cap in (1024, 384):
    fp = loc / f"reads_cap{cap}.json"
    if not fp.exists():
        continue
    r = json.load(open(fp))
    c = collections.Counter()
    for k, v in r.items():
        j = v.get("json") or {}
        c[f"{j.get('year')}-{j.get('month')}-{j.get('day')}" if j.get("year") or j.get("month") else "no date"] += 1
    print(f"   cap {cap}: {dict(c.most_common())}")
print("   OCR date lines:", collections.Counter(t for v in m.values() if v["level"] == "date" for t in v["lines"]).most_common(4))
