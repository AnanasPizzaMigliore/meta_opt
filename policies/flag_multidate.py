"""Pre-flag held-out products whose packaging seems to carry more than one printed date.

For every product (p004-p040), every clip (a, b, d and the no-date side c) and every OCR run (PP-OCRv5 mobile on all
sampled frames; PP-OCRv6 tiny/small/medium on the first 7), parse all date candidates from the recognised lines with
the paper's date rules (ocr_baseline.parse_dates). A product is flagged when a date that does not match its ground
truth is parsed on at least 2 frames. For each flagged product, save a crop showing that other date and, if any, a
crop showing the true date, and build one contact sheet for the author to check.

  llm-env python flag_multidate.py [out dir]
"""
import collections
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import real_eval as re_

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("multidate_check")
P2 = Path("/home/user/qwen/phone_bench/questcap_captures/pilot2")
RUNS = [("v5", P2 / "loc")] + [(f"v6{t}", P2 / "loc_v6" / t) for t in ("tiny", "small", "medium")]
FONT = ImageFont.truetype(str(next(Path.home().glob(".TinyTeX/texmf-dist/fonts/opentype/public/libertine/LinBiolinum_R.otf"))), 22)


def tri(d):
    y, m, dd = d
    return (re_.ob.to_year(y) if y is not None else None, m, dd)


def fmt(t):
    return "-".join("??" if v is None else f"{v:02d}" if i else str(v) for i, v in enumerate(t))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    products = [p for p in sorted(re_.TRUTH) if p >= "p004"]
    report, tiles = {}, []
    for p in products:
        truth = tuple(re_.TRUTH[p])
        other = collections.defaultdict(list)      # date -> [(run, clip, frame, line)]
        true_hits = []
        for run, L in RUNS:
            for c in "abdc":
                mp = L / f"{p}_{c}_rot0" / "meta.json"
                if not mp.exists():
                    continue
                m = json.load(open(mp))
                for f, x in m.items():
                    for line in x.get("all_lines", []):
                        for d in re_.ob.parse_dates(line, compact=False):
                            t = tri(d)
                            if t[0] is None or t[1] is None or t[2] is None or not 2024 <= t[0] <= 2030:
                                continue
                            if re_.match(t, truth):
                                true_hits.append((run, c, f, line))
                            else:
                                other[t].append((run, c, f, line))
        frames = {t: {(r, c, f) for r, c, f, _ in v} for t, v in other.items()}
        multi = {t: v for t, v in other.items() if len(frames[t]) >= 2}
        if not multi:
            continue
        report[p] = {"truth": fmt(truth), "other_dates": {fmt(t): {"frames": len(frames[t]),
                     "example_lines": sorted({ln for *_, ln in v})[:4]} for t, v in sorted(multi.items(), key=lambda kv: -len(frames[kv[0]]))}}
        best_t = max(multi, key=lambda t: len(frames[t]))
        r, c, f, line = multi[best_t][0]
        L = dict(RUNS)[r]
        crops = [(L / f"{p}_{c}_rot0" / "x1" / f, f"{p} clip {c}: OCR '{line[:28]}' -> {fmt(best_t)}")]
        if true_hits:
            r2, c2, f2, l2 = true_hits[0]
            crops.append((dict(RUNS)[r2] / f"{p}_{c2}_rot0" / "x1" / f2, f"{p} clip {c2}: '{l2[:28]}' = truth {fmt(truth)}"))
        for path, label in crops:
            if path.exists():
                im = Image.open(path).convert("RGB")
                im.save(OUT / f"{p}_{path.parent.parent.name}_{path.name}")
                tiles.append((im, label))
    json.dump(report, open(OUT / "flagged.json", "w"), indent=1, ensure_ascii=False)
    # contact sheet: tiles scaled to a common height for viewing only (originals saved alongside at full size)
    H, W = 260, 520
    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * W, rows * (H + 40)), "white")
    dr = ImageDraw.Draw(sheet)
    for k, (im, label) in enumerate(tiles):
        s = min(W / im.width, H / im.height)
        t = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
        x, y = (k % cols) * W, (k // cols) * (H + 40)
        sheet.paste(t, (x, y))
        dr.text((x + 4, y + H + 6), label, fill="black", font=FONT)
    sheet.save(OUT / "contact_sheet.jpg", quality=92)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"{len(report)} of {len(products)} products flagged; sheet: {OUT / 'contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
