"""Localise the date text in frames of a real Quest 3S passthrough clip (same logic as
../make_text_crops_mobile.py: PP-OCRv5 mobile detector + latin recogniser, crop around date-like lines,
2x bicubic zoom; whole frame if no date-like line is found).

  ocr-env python loc_clip.py <clip dir with fNNN.jpg> <out dir> [--every 6] [--rot 0|90|180|270]

--rot rotates every frame first (the wearer may hold the product upside down).
Writes <out>/x2/<frame>.jpg, <out>/x1/<frame>.jpg and <out>/meta.json.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

MONTH = r"(jan|ene|feb|fev|mar|abr|apr|avr|may|mai|jun|jui|jul|ago|aug|aou|sep|set|oct|out|okt|nov|dic|dec|dez)"
DATE_PATTERNS = [
    (3, re.compile(r"\b\d{1,2}\s*[./\-]\s*\d{1,2}\s*[./\-]\s*\d{2,4}\b")),
    (3, re.compile(rf"\d{{1,2}}\s*{MONTH}\w*\.?\s*\d{{2,4}}|{MONTH}\w*\.?\s*\d{{2,4}}", re.I)),
    (2, re.compile(r"\b\d{1,2}\s*[./\-]\s*20[2-4]\d\b")),
    (1, re.compile(r"\b\d{1,2}\s*[/\-]\s*\d{2}\b")),
    (1, re.compile(r"\b20[2-4]\d\b")),
]
KEY_RE = re.compile(r"cad|consum|exp|best|use\s*by|\bbb|\bbbe|venc|val|prefer|mhd|dlc|dluo|ddm|fecha|date", re.I)


def date_score(text):
    return max((sc for sc, rx in DATE_PATTERNS if rx.search(text)), default=0)


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def locate(lines, W, H):
    """lines: [(text, [x0, y0, x1, y1])] -> (crop box, level, chosen lines)."""
    scored = [(date_score(t), t, b) for t, b in lines]
    level = "date"
    if not any(sc for sc, _, _ in scored):
        scored = [(1 if len(re.findall(r"\d", t)) >= 4 else 0, t, b) for t, b in lines]
        level = "digits"
    keylines = [(t, b) for t, b in lines if KEY_RE.search(t)]

    def near(b, ref, reach):
        return abs(center(b)[0] - center(ref)[0]) <= reach and abs(center(b)[1] - center(ref)[1]) <= reach

    if not any(sc for sc, _, _ in scored):
        return [0, 0, W, H], "fallback_whole", []

    def total(e):
        sc, t, b = e
        reach = 2.0 * max(b[2] - b[0], b[3] - b[1], 32)
        return sc + (2 if any(near(kb, b, reach) for _, kb in keylines) else 0) if sc else -1
    best = max(scored, key=total)
    reach = 2.0 * max(best[2][2] - best[2][0], best[2][3] - best[2][1], 32)
    chosen = [(t, b) for sc, t, b in scored if sc and near(b, best[2], reach)]
    chosen += [(t, b) for t, b in keylines if near(b, best[2], reach)]
    box = union([b for _, b in chosen])
    pad = 0.25 * min(box[2] - box[0], box[3] - box[1]) + 16
    box = [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad]
    for a, lim in ((0, W), (1, H)):
        if box[a + 2] - box[a] < 96:
            c = (box[a] + box[a + 2]) / 2
            box[a], box[a + 2] = c - 48, c + 48
    box = [max(0, int(box[0])), max(0, int(box[1])), min(W, int(box[2])), min(H, int(box[3]))]
    return box, level, chosen


def sharpness(img):
    g = np.asarray(img.convert("L"), dtype=np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--rot", type=int, default=0, choices=[0, 90, 180, 270])
    a = ap.parse_args()
    out = Path(a.out)
    (out / "x1").mkdir(parents=True, exist_ok=True)
    (out / "x2").mkdir(exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    ocr = PaddleOCR(text_detection_model_name="PP-OCRv5_mobile_det", text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
                    use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
    meta = {}
    for fp in sorted(Path(a.clip).glob("f*.jpg"))[::a.every]:
        img = Image.open(fp).convert("RGB")
        if a.rot:
            img = img.rotate(a.rot, expand=True)
        W, H = img.size
        src = out / "frames" / fp.name
        img.save(src, quality=95)
        t0 = time.time()
        r = ocr.predict(str(src))[0]
        dt = time.time() - t0
        lines = [(t, [int(v) for v in b]) for t, b in zip(r["rec_texts"], r["rec_boxes"])]
        box, level, chosen = locate(lines, W, H)
        crop = img.crop(box)
        crop.save(out / "x1" / fp.name, quality=95)
        crop.resize((crop.width * 2, crop.height * 2), Image.BICUBIC).save(out / "x2" / fp.name, quality=95)
        meta[fp.name] = {"n_lines": len(lines), "box": box, "level": level, "lines": [t for t, _ in chosen],
                         "all_lines": [t for t, _ in lines], "crop_wh": [crop.width, crop.height],
                         "sharpness": round(sharpness(img), 1), "ocr_s": round(dt, 2)}
        print(fp.name, level, crop.size, [t for t, _ in chosen][:4], f"sharp={meta[fp.name]['sharpness']}", flush=True)
    json.dump(meta, open(out / "meta.json", "w"), indent=1, ensure_ascii=False)
