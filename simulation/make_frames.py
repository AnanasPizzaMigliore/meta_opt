"""Simulated video frames of each test product for the multi-frame voting study.

Frame k of an image = the 12 MP photo seen from a slightly different pose and under different capture
conditions, then sampled at headset resolution (product ~960 px across), the way consecutive passthrough
frames differ while a user holds a product:
  rotation +-3 deg, distance +-10 % (scale), brightness/contrast +-8 %, defocus/motion blur sigma 0-0.8 px,
  sensor noise std 2-4 grey levels, JPEG quality 85-92.
Each frame is then localised independently with PP-OCRv5 (same rule as make_text_crops.py) and the
date-text crop is zoomed 2x:  frames/f<k>/<file>  +  frames/f<k>_meta.json

  python make_frames.py <k>          (one process per frame index)

Optimistic like the rest of quest_sim: perturbations of one sharp photo are more alike than real frames.
"""
import json
import re
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from paddleocr import PaddleOCR

K = int(sys.argv[1])
DET = sys.argv[2] if len(sys.argv) > 2 else "PP-OCRv5_server_det"     # PP-OCRv5_mobile_det = runs on the Quest
TAG = "" if DET == "PP-OCRv5_server_det" else "_m"
N = int(sys.argv[3]) if len(sys.argv) > 3 else None
HERE = Path(__file__).parent
SRC = Path("/home/user/Dataset/test")
FILES = [e["filename"] for e in json.load(open("/home/user/Dataset/test.json"))][:N]
S = 960

MONTH = r"(jan|ene|feb|fev|mar|abr|apr|avr|may|mai|jun|jui|jul|ago|aug|aou|sep|set|oct|out|okt|nov|dic|dec|dez)"
DATE_PATTERNS = [
    (3, re.compile(r"\b\d{1,2}\s*[./\-]\s*\d{1,2}\s*[./\-]\s*\d{2,4}\b")),
    (3, re.compile(rf"\d{{1,2}}\s*{MONTH}\w*\.?\s*\d{{2,4}}|{MONTH}\w*\.?\s*\d{{2,4}}", re.I)),
    (2, re.compile(r"\b\d{1,2}\s*[./\-]\s*20[2-4]\d\b")),
    (1, re.compile(r"\b\d{1,2}\s*[/\-]\s*\d{2}\b")),
    (1, re.compile(r"\b20[2-4]\d\b")),
]
KEY_RE = re.compile(r"cad|consum|exp|best|use\s*by|\bbb|\bbbe|venc|val|prefer|mhd|dlc|dluo|ddm|fecha|date", re.I)
date_score = lambda t: max((sc for sc, rx in DATE_PATTERNS if rx.search(t)), default=0)
center = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
union = lambda bs: [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def frame_of(fn):
    rng = np.random.default_rng(zlib.crc32(f"{fn}:{K}".encode()))  # stable across processes
    img = Image.open(SRC / fn).convert("RGB")
    img = img.rotate(rng.uniform(-3, 3), resample=Image.BICUBIC, expand=False, fillcolor=(128, 128, 128))
    s = S * rng.uniform(0.9, 1.1)
    k = s / max(img.size)
    img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.92, 1.08))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.92, 1.08))
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 0.8)))
    a = np.asarray(img, dtype=np.float32) + rng.normal(0, rng.uniform(2, 4), (img.height, img.width, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), int(rng.integers(85, 93))


def locate(lines, W, H):
    scored = [(date_score(t), t, b) for t, b in lines]
    level = "date"
    if not any(sc for sc, _, _ in scored):
        scored = [(1 if len(re.findall(r"\d", t)) >= 4 else 0, t, b) for t, b in lines]
        level = "digits"
    keylines = [(t, b) for t, b in lines if KEY_RE.search(t)]
    near = lambda b, ref, reach: abs(center(b)[0] - center(ref)[0]) <= reach and abs(center(b)[1] - center(ref)[1]) <= reach
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
    for a in (0, 1):
        if box[a + 2] - box[a] < 96:
            c = (box[a] + box[a + 2]) / 2
            box[a], box[a + 2] = c - 48, c + 48
    return [max(0, int(box[0])), max(0, int(box[1])), min(W, int(box[2])), min(H, int(box[3]))], level, [t for t, _ in chosen]


ocr = PaddleOCR(text_detection_model_name=DET, text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
out = HERE / f"frames{TAG}" / f"f{K}"
out.mkdir(parents=True, exist_ok=True)
raw = HERE / "frames_raw" / f"f{K}"          # the uncropped frame, so other localisation rules can be replayed
raw.mkdir(parents=True, exist_ok=True)
meta = {}
tmp = Path(tempfile.mkdtemp()) / "frame.jpg"
for i, fn in enumerate(FILES):
    img, q = frame_of(fn)
    img.save(tmp, quality=q)
    img.save(raw / fn, quality=q)
    img = Image.open(tmp).convert("RGB")
    r = ocr.predict(str(tmp))[0]
    lines = [(t, [int(v) for v in b]) for t, b in zip(r["rec_texts"], r["rec_boxes"])]
    box, level, chosen = locate(lines, *img.size)
    crop = img.crop(box)
    crop.resize((crop.width * 2, crop.height * 2), Image.BICUBIC).save(out / fn, quality=95)
    meta[fn] = {"box": box, "level": level, "lines": chosen, "frame_wh": list(img.size),
                "all_lines": [[t, b, float(sc)] for (t, b), sc in zip(lines, r["rec_scores"])]}
    if i % 50 == 0:
        print(f"f{K} {i}/{len(FILES)} {fn} {level} {crop.size} {chosen[:2]}", flush=True)
json.dump(meta, open(HERE / f"frames{TAG}" / f"f{K}_meta.json", "w"), indent=1, ensure_ascii=False)
print(f"f{K} done")
