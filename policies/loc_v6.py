"""Sensitivity check with PP-OCRv6 (PaddleOCR 3.7, released 2026-06-11): localise the replayed real frames (the first
7 sampled frames of the 148 held-out clips) with a PP-OCRv6 tier, same locate() rules, crops and 2x zoom as
loc_clip.py, so that the VLM reads and the stopping rules can be recomputed.

  ocr6-env python loc_v6.py <tier: tiny|small|medium> [out root]
Writes <out root>/<tier>/<clip>_rot0/{meta.json,x1/,x2/}.
"""
import json
import sys
import time
from pathlib import Path

from PIL import Image
from paddleocr import PaddleOCR

sys.path.insert(0, str(Path(__file__).parent))
from loc_clip import locate, sharpness  # noqa: E402

tier = sys.argv[1]
ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/user/qwen/phone_bench/questcap_captures/pilot2/loc_v6")
LOC = Path("/home/user/qwen/phone_bench/questcap_captures/pilot2/loc")
names = Path("/home/user/qwen/phone_bench/dvfs/revision/replay_images.txt").read_text().split()
ocr = PaddleOCR(text_detection_model_name=f"PP-OCRv6_{tier}_det", text_recognition_model_name=f"PP-OCRv6_{tier}_rec",
                use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
metas = {}
t_start = time.time()
for k, x2 in enumerate(names):
    clip, _, f = x2.split("/")
    out = ROOT / tier / clip
    (out / "x1").mkdir(parents=True, exist_ok=True)
    (out / "x2").mkdir(exist_ok=True)
    src = LOC / clip / "frames" / f                       # the same frames the v5 run used
    img = Image.open(src).convert("RGB")
    W, H = img.size
    t0 = time.time()
    r = ocr.predict(str(src))[0]
    dt = time.time() - t0
    lines = [(t, [int(v) for v in b]) for t, b in zip(r["rec_texts"], r["rec_boxes"])]
    box, level, chosen = locate(lines, W, H)
    crop = img.crop(box)
    crop.save(out / "x1" / f, quality=95)
    crop.resize((crop.width * 2, crop.height * 2), Image.BICUBIC).save(out / "x2" / f, quality=95)
    metas.setdefault(clip, {})[f] = {"n_lines": len(lines), "box": box, "level": level,
                                    "lines": [t for t, _ in chosen], "all_lines": [t for t, _ in lines],
                                    "crop_wh": [crop.width, crop.height], "sharpness": round(sharpness(img), 1),
                                    "ocr_s": round(dt, 2)}
    if k % 100 == 0:
        print(f"{tier} {k}/{len(names)} {time.time() - t_start:.0f}s {clip}/{f} {level} lines={len(lines)}", flush=True)
for clip, m in metas.items():
    json.dump(m, open(ROOT / tier / clip / "meta.json", "w"), indent=1, ensure_ascii=False)
print(f"done {tier}: {len(names)} frames in {time.time() - t_start:.0f}s")
