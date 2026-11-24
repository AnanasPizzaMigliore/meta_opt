"""Record PP-OCRv5 per-line recognition confidence for the real frames (held-out replay frames and the pilot's first
7 frames per clip), with the same models and settings as loc_clip.py, and check that the recognised text equals the
stored meta.json all_lines.   ocr-env python ocr_scores.py <out json>
"""
import json
import sys
from pathlib import Path

from paddleocr import PaddleOCR

LOC = Path("/home/user/qwen/phone_bench/questcap_captures/pilot2/loc")
names = Path("/home/user/qwen/phone_bench/dvfs/revision/replay_images.txt").read_text().split()
pilot = []
for p in ("p001", "p002", "p003"):
    for c in "abdc":
        d = LOC / f"{p}_{c}_rot0"
        if (d / "meta.json").exists():
            pilot += [f"{d.name}/x2/{f}" for f in sorted(json.load(open(d / "meta.json")))[:7]]
ocr = PaddleOCR(text_detection_model_name="PP-OCRv5_mobile_det", text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
out, same = {}, 0
for k, x2 in enumerate(pilot + names):
    clip, _, f = x2.split("/")
    r = ocr.predict(str(LOC / clip / "frames" / f))[0]
    texts, scores = list(r["rec_texts"]), [float(v) for v in r["rec_scores"]]
    meta = json.load(open(LOC / clip / "meta.json"))[f]
    same += texts == meta["all_lines"]
    out[f"{clip}/{f}"] = {"texts": texts, "scores": scores}
    if k % 200 == 0:
        print(k, len(pilot) + len(names), x2, f"identical so far {same}/{k + 1}", flush=True)
json.dump(out, open(sys.argv[1], "w"))
print(f"done: {len(out)} frames ({len(pilot)} pilot); text identical to stored run on {same}")
