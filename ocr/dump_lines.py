"""Host side of the localisation replay: rerun the PaddleOCR pipeline used for the real-video crops (PP-OCRv5 mobile
detector + latin recogniser, same settings as loc_clip.py) on every replayed frame and write, per frame, the image
path and the line boxes, so that the headset can time the detector on the real frame and the recogniser on the real
line crops (ocrreplay.cpp).

  ocr-env python dump_lines.py <replay_images.txt> <out list> <out json>
List format:  F <frame jpg on headset> <W> <H>  then one  L x0 y0 x1 y1  per recognised line.
"""
import json
import sys
from pathlib import Path

from paddleocr import PaddleOCR

LOC = Path("/home/user/qwen/phone_bench/questcap_captures/pilot2/loc")
imgs, out_list, out_json = sys.argv[1:4]
ocr = PaddleOCR(text_detection_model_name="PP-OCRv5_mobile_det", text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
rows, info, same = [], {}, 0
names = Path(imgs).read_text().split()
for k, x2 in enumerate(names):
    clip, _, f = x2.split("/")
    src = LOC / clip / "frames" / f
    r = ocr.predict(str(src))[0]
    boxes = [[int(v) for v in b] for b in r["rec_boxes"]]
    texts = list(r["rec_texts"])
    meta = json.load(open(LOC / clip / "meta.json"))[f]
    same += texts == meta["all_lines"]
    H, W = r["doc_preprocessor_res"]["output_img"].shape[:2] if "doc_preprocessor_res" in r else (1728, 1728)
    rows.append(f"F {clip}_{f} {W} {H}")
    rows += [f"L {b[0]} {b[1]} {b[2]} {b[3]}" for b in boxes]
    info[x2] = dict(n=len(boxes), same_as_meta=texts == meta["all_lines"])
    if k % 100 == 0:
        print(k, x2, len(boxes), f"same so far {same}/{k + 1}", flush=True)
Path(out_list).write_text("\n".join(rows) + "\n")
json.dump(info, open(out_json, "w"))
print(f"done {len(names)} frames; recognised lines identical to the stored run on {same}")
