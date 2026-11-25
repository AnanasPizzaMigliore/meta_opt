"""Build the live-loop plan for the 148 held-out clips: every sampled frame of each clip in time order, with the
detector's gate level, the date the OCR rules parse from the frame's recognised lines (same function as the paper's
analysis), and the frame's line boxes (from the same PP-OCRv5 pipeline on the server). Also stages frames and 2x
crops under device file names and writes the request template (same prompt and decoding as the deployed config).

  python live/make_live_plan.py [clip dir]
"""
import json
import shutil
import sys
from pathlib import Path


R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "policies"))
import real_eval as re_  # noqa: E402

H = Path(__file__).resolve().parent
L = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "data" / "real" / "clips"  # per-clip meta.json, frames/, x2/
PROMPT = (R / "bench" / "prompt.txt").read_text()


def boxes_from(listfile):
    out, cur = {}, None
    for line in open(listfile):
        t = line.split()
        if not t:
            continue
        if t[0] == "F":
            cur = t[1]
            out[cur] = []
        else:
            out[cur].append(t[1:5])
    return out


def fmt(t):
    if t == re_.rp.NONE:
        return "NONE"
    y, m, d = t
    return f"{y:04d}-{m:02d}-{d:02d}".replace("-00", "-??") if False else \
        f"{'????' if y is None else f'{y:04d}'}-{'??' if m is None else f'{m:02d}'}-{'??' if d is None else f'{d:02d}'}"


def main():
    bx = boxes_from(H / "replay_lines.txt")
    bx.update(boxes_from(H / "live_lines_new.txt"))
    fd, cd = H / "live_frames", H / "live_crops"
    fd.mkdir(exist_ok=True)
    cd.mkdir(exist_ok=True)
    lines, n = [], 0
    for p in sorted(re_.TRUTH):
        if p < "p004":
            continue
        for c in "abdc":
            clip = f"{p}_{c}_rot0"
            m = json.load(open(L / clip / "meta.json"))
            names = sorted(m)
            lines.append(f"C {clip} {len(names)}")
            for f in names:
                dev = f"{clip}_{f}"
                ocr = fmt(re_.ocr_date(m[f]))
                b = bx[dev]
                lines.append(f"F {dev} {m[f]['level']} {ocr} {len(b)}")
                lines += ["L " + " ".join(v) for v in b]
                if not (fd / dev).exists():
                    shutil.copy(L / clip / "frames" / f, fd / dev)
                if not (cd / dev).exists():
                    shutil.copy(L / clip / "x2" / f, cd / dev)
                n += 1
    (H / "live_plan.txt").write_text("\n".join(lines) + "\n")
    for name, stop in (("opt", True), ("stock", False)):
        payload = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,@@B64@@"}},
            {"type": "text", "text": PROMPT}]}],
            "max_tokens": 128, "temperature": 0, "stream": False, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}}
        if stop:
            payload["stop"] = ["\n"]
        s = json.dumps(payload)
        pre, suf = s.split("@@B64@@")
        (H / f"live_prefix_{name}.json").write_text(pre)
        (H / f"live_suffix_{name}.json").write_text(suf)
    print(f"{n} frames in {len(lines)} plan lines")


if __name__ == "__main__":
    main()
