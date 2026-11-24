"""Headset replay, part 1: read the first 7 sampled frames of every held-out real clip (p004-p040, 148 clips) with
the deployed VLM on the Quest (config +stop_if, default clocks, as the 544-crop accuracy run), one read per frame.

  python replay_vlm.py revision/replay_vlm.jsonl        (resumes: frames already in the output are skipped)

read_s = server prompt + decode time (server timings; excludes the adb transport). Per-clip policy times are composed
later from these per-frame reads plus on-headset localisation (replay_time.py).
"""
import json
import sys
import time
from pathlib import Path

import e2e_bench as eb
import server_bench as sb

HERE = Path(__file__).parent
LOC = Path("/home/user/qwen/phone_bench/questcap_captures/pilot2/loc")
NAME = "+stop_if"


def main(out):
    imgs = (HERE / "revision/replay_images.txt").read_text().split()
    done = set()
    if Path(out).exists():
        done = {json.loads(l)["image"] for l in open(out)}
    todo = [i for i in imgs if i not in done]
    print(f"{len(done)} done, {len(todo)} to read", flush=True)
    if not todo:
        return
    model, env, extra, tf, stop, lv = eb.CONFIGS[NAME]
    eb.start(model, env, extra)
    sb.sh("am broadcast -a com.oculus.vrpowermanager.prox_close")
    eb.read("0003.jpg", tf, stop)                   # warm-up on a test-set crop outside the replay
    sb.CROPS = LOC
    t0 = time.time()
    for k, img in enumerate(todo):
        if k % 50 == 0:
            sb.sh("am broadcast -a com.oculus.vrpowermanager.prox_close")
        js = eb.read(img, tf, stop)
        t = js.get("timings", {})
        msg = js["choices"][0]["message"]["content"]
        row = dict(config=NAME, image=img, prompt_n=t.get("prompt_n"), cache_n=t.get("cache_n"),
                   prompt_ms=t.get("prompt_ms"), gen=t.get("predicted_n"), decode_ms=t.get("predicted_ms"),
                   read_s=(t.get("prompt_ms", 0) + t.get("predicted_ms", 0)) / 1e3, out=msg, date=eb.date_of(msg),
                   wall=time.time())
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        el = time.time() - t0
        print(f"[{k + 1}/{len(todo)} {el / 60:.0f} min, eta {el / (k + 1) * (len(todo) - k - 1) / 60:.0f} min] {img} "
              f"prompt {row['prompt_n']} read {row['read_s']:.2f}s {row['date']}", flush=True)
    sb.sh("pkill -f '[l]lama-server-opencl'")


if __name__ == "__main__":
    main(sys.argv[1])
