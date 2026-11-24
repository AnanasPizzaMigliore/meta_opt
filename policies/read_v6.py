"""Read the PP-OCRv6 crops of every replayed clip with the VLM (host GPU, same server, prompt and decoding as
read_crops.py), one server for all clips.   llm-env python read_v6.py <tier> [--cap 1024]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/qwen")
import eval_gguf as E

G = "/home/user/qwen/gguf/"
ROOT = Path("/home/user/qwen/phone_bench/questcap_captures/pilot2/loc_v6")
ap = argparse.ArgumentParser()
ap.add_argument("tier")
ap.add_argument("--cap", type=int, default=1024)
ap.add_argument("--port", type=int, default=8796)
a = ap.parse_args()
E.EXTRA_SERVER_ARGS.extend(["--image-max-tokens", str(a.cap)])
proc, base = E.start_server(G + "qwen35_groupA_a3_Q4_0_tv.gguf", G + "qwen35_groupA_a3_mmproj_Q8_0.gguf", "vulkan", a.port)
try:
    for clip in sorted((ROOT / a.tier).iterdir()):
        out = clip / f"reads_cap{a.cap}.json"
        if out.exists():
            continue
        res = {}
        for fp in sorted((clip / "x2").glob("*.jpg")):
            try:
                content, lat, ntok = E.query_server(base, str(fp))
                try:
                    js = E.extract_json(content)
                except Exception:
                    js = None
                res[fp.name] = {"content": content, "json": js, "latency_s": round(lat, 2), "gen_tokens": ntok}
            except Exception as e:
                res[fp.name] = {"error": str(e)}
        json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
        print(a.tier, clip.name, len(res), flush=True)
finally:
    E.stop_server(proc)
