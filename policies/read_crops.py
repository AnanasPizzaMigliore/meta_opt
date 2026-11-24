"""Read every image of a folder with the fine-tuned VLM (host GPU; same server, prompt and decoding as eval_gguf.py).

  llm-env python read_crops.py <image dir> <out json> [--cap 1024] [--port 8795]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/qwen")
import eval_gguf as E

G = "/home/user/qwen/gguf/"
ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("out")
ap.add_argument("--cap", type=int, default=1024)
ap.add_argument("--port", type=int, default=8795)
a = ap.parse_args()
E.EXTRA_SERVER_ARGS.extend(["--image-max-tokens", str(a.cap)])
proc, base = E.start_server(G + "qwen35_groupA_a3_Q4_0_tv.gguf", G + "qwen35_groupA_a3_mmproj_Q8_0.gguf", "vulkan", a.port)
res = {}
try:
    for fp in sorted(Path(a.folder).glob("*.jpg")):
        try:
            content, lat, ntok = E.query_server(base, str(fp))
            try:
                js = E.extract_json(content)
            except Exception:
                js = None
            res[fp.name] = {"content": content, "json": js, "latency_s": round(lat, 2), "gen_tokens": ntok}
        except Exception as e:
            res[fp.name] = {"error": str(e)}
        print(fp.name, json.dumps(res[fp.name], ensure_ascii=False)[:200], flush=True)
finally:
    E.stop_server(proc)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
