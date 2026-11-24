"""End-to-end read latency on the Quest with a persistent llama-server, adding one optimisation at a time.

  python e2e_bench.py out.jsonl <config>...      (configs below; see summarize: python e2e_bench.py --sum out.jsonl)

read_s = server prompt time (ViT + LLM prefill) + decode time, from the server's own timings
(excludes the adb transport, which a real app does not have).
"""
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

import server_bench as sb

BYPASS = "GGML_OPENCL_A7X_F32_LM_BYPASS=0"
SRV_BASE = "-ngl 99 -t 4 -c 4096 -np 1 --image-max-tokens 1024 --image-min-tokens 64 -cram 512"
# name: model, env, extra server args, text_first, stop_at_sentence, (cpuLevel, gpuLevel) or None
CONFIGS = {
    "orig":        (sb.BASE, "", "--no-mmproj-offload", False, False, None),
    "orig_max":    (sb.BASE, "", "--no-mmproj-offload", False, False, (7, 7)),   # baseline at matched (max) clocks
    "+gpu_vit":    (sb.BASE, BYPASS, "", False, False, None),
    "+task_vocab": (sb.TV, BYPASS, "", False, False, None),
    "+prefix":     (sb.TV, BYPASS, "-ctxcp 8 -cms 32", True, False, None),
    "+stop":       (sb.TV, BYPASS, "-ctxcp 8 -cms 32", True, True, None),
    "+stop_max":   (sb.TV, BYPASS, "-ctxcp 8 -cms 32", True, True, (7, 7)),
    "+stop_eco":   (sb.TV, BYPASS, "-ctxcp 8 -cms 32", True, True, (1, 4)),
    "+cap384_max": (sb.TV, BYPASS, "-ctxcp 8 -cms 32 --image-max-tokens 384", True, True, (7, 7)),
    "+cap384_eco": (sb.TV, BYPASS, "-ctxcp 8 -cms 32 --image-max-tokens 384", True, True, (1, 4)),
    # image first (as fine-tuned): no instruction-prefix reuse, but no false dates on no-date images
    "+stop_if":        (sb.TV, BYPASS, "", False, True, None),
    "+stop_if_max":    (sb.TV, BYPASS, "", False, True, (7, 7)),
    "+stop_if_eco":    (sb.TV, BYPASS, "", False, True, (1, 4)),
    "+cap384_if_max":  (sb.TV, BYPASS, "--image-max-tokens 384", False, True, (7, 7)),
    "+cap384_if_eco":  (sb.TV, BYPASS, "--image-max-tokens 384", False, True, (1, 4)),
    "cpuvit6":     (sb.TV, BYPASS, "--no-mmproj-offload -ctxcp 8 -cms 32", True, True, None),
}


def start(model, env, extra, threads_mask="3c", threads=4):
    sb.sh("pkill -f '[l]lama-server-opencl'; sleep 1")
    sb.sh(f"cd {sb.R} && ({env} nohup taskset {threads_mask} ./llama-server-opencl -m {model} "
          f"--mmproj qwen35_groupA_a3_mmproj_Q8_0.gguf {SRV_BASE.replace('-t 4', f'-t {threads}')} --port {sb.PORT} "
          f"{extra} > server.log 2>&1 &)")
    for _ in range(120):
        if '"ok"' in sb.sh(f"printf 'GET /health HTTP/1.0\\r\\n\\r\\n' | nc -q 5 127.0.0.1 {sb.PORT}"):
            return
        time.sleep(1)
    raise RuntimeError(sb.sh(f"tail -20 {sb.R}/server.log"))


def read(img, text_first, stop):
    import base64
    imgc = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
            base64.b64encode((sb.CROPS / img).read_bytes()).decode()}}
    txt = {"type": "text", "text": sb.PROMPT}
    payload = {"messages": [{"role": "user", "content": [txt, imgc] if text_first else [imgc, txt]}],
               "max_tokens": 128, "temperature": 0, "stream": False, "cache_prompt": True,
               "chat_template_kwargs": {"enable_thinking": False}}
    if stop:
        payload["stop"] = ["\n"]
    body = json.dumps(payload).encode()
    req = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n"
           f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body
    (sb.HERE / "req.http").write_bytes(req)
    for attempt in range(4):                       # Wi-Fi adb occasionally drops a push/shell; retry, never re-time
        sb.subprocess.run([sb.ADB, "push", str(sb.HERE / "req.http"), f"{sb.R}/req.http"], capture_output=True)
        raw = sb.sh(f"{sb.R}/httpc {sb.PORT} {sb.R}/req.http")
        if "{" in raw:
            return json.loads(raw[raw.index("{"):])
        print(f"  retry {attempt + 1} for {img}: empty reply", flush=True)
        time.sleep(5)
    raise RuntimeError(f"no reply for {img}")


def date_of(text):
    m = re.search(r"reads (\d{4}|\?{4})-(\d{2}|\?\?)-(\d{2}|\?\?)", text)
    return m.groups() if m else ("none",)


def main(out, names):
    for name in names:
        model, env, extra, tf, stop, lv = CONFIGS[name]
        if name == "cpuvit6":
            start(model, env, extra, "3f", 6)
        else:
            start(model, env, extra)
        if lv:
            sb.sh(f"setprop debug.oculus.cpuLevel {lv[0]}; setprop debug.oculus.gpuLevel {lv[1]}")
        read("0003.jpg", tf, stop)              # warm-up on an image outside the set (fills the instruction prefix cache)
        for img in sb.IMAGES:
            js = read(img, tf, stop)
            t = js.get("timings", {})
            msg = js["choices"][0]["message"]["content"]
            row = dict(config=name, image=img, prompt_n=t.get("prompt_n"), cache_n=t.get("cache_n"),
                       prompt_ms=t.get("prompt_ms"), gen=t.get("predicted_n"), decode_ms=t.get("predicted_ms"),
                       read_s=(t.get("prompt_ms", 0) + t.get("predicted_ms", 0)) / 1e3, out=msg, date=date_of(msg))
            with open(out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"{name:12s} {img} prompt {row['prompt_n']} (cached {row['cache_n']}) {row['prompt_ms']:.0f} ms  "
                  f"decode {row['gen']} tok {row['decode_ms']:.0f} ms  read {row['read_s']:.2f}s  {row['date']}", flush=True)
        sb.sh("setprop debug.oculus.cpuLevel ''; setprop debug.oculus.gpuLevel ''")
    sb.sh("pkill -f '[l]lama-server-opencl'")


def summarize(path):
    rows = [json.loads(l) for l in open(path)]
    cfgs = list(dict.fromkeys(r["config"] for r in rows))
    ref = {r["image"]: r for r in rows if r["config"] == cfgs[0]}
    print(f"{'config':12s} {'prefill s':>9s} {'decode s':>8s} {'read s':>7s} {'median':>7s} {'speedup':>8s} {'same date':>9s}")
    for c in cfgs:
        rs = [r for r in rows if r["config"] == c]
        rd = st.mean(r["read_s"] for r in rs)
        print(f"{c:12s} {st.mean(r['prompt_ms'] for r in rs) / 1e3:9.2f} {st.mean(r['decode_ms'] for r in rs) / 1e3:8.2f} "
              f"{rd:7.2f} {st.median(r['read_s'] for r in rs):7.2f} "
              f"{st.mean(ref[r['image']]['read_s'] for r in rs) / rd:7.2f}x "
              f"{sum(tuple(r['date']) == tuple(ref[r['image']]['date']) for r in rs):6d}/{len(rs)}")


if __name__ == "__main__":
    import os
    if os.environ.get("E2E_IMAGES"):                 # file with one image name per line
        sb.IMAGES = Path(os.environ["E2E_IMAGES"]).read_text().split()
    if os.environ.get("E2E_CROPS"):                  # crop folder (default: server-detector crops C_S960_x2)
        sb.CROPS = Path(os.environ["E2E_CROPS"])
    if sys.argv[1] == "--sum":
        summarize(sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2:])
