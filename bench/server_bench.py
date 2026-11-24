"""llama-server ON the Quest (OpenCL LLM, CPU ViT), driven from this machine through adb.

Requests are sent from a headset shell with toybox nc, so no port forwarding is needed. Each config
starts a fresh server with the given flags; every image is read once as warm-up for the config and
then measured. Optional per-read DVFS: levels are set before each request (static).

  python server_bench.py <out.jsonl> <config-name>...   (configs below)
"""
import base64
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ADB = __import__("os").environ.get("E2E_ADB") or str(Path.home() / "android-sdk/platform-tools/adb")  # E2E_ADB: e.g. the guarded adbq
R = "/data/local/tmp/edb"
PORT = 8898
HERE = Path(__file__).parent
PROMPT = Path("/home/user/qwen/phone_bench/bundle/prompt.txt").read_text()
CROPS = Path("/home/user/qwen/quest_sim/C_S960_x2")
IMAGES = ["0047.jpg", "0129.jpg", "0239.jpg", "0294.jpg", "0536.jpg", "0577.jpg", "0669.jpg", "1381.jpg",
          "1547.jpg", "1563.jpg"]
MTP = "qwen35_groupA_a3mtp_Q4_0.gguf"
BASE = "qwen35_groupA_a3_Q4_0.gguf"
TV = "qwen35_groupA_a3_Q4_0_tv.gguf"
MTPTV = "qwen35_groupA_a3mtp_Q4_0_tv.gguf"
CONFIGS = {   # name: (model, extra server args, (cpuLevel, gpuLevel) or None)
    "base":        (BASE, [], None),
    "mtp1":        (MTP, ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"], None),
    "mtp2":        (MTP, ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"], None),
    "mtp3":        (MTP, ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"], None),
    "cpu_base":    (BASE, ["-dev", "none"], None),
    "cpu_mtp1":    (MTP, ["-dev", "none", "--spec-type", "draft-mtp", "--spec-draft-n-max", "1"], None),
    "cpu_mtp2":    (MTP, ["-dev", "none", "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"], None),
    "cpu_mtp3":    (MTP, ["-dev", "none", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"], None),
    "tv":          (TV, [], None),
    "tv_mtp3":     (MTPTV, ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"], None),
    "cpu_tv":      (TV, ["-dev", "none"], None),
    "cpu_tv_mtp1": (MTPTV, ["-dev", "none", "--spec-type", "draft-mtp", "--spec-draft-n-max", "1"], None),
    "cpu_tv_mtp2": (MTPTV, ["-dev", "none", "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"], None),
    "cpu_tv_mtp3": (MTPTV, ["-dev", "none", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"], None),
    "cpu_mtp4":    (MTP, ["-dev", "none", "--spec-type", "draft-mtp", "--spec-draft-n-max", "4"], None),
}


def sh(cmd, timeout=900):
    return subprocess.run([ADB, "shell", cmd], capture_output=True, text=True, errors="replace",
                          timeout=timeout).stdout


def start_server(model, extra):
    sh("pkill -f '[l]lama-server-opencl'; sleep 1")
    args = " ".join(extra)
    sh(f"cd {R} && (nohup taskset 3c ./llama-server-opencl -m {model} --mmproj qwen35_groupA_a3_mmproj_Q8_0.gguf "
       f"-ngl 99 --no-mmproj-offload -t 4 -c 4096 -np 1 --port {PORT} --image-max-tokens 1024 "
       f"--image-min-tokens 64 {args} > server.log 2>&1 &)")
    for _ in range(120):
        if '"ok"' in sh(f"printf 'GET /health HTTP/1.0\\r\\n\\r\\n' | nc -q 5 127.0.0.1 {PORT}"):
            return
        time.sleep(1)
    raise RuntimeError("server did not come up:\n" + sh(f"tail -20 {R}/server.log"))


def request(img, extra_payload=None):
    content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                base64.b64encode((CROPS / img).read_bytes()).decode()}},
               {"type": "text", "text": PROMPT}]
    payload = {"messages": [{"role": "user", "content": content}], "max_tokens": 128, "temperature": 0,
               "stream": False, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
    payload.update(extra_payload or {})
    body = json.dumps(payload).encode()
    req = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\n"
           f"Content-Length: {len(body)}\r\n\r\n").encode() + body
    local = HERE / "req.http"
    local.write_bytes(req)
    subprocess.run([ADB, "push", str(local), f"{R}/req.http"], capture_output=True)
    t0 = time.time()
    raw = sh(f"{R}/httpc {PORT} {R}/req.http")
    wall = time.time() - t0
    js = json.loads(raw[raw.index("{"):])
    return js, wall


def vit_ms_from_log(n_before):
    lines = sh(f"cat {R}/server.log").splitlines()[n_before:]
    m = [re.search(r"image (?:slice )?encoded in (\d+) ms", l) for l in lines]
    m = [int(x.group(1)) for x in m if x]
    return (m[-1] if m else None), len(lines) + n_before


def main():
    out = Path(sys.argv[1])
    for name in sys.argv[2:]:
        model, extra, lv = CONFIGS[name]
        start_server(model, extra)
        if lv:
            sh(f"setprop debug.oculus.cpuLevel {lv[0]}; setprop debug.oculus.gpuLevel {lv[1]}")
        n_log = len(sh(f"cat {R}/server.log").splitlines())
        request(IMAGES[0])                                 # warm-up
        for img in IMAGES:
            n_log = len(sh(f"cat {R}/server.log").splitlines())
            js, wall = request(img)
            t = js.get("timings", {})
            vit, _ = vit_ms_from_log(n_log)
            msg = js["choices"][0]["message"]["content"]
            row = dict(config=name, image=img, wall_s=wall, vit_ms=vit, prompt_n=t.get("prompt_n"),
                       prompt_ms=t.get("prompt_ms"), gen=t.get("predicted_n"), decode_ms=t.get("predicted_ms"),
                       draft_n=t.get("draft_n"), draft_acc=t.get("draft_n_accepted"), out=msg)
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"{name:8s} {img} prompt {row['prompt_n']} tok {row['prompt_ms']:.0f} ms (vit {vit}) "
                  f"decode {row['gen']} tok {row['decode_ms']:.0f} ms  drafts {row['draft_n']}/{row['draft_acc']}  "
                  f"wall {wall:.2f}s  {msg.splitlines()[0][:60]}", flush=True)
        sh("setprop debug.oculus.cpuLevel ''; setprop debug.oculus.gpuLevel ''")
    sh("pkill -f '[l]lama-server-opencl'")


if __name__ == "__main__":
    main()
