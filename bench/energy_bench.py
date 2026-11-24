"""Energy per read on the Quest 3S, on battery (adb over Wi-Fi), for the e2e_bench configurations.

Per configuration: start llama-server, keep a Perfetto battery trace (current + voltage every 100 ms) running,
measure IDLE_S of idle with the model loaded, then read each image; every request is wrapped in boot-clock
markers taken on the headset. Energy of a read = integral of battery power over its window; 'above idle'
subtracts the configuration's own idle power. Two rounds, the second in reverse order, so battery drain and
warming do not favour any configuration.

  python energy_bench.py out.jsonl
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import e2e_bench as e2e
import server_bench as sb

sb.ADB = str(Path.home() / "qwen/phone_bench/adbq")        # guarded adb: never starts a local server
CFG_DIR = "/data/misc/perfetto-configs/pwr_long.cfg"
TRACE = "/data/misc/perfetto-traces/energy.pftrace"
HERE = Path(__file__).parent
IDLE_S = 10
IMAGES = Path(HERE / "images40.txt").read_text().split()[:15]
CONFIGS = ["orig", "+gpu_vit", "+task_vocab", "+stop_if", "+stop_if_max", "+stop_if_eco", "+cap384_if_max",
           "+cap384_if_eco"]
import os
if os.environ.get("E2E_IMAGES"):                    # revision: the seeded 40-crop sample
    IMAGES = Path(os.environ["E2E_IMAGES"]).read_text().split()
if os.environ.get("E2E_CROPS"):
    sb.CROPS = Path(os.environ["E2E_CROPS"])
if os.environ.get("ENERGY_CONFIGS"):
    CONFIGS = os.environ["ENERGY_CONFIGS"].split()
STOP_LEVEL = 25


def uptime():
    return float(sb.sh("cut -d' ' -f1 /proc/uptime").strip())


def battery():
    out = sb.sh("dumpsys battery")
    g = lambda k: next((l.split(":")[1].strip() for l in out.splitlines() if l.strip().startswith(k)), "")
    return int(g("level") or 0), int(g("temperature") or 0) / 10, g("AC powered"), g("USB powered")


def timed_read(img, text_first, stop):
    """like e2e.read, but brackets the request with on-device boot-clock times."""
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
    (HERE / "req.http").write_bytes(req)
    for attempt in range(4):                       # Wi-Fi adb occasionally drops a push/shell; retry
        subprocess.run([sb.ADB, "push", str(HERE / "req.http"), f"{sb.R}/req.http"], capture_output=True)
        raw = sb.sh(f"t0=$(cut -d' ' -f1 /proc/uptime); {sb.R}/httpc {sb.PORT} {sb.R}/req.http; "
                    f"t1=$(cut -d' ' -f1 /proc/uptime); echo; echo MARK $t0 $t1")
        if "MARK" in raw and "{" in raw[:raw.rindex("MARK")]:
            break
        print(f"  retry {attempt + 1} for {img}", flush=True)
        time.sleep(5)
    mark = raw[raw.rindex("MARK"):].split()
    js = json.loads(raw[raw.index("{"):raw.rindex("MARK")])
    return js, float(mark[1]), float(mark[2])


def energy(trace, windows):
    from perfetto.trace_processor import TraceProcessor
    tp = TraceProcessor(trace=str(trace))
    rows = {}
    for r in tp.query("select t.name n, c.ts ts, c.value v from counter c join counter_track t on c.track_id=t.id "
                      "where t.name in ('batt.current_ua','batt.voltage_uv') order by c.ts"):
        rows.setdefault(r.n, []).append((r.ts / 1e9, r.v))
    tp.close()
    volt = dict(rows["batt.voltage_uv"])
    pw = [(t, -i * volt.get(t, 3.9e6) / 1e12) for t, i in rows["batt.current_ua"]]   # W, discharge > 0

    def integ(a, z):
        e = 0.0
        for (t0, p0), (t1, _) in zip(pw, pw[1:]):
            lo, hi = max(a, t0), min(z, t1)
            if hi > lo:
                e += p0 * (hi - lo)
        return e
    if not pw or pw[-1][0] < max(z for _, z in windows) or pw[0][0] > min(a for a, _ in windows):
        raise RuntimeError(f"battery trace {pw[0][0] if pw else None}..{pw[-1][0] if pw else None} does not cover "
                           f"the read windows {min(a for a, _ in windows)}..{max(z for _, z in windows)}")
    return [integ(a, z) for a, z in windows], len(pw)


def run_config(name, rnd, out):
    model, env, extra, tf, stop, lv = e2e.CONFIGS[name]
    e2e.start(model, env, extra)
    if lv:
        sb.sh(f"setprop debug.oculus.cpuLevel {lv[0]}; setprop debug.oculus.gpuLevel {lv[1]}")
    timed_read("0003.jpg", tf, stop)                # warm-up, outside the set
    pid = sb.sh(f"rm -f {TRACE}; perfetto --background --txt -c {CFG_DIR} -o {TRACE}").strip()
    time.sleep(1.0)
    i0 = uptime()
    time.sleep(IDLE_S)
    i1 = uptime()
    reads = []
    for img in IMAGES:
        js, a, z = timed_read(img, tf, stop)
        t = js.get("timings", {})
        reads.append(dict(image=img, t0=a, t1=z, prompt_n=t.get("prompt_n"), prompt_ms=t.get("prompt_ms"),
                          gen=t.get("predicted_n"), decode_ms=t.get("predicted_ms"),
                          date=e2e.date_of(js["choices"][0]["message"]["content"])))
    time.sleep(1.0)
    sb.sh(f"kill -TERM {pid}; for i in 1 2 3 4 5 6 7 8; do kill -0 {pid} 2>/dev/null || break; sleep 0.5; done")
    local = HERE / "traces" / f"energy_{rnd}_{name.strip('+')}.pftrace"
    local.parent.mkdir(exist_ok=True)
    subprocess.run([sb.ADB, "pull", TRACE, str(local)], capture_output=True)
    sb.sh("setprop debug.oculus.cpuLevel ''; setprop debug.oculus.gpuLevel ''")
    es, n = energy(local, [(i0, i1)] + [(r["t0"], r["t1"]) for r in reads])
    p_idle = es[0] / (i1 - i0)
    lvl, temp, ac, usb = battery()
    with open(out, "a") as fh:
        for r, e in zip(reads, es[1:]):
            r.update(config=name, round=rnd, e_j=e, e_above_idle_j=e - p_idle * (r["t1"] - r["t0"]), p_idle_w=p_idle,
                     window_s=r["t1"] - r["t0"], battery=lvl, temp_c=temp, n_samples=n)
            fh.write(json.dumps(r) + "\n")
    tot = sum(es[1:])
    print(f"[r{rnd}] {name:15s} idle {p_idle:4.2f} W  reads {sum(r['t1'] - r['t0'] for r in reads):6.1f} s  "
          f"energy {tot:7.1f} J ({tot / len(reads):5.2f} J/read)  battery {lvl}% {temp} C  ac={ac} usb={usb}", flush=True)
    return lvl


def main(out):
    lvl, temp, ac, usb = battery()
    if ac != "false" or usb != "false":
        sys.exit(f"not on battery (AC {ac}, USB {usb})")
    sb.sh("am broadcast -a com.oculus.vrpowermanager.prox_close")      # keep the headset awake while unworn
    try:
        rounds = [int(x) for x in os.environ.get("ENERGY_ROUNDS", "1 2").split()]
        for rnd, order in ((k, CONFIGS if k % 2 else CONFIGS[::-1]) for k in rounds):
            for name in order:
                if run_config(name, rnd, out) <= STOP_LEVEL:
                    print("battery low, stopping")
                    return
    finally:
        sb.sh("pkill -f '[l]lama-server-opencl'; setprop debug.oculus.cpuLevel ''; setprop debug.oculus.gpuLevel ''; "
              "am broadcast -a com.oculus.vrpowermanager.automation_disable")


if __name__ == "__main__":
    main(sys.argv[1])
