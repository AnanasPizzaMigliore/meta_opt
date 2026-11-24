"""Deployed-backend accuracy: headset (+stop_if, OpenCL, default clocks, cap 1024) on the 544 on-device-detector
crops (0003.jpg is the warm-up image and is excluded), against the desktop Vulkan reads of the same crops."""
import json, importlib.util, re
spec = importlib.util.spec_from_file_location("ev", "/home/user/qwen/eval_gguf.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
gt = {e["filename"]: e for e in json.load(open(ev.GT_FILE))}
test = [e["filename"] for e in json.load(open(ev.TEST_JSON))]


def parse(out):
    m = re.search(r"reads (\d{4}|\?{4})-(\d{1,2}|\?\?)-(\d{1,2}|\?\?)", out)
    if m:
        return dict(zip(("year", "month", "day"), (None if "?" in x else int(x) for x in m.groups())))
    if "No expiration" in out:
        return {"year": None, "month": None, "day": None}
    return {"parse_error": out}


H = {}
for l in open("/home/user/qwen/phone_bench/dvfs/revision/accuracy_all.jsonl"):
    r = json.loads(l)
    H[r["image"]] = parse(r["out"])
D = json.load(open("/home/user/qwen/quest_sim/results/Cm_S960_x2_cap1024.json"))
files = [f for f in test if f in H]
nod = [f for f in files if all(gt[f].get(k) is None for k in ("year", "month", "day"))]


def tri(p):
    return tuple(ev.norm_year(p.get("year")) if k == "year" else ev.norm(p.get(k)) for k in ("year", "month", "day"))


def report(name, R):
    R = {f: (R[f] if isinstance(R[f], dict) else {}) for f in files}
    s = ev.compute_accuracy(R, gt, files)["ALL"]
    false = sum(tri(R[f]) != (None, None, None) for f in nod)
    perr = sum("parse_error" in R[f] or "error" in R[f] for f in files)
    print(f"{name:28s} exact {s['full_correct']}/{s['gt_has_date']} = {100 * s['full_correct'] / s['gt_has_date']:.1f}%  "
          f"false dates {false}/{len(nod)}  unparsed {perr}")
    return R


h = report("headset +stop_if (OpenCL)", H)
d = report("desktop tv, cap 1024 (Vulkan)", D)
same = sum(tri(h[f]) == tri(d[f]) for f in files)
print(f"identical (y,m,d) answers headset vs desktop: {same}/{len(files)}")
