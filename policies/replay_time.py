"""Headset replay, part 3: time to answer or abstain per clip, for each stopping rule, on the 37 held-out products.

Every frame's cost is measured on the Quest 3S: localisation = detector + post-processing + recogniser on the real
frame and its real line crops (ocrreplay, CPU, 4 threads); read = one VLM read of the frame's crop (replay_vlm.py,
GPU, deployed config). A clip's time under a rule is the sum of those costs along the rule's own stopping path
(frames are processed one after another; nothing overlaps). Decisions are taken twice: from the server reads used
in Table 3 (to time exactly the outcomes reported there) and from the headset's own reads (deployed outcomes).

  python replay_time.py [replay_vlm.jsonl] [replay_loc.tsv]
"""
import collections
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

import real_eval as re_
import reading_policy as rp

DV = Path("/home/user/qwen/phone_bench/dvfs/revision")
VLM = Path(sys.argv[1]) if len(sys.argv) > 1 else DV / "replay_vlm.jsonl"
LOCT = Path(sys.argv[2]) if len(sys.argv) > 2 else DV / "replay_loc.tsv"
B = rp.BUDGET


def sent_tri(text):
    m = re.search(r"reads (\d{4}|\?{4})-(\d{2}|\?\?)-(\d{2}|\?\?)", text)
    if not m:
        return rp.NONE
    return tuple(None if "?" in g else int(g) for g in m.groups())


def clip_frames(p, c, hv, lt):
    d = re_.L / f"{p}_{c}_rot0"
    m, r = json.load(open(d / "meta.json")), json.load(open(d / "reads_cap1024.json"))
    out = []
    for f in sorted(r)[:B]:
        key = f"{p}_{c}_rot0/x2/{f}"
        h = hv.get(key)
        out.append(dict(level=m[f]["level"], vlm=rp.tri(r[f]), ocr=re_.ocr_date(m[f]),
                        hvlm=sent_tri(h["out"]) if h else None, read=h["read_s"] if h else None,
                        loc=lt.get(f"{p}_{c}_rot0_{f}")))
    return out


# each rule: frames (with key 'v' = the VLM triple used) -> (answer | 'abstain', [indices localised], [indices read])
def until2(fr, gated=False, check_valid=False, confirm=False):
    seen = collections.Counter()
    read = []
    for i, x in enumerate(fr):
        if gated and x["level"] not in rp.GATE_LEVELS:
            continue
        read.append(i)
        v = x["v"]
        if confirm and v != rp.NONE and v == x["ocr"]:
            return v, i + 1, read
        if check_valid and not rp.valid(v):
            continue
        seen[v] += 1
        if seen[v] == 2:
            return v, i + 1, read
    return "abstain", len(fr), read


def ocr_confirmed(fr):
    read = []
    for i, x in enumerate(fr):
        read.append(i)
        if x["v"] != rp.NONE and x["v"] == x["ocr"]:
            return x["v"], i + 1, read
    return "abstain", len(fr), read


def ocr_first(fr):
    for i, x in enumerate(fr):
        if x["ocr"] != rp.NONE:
            return x["ocr"], i + 1, []
    return "abstain", len(fr), []


def ocr_until2(fr):
    seen = collections.Counter()
    for i, x in enumerate(fr):
        if x["ocr"] != rp.NONE:
            seen[x["ocr"]] += 1
            if seen[x["ocr"]] == 2:
                return x["ocr"], i + 1, []
    return "abstain", len(fr), []


RULES = {
    "Until 2 agree": lambda fr: until2(fr),
    "Fixed policy (gated, calendar)": lambda fr: until2(fr, gated=True, check_valid=True),
    "Until OCR confirms or 2 agree": lambda fr: until2(fr, confirm=True),
    "VLM confirmed by OCR": ocr_confirmed,
    "OCR only: first parsed date": ocr_first,
    "OCR only: until 2 OCR dates agree": ocr_until2,
}


def q(a, p):
    return float(np.percentile(a, p))


def main():
    hv = {json.loads(l)["image"]: json.loads(l) for l in open(VLM)} if VLM.exists() else {}
    lt = {}
    if LOCT.exists():
        for line in open(LOCT):
            if line.startswith("frame"):
                continue
            t = line.rstrip("\n").split("\t")
            lt[t[0]] = float(t[8]) / 1e3
    products = [p for p in sorted(re_.TRUTH) if p >= "p004"]
    clips = {(p, c): clip_frames(p, c, hv, lt) for p in products for c in "abdc"}
    allf = [x for fr in clips.values() for x in fr]
    have_loc = sum(x["loc"] is not None for x in allf); have_read = sum(x["read"] is not None for x in allf)
    print(f"{len(clips)} clips, {len(allf)} frames; headset localisation for {have_loc}, headset read for {have_read}")
    if have_loc:
        L_ = [x["loc"] for x in allf if x["loc"] is not None]
        print(f"localisation per frame: median {st.median(L_):.2f} s, p90 {q(L_, 90):.2f} s")
    if have_read:
        R_ = [x["read"] for x in allf if x["read"] is not None]
        print(f"read per frame: median {st.median(R_):.2f} s, p90 {q(R_, 90):.2f} s")
        same = sum(x["hvlm"] == x["vlm"] for x in allf if x["hvlm"] is not None)
        print(f"headset read = server read on {same}/{have_read} frames")
    for src in ("vlm", "hvlm"):
        if src == "hvlm" and have_read < len(allf):
            print("\n(headset decisions need every frame read on the headset; skipped)")
            continue
        print(f"\n=== decisions from {'server reads (Table 3)' if src == 'vlm' else 'headset reads (deployed)'} ===")
        print(f"{'rule':36s} {'corr':>4s} {'wrong':>5s} {'nodate':>6s} {'abst':>4s} {'false':>5s} {'prec':>6s} "
              f"{'loc':>4s} {'reads':>5s} | {'time to answer or abstain (s)':>30s}")
        for name, f in RULES.items():
            cnt = collections.Counter(); T = []; Td = []; nl = []; nr = []; complete = True
            for (p, c), fr in clips.items():
                for x in fr:
                    x["v"] = x[src]
                a, n, rd = f(fr)
                o = re_.outcome(a, None if c == "c" else tuple(re_.TRUTH[p]))
                cnt[o] += 1
                nl.append(n); nr.append(len(rd))
                if any(fr[i]["loc"] is None for i in range(n)) or any(fr[i]["read"] is None for i in rd):
                    complete = False
                    continue
                t = sum(fr[i]["loc"] for i in range(n)) + sum(fr[i]["read"] for i in rd)
                T.append(t)
                if c != "c":
                    Td.append(t)
            em = cnt["correct"] + cnt["wrong"] + cnt["false"]
            tt = (f"median {st.median(T):5.1f}  p90 {q(T, 90):5.1f}  mean {st.mean(T):5.1f}" if complete and T
                  else f"incomplete ({len(T)} clips timed)")
            print(f"{name:36s} {cnt['correct']:4d} {cnt['wrong']:5d} {cnt['missed']:6d} {cnt['abstain']:4d} "
                  f"{cnt['false']:5d} {100 * cnt['correct'] / max(1, em):6.1f} {st.mean(nl):4.1f} {st.mean(nr):5.1f} | {tt}")


if __name__ == "__main__":
    main()
