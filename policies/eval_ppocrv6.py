"""Table 3 outcomes with PP-OCRv6 localisation and OCR dates (sensitivity check), next to the PP-OCRv5 run.

  python eval_ppocrv6.py tiny small medium
Same rules, same frames (first 7 sampled frames of the 148 held-out clips), same VLM and decoding; only the
detector + recogniser differ (and therefore the gate level, the crop the VLM reads, and the OCR date).
"""
import collections
import json
import random
import sys
from pathlib import Path

import real_eval as re_
import replay_time as rt

V6 = re_.SET / "clips_ppocrv6"


def run(L, label):
    re_.L = L
    products = [p for p in sorted(re_.TRUTH) if p >= "p004"]
    clips = {(p, c): rt.clip_frames(p, c, {}, {}) for p in products for c in "abdc"}
    allf = [x for fr in clips.values() for x in fr]
    lv = collections.Counter(x["level"] for x in allf)
    ocr_ok = sum(1 for (p, c), fr in clips.items() if c != "c" for x in fr
                 if x["ocr"] != rt.rp.NONE and re_.match(x["ocr"], tuple(re_.TRUTH[p])))
    vlm_ok = sum(1 for (p, c), fr in clips.items() if c != "c" for x in fr
                 if x["vlm"] != rt.rp.NONE and x["vlm"] not in ("abstain",) and re_.match(x["vlm"], tuple(re_.TRUTH[p])))
    nd = sum(len(fr) for (p, c), fr in clips.items() if c != "c")
    print(f"\n=== {label}: frames {len(allf)}; levels {dict(lv)}; per dated frame: OCR date correct {ocr_ok}/{nd} "
          f"({100 * ocr_ok / nd:.1f}%), VLM correct {vlm_ok}/{nd} ({100 * vlm_ok / nd:.1f}%)")
    print(f"{'rule':38s} {'corr':>4s} {'wrong':>5s} {'nodate':>6s} {'abst':>4s} {'false':>5s} {'prec':>6s} {'reads':>5s}")
    for name, f in rt.RULES.items():
        cnt = collections.Counter(); nr = []
        for (p, c), fr in clips.items():
            for x in fr:
                x["v"] = x["vlm"]
            a, n, rd = f(fr)
            cnt[re_.outcome(a, None if c == "c" else tuple(re_.TRUTH[p]))] += 1
            nr.append(len(rd))
        em = cnt["correct"] + cnt["wrong"] + cnt["false"]
        print(f"{name:38s} {cnt['correct']:4d} {cnt['wrong']:5d} {cnt['missed']:6d} {cnt['abstain']:4d} {cnt['false']:5d} "
              f"{100 * cnt['correct'] / max(1, em):6.1f} {sum(nr) / len(nr):5.1f}")
    trans, net = collections.Counter(), {p: 0 for p in products}
    for p in products:
        for c in "abd":
            fr = clips[(p, c)]
            for x in fr:
                x["v"] = x["vlm"]
            t = tuple(re_.TRUTH[p])
            a = re_.outcome(rt.ocr_confirmed(fr)[0], t); b = re_.outcome(rt.ocr_until2(fr)[0], t)
            trans[(a, b)] += 1; net[p] += (a == "correct") - (b == "correct")
    rng = random.Random(1)
    v = sorted(sum(net[rng.choice(products)] for _ in products) for _ in range(5000))
    print(f"confirmed vs OCR-only (until 2): net {sum(net.values())} (95% {v[125]}..{v[4874]}); {dict(trans)}")


if __name__ == "__main__":
    run(re_.SET / "clips", "PP-OCRv5 mobile (paper)")
    for tier in sys.argv[1:]:
        d = V6 / tier
        if all((d / f"{p}_{c}_rot0" / "reads_cap1024.json").exists() for p in sorted(re_.TRUTH) if p >= "p004" for c in "abdc"):
            run(d, f"PP-OCRv6 {tier}")
        else:
            print(f"\n(PP-OCRv6 {tier}: reads not complete yet)")
