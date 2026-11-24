"""Tuned OCR-only baseline, step 2: evaluate the OCR-only settings fixed in ocr_conf_choice.json (chosen on the
simulated development data) once on the 37 held-out real products, against the OCR cross-check, paired by product.
Same frames, same OCR text (rerun with scores, identical text), same date rules; time = localisation along the stopping
path (OCR-only) or localisation + VLM calls (cross-check, OCR-date skip), per-frame costs measured on the headset.
Also reports the pilot products (p001-p003) as a second development check, and the whole sweep for transparency.

  llm-env python ocr_conf_eval.py
"""
import collections
import json
import random
import statistics as st
from pathlib import Path

import numpy as np

import ocr_conf_tune as T
import real_eval as re_
import replay_time as rt

HERE = Path(__file__).parent
SC = json.load(open(HERE / "ocr_scores_v5.json"))
CH = json.load(open(HERE / "ocr_conf_choice.json"))


def frames(p, c, hv, lt):
    fr = rt.clip_frames(p, c, hv, lt)
    keys = sorted(json.load(open(re_.L / f"{p}_{c}_rot0" / "reads_cap1024.json")))[:7]
    for x, f in zip(fr, keys):
        s = SC[f"{p}_{c}_rot0/{f}"]
        dt, conf = T.select_conf(list(zip(s["texts"], s["scores"])))
        x["oc"] = (T.norm(dt) if dt else None, conf)
        x["v"] = x["vlm"]
    return fr


def ocr_rule(fr, k, tau):
    a, n = T.rule([x["oc"] for x in fr], k, tau)
    return (a if a is not None else "abstain"), n, []


def evaluate(products, name, f, hv, lt):
    per, T_ = {}, []
    for p in products:
        per[p] = []
        for c in "abdc":
            fr = frames(p, c, hv, lt)
            a, n, rd = f(fr)
            o = re_.outcome(a, None if c == "c" else tuple(re_.TRUTH[p]))
            per[p].append((c, o))
            if all(x["loc"] is not None for x in fr[:n]):
                T_.append(sum(fr[i]["loc"] for i in range(n)) + sum(fr[i]["read"] for i in rd))
    cnt = collections.Counter(o for v in per.values() for _, o in v)
    em = cnt["correct"] + cnt["wrong"] + cnt["false"]
    t = f"median {st.median(T_):.1f} (p90 {np.percentile(T_, 90):.1f}) s" if T_ else "--"
    print(f"{name:40s} correct {cnt['correct']:3d} wrong {cnt['wrong']:2d} false {cnt['false']:2d} "
          f"cov {100 * (cnt['correct'] + cnt['wrong']) / sum(1 for v in per.values() for c, _ in v if c != 'c'):5.1f}% "
          f"prec {100 * cnt['correct'] / em if em else float('nan'):5.1f}% | {t}")
    return per


def paired(a, b, products, n=5000):
    net = {p: sum((x == "correct") - (y == "correct") for (_, x), (_, y) in zip(a[p], b[p])) for p in products}
    err = {p: sum((x in ("wrong", "false")) - (y in ("wrong", "false")) for (_, x), (_, y) in zip(a[p], b[p])) for p in products}
    rng = random.Random(1)
    bs = sorted(sum(net[rng.choice(products)] for _ in products) for _ in range(n))
    be = sorted(sum(err[rng.choice(products)] for _ in products) for _ in range(n))
    return sum(net.values()), (bs[int(.025 * n)], bs[int(.975 * n)]), sum(err.values()), (be[int(.025 * n)], be[int(.975 * n)])


def main():
    hv = {json.loads(l)["image"]: json.loads(l) for l in open(rt.VLM)}
    lt = {}
    for line in open(rt.LOCT):
        if not line.startswith("frame"):
            t = line.split("\t")
            lt[t[0]] = float(t[8]) / 1e3
    held = [p for p in sorted(re_.TRUTH) if p >= "p004"]
    pilot = [p for p in sorted(re_.TRUTH) if p < "p004"]
    a, b = CH["fallback_a"], CH["fallback_b"]
    print(f"registered: (a) k={a['k']} tau={a['tau']}  (b) k={b['k']} tau={b['tau']}  at {CH['fallback_registered_at']}")
    for label, prods in (("PILOT (development check)", pilot), ("HELD-OUT (37 products)", held)):
        print(f"\n=== {label} ===")
        xc = evaluate(prods, "OCR cross-check (OCR-date skip)", rt.ocr_confirmed_skip, hv, lt)
        ra = evaluate(prods, f"OCR-only (a) k={a['k']} tau={a['tau']}", lambda fr: ocr_rule(fr, a["k"], a["tau"]), hv, lt)
        rb = evaluate(prods, f"OCR-only (b) k={b['k']} tau={b['tau']}", lambda fr: ocr_rule(fr, b["k"], b["tau"]), hv, lt)
        r2 = evaluate(prods, "OCR repeat (k=2, no threshold)", lambda fr: ocr_rule(fr, 2, 0.0), hv, lt)
        if label.startswith("HELD"):
            for nm, r in (("(a)", ra), ("(b)", rb), ("OCR repeat", r2)):
                d, ci, e, cie = paired(xc, r, prods)
                print(f"  cross-check minus {nm}: correct {d:+d} (95% CI {ci[0]:+d}..{ci[1]:+d}); "
                      f"wrong+false {e:+d} (95% CI {cie[0]:+d}..{cie[1]:+d})")
    print("\n=== held-out sweep (for transparency; not used for selection) ===")
    for k in T.K:
        for tau in (0.0, 0.8, 0.9, 0.95, 0.97, 0.99):
            evaluate(held, f"OCR-only k={k} tau={tau}", lambda fr, k=k, tau=tau: ocr_rule(fr, k, tau), hv, lt)


if __name__ == "__main__":
    main()
