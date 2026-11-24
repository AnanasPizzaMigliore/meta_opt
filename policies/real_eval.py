"""Real-video evaluation on the Quest 3S capture (pilot2 set, products p001-p040), policies side by side.

Per product: three dated clips (a bent arm, b close, d straight arm) and one no-date side (c). Frames are every
6th camera frame (~0.75 s apart), localised with the on-device detector and read by the VLM (1024-token cap).

Policies (the last is the fixed policy of reading_policy.py; rules unchanged):
  single read         one VLM read of one frame; pooled over all sampled frames (the expected single read)
  until 2 agree       read every frame in order, answer the first (y, m, d) seen twice, at most 7 frames
  OCR-confirmed       answer at the first frame (<= 7) where the VLM date equals the date the OCR rules parse
  gated               until 2 agree, reading only frames with a date-like or digit line
  gated + calendar    gated, and impossible dates never count (FIXED POLICY)

Scoring follows the paper's metric (eval_gguf.compute_accuracy): year and month must match; the day must match
when the printed date has one. Outcomes per dated clip: correct | wrong (another date) | missed ("no date") |
abstain; per no-date side: false (any date) | ok. Date precision = correct / all dates emitted (dated + no-date).
Intervals: 95% bootstrap over products (each product keeps its four clips together), 5000 resamples.

  python real_eval.py            all 40 products, with the fresh subset p004-p040 reported first
"""
import collections
import importlib.util
import json
import random
import statistics as st
from pathlib import Path

import reading_policy as rp

REPO = Path(__file__).resolve().parent.parent
SET = REPO / "data" / "real"
L = SET / "clips"
TRUTH = json.load(open(SET / "truth.json"))
spec = importlib.util.spec_from_file_location("ob", Path(__file__).resolve().parent / "ocr_baseline.py")
ob = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ob)
READ_S, LOC_S = 2.23, 3.27         # rough per-frame model only: median read (Table 1) and median localisation of a real frame (headset replay)


def match(a, t):
    return a[0] == t[0] and a[1] == t[1] and (t[2] is None or a[2] == t[2])


def ocr_date(m):
    chosen, _ = ob.select(m.get("all_lines", []))
    if not chosen:
        return rp.NONE
    y, mo, d = chosen
    y = y + 2000 if isinstance(y, int) and y < 100 else y
    return (y, mo, d)


def load(p, c):
    d = L / f"{p}_{c}_rot0"
    m, r = json.load(open(d / "meta.json")), json.load(open(d / "reads_cap1024.json"))
    return [dict(level=m[f]["level"], vlm=rp.tri(r[f]), ocr=ocr_date(m[f])) for f in sorted(r)]


def ocr_confirmed(frames, budget=rp.BUDGET):
    for i, fr in enumerate(frames[:budget]):
        if fr["vlm"] != rp.NONE and fr["vlm"] == fr["ocr"]:
            return fr["vlm"], i + 1, i + 1
    return "abstain", min(len(frames), budget), min(len(frames), budget)


POLICIES = {
    "until 2 agree": lambda fr: rp.run([(x["level"], x["vlm"]) for x in fr], gated=False, check_valid=False),
    "OCR-confirmed": ocr_confirmed,
    "gated": lambda fr: rp.run([(x["level"], x["vlm"]) for x in fr], gated=True, check_valid=False),
    "gated + calendar (fixed)": lambda fr: rp.run([(x["level"], x["vlm"]) for x in fr], gated=True, check_valid=True),
}


def outcome(ans, truth):
    if truth is None:
        return "ok" if ans in ("abstain", rp.NONE) else "false"
    if ans == "abstain":
        return "abstain"
    if ans == rp.NONE:
        return "missed"
    return "correct" if match(ans, truth) else "wrong"


def evaluate(products):
    """-> {policy: {product: [(clip, outcome, localised, reads)]}}, plus single-read frame tallies."""
    res = {k: {} for k in POLICIES}
    single = {}
    for p in products:
        t = tuple(TRUTH[p])
        single[p] = []
        for k in POLICIES:
            res[k][p] = []
        for c in "abdc":
            truth = None if c == "c" else t
            fr = load(p, c)
            for x in fr:
                single[p].append((c, outcome(x["vlm"], truth)))
            for k, f in POLICIES.items():
                a, n, r = f(fr)
                res[k][p].append((c, outcome(a, truth), n, r))
    return res, single


def tally(rows):
    c = collections.Counter(o for _, o, *_ in rows)
    emitted = c["correct"] + c["wrong"] + c["false"]
    return c, emitted


def stats(rows):
    c, emitted = tally(rows)
    dated = sum(1 for cl, *_ in rows if cl != "c")
    neg = len(rows) - dated
    return dict(dated=dated, neg=neg, correct=c["correct"], wrong=c["wrong"], missed=c["missed"],
                abstain=c["abstain"], false=c["false"], emitted=emitted,
                prec=c["correct"] / emitted if emitted else float("nan"))


def boot(per_product, fn, n=5000, seed=1):
    rng = random.Random(seed)
    ps = list(per_product)
    vals = []
    for _ in range(n):
        rows = [r for p in (rng.choice(ps) for _ in ps) for r in per_product[p]]
        v = fn(rows)
        if v == v:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def report(title, products):
    res, single = evaluate(products)
    print(f"\n=== {title}: {len(products)} products ({products[0]}..{products[-1]}) ===")
    srows = [r for p in products for r in single[p]]
    s = stats(srows)
    nd = sum(1 for cl, _ in srows if cl != "c"); nn = len(srows) - nd
    print(f"{'single read (per frame)':26s} exact {s['correct']}/{nd} = {100 * s['correct'] / nd:.1f}%  "
          f"wrong {s['wrong']}  missed {s['missed']} | false dates {s['false']}/{nn} = {100 * s['false'] / nn:.1f}% "
          f"| precision {100 * s['prec']:.1f}%")
    out = {}
    for k in POLICIES:
        rows = [r for p in products for r in res[k][p]]
        s = stats(rows)
        lo, hi = boot(res[k], lambda rr: stats(rr)["prec"])
        clo, chi = boot(res[k], lambda rr: (lambda z: z["correct"] / z["dated"])(stats(rr)))
        loc = st.mean(r[2] for r in rows); rd = st.mean(r[3] for r in rows)
        print(f"{k:26s} correct {s['correct']:3d}/{s['dated']} ({100 * s['correct'] / s['dated']:.1f}%, CI {100 * clo:.0f}-{100 * chi:.0f}) "
              f"wrong {s['wrong']:2d} missed {s['missed']:2d} abstain {s['abstain']:2d} | false {s['false']}/{s['neg']} "
              f"| precision {100 * s['prec']:.1f}% (CI {100 * lo:.0f}-{100 * hi:.0f}) "
              f"| frames {loc:.1f} localised, {rd:.1f} read, ~{loc * LOC_S + rd * READ_S:.1f} s modelled")
        out[k] = res[k]
    fixed = res["gated + calendar (fixed)"]
    print("fixed policy by posture:")
    for c, name in (("a", "bent arm"), ("b", "close"), ("d", "straight arm"), ("c", "no-date side")):
        cc = collections.Counter(o for p in products for cl, o, *_ in fixed[p] if cl == c)
        print(f"   {name:13s} {dict(cc)}")
    return out


if __name__ == "__main__":
    have = [p for p in sorted(TRUTH) if all((L / f"{p}_{c}_rot0" / "reads_cap1024.json").exists() for c in "abdc")]
    fresh = [p for p in have if p >= "p004"]
    report("FRESH (fixed policy set in advance)", fresh)
    report("ALL", have)
