"""Table 1 of the revision from latency_r{1,2,3}.jsonl (40 seeded crops x 7 configs x 3 rounds, headset, Wi-Fi adb).

Per image, read time = mean over the 3 rounds. Reported: median and p90 over images, paired speed-up
(median of per-image ratios against a reference) with an image bootstrap 95% interval, and the spread of
per-round medians (repeatability). Same-date = image answers identical to the baseline at the same clock in all rounds.
"""
import json
import random
import statistics as st
from pathlib import Path

D = Path(__file__).parent
CFGS = ["orig", "+gpu_vit", "+task_vocab", "+stop_if", "orig_max", "+stop_if_max", "+cap384_if_max"]
REF = {c: ("orig_max" if c.endswith("_max") else "orig") for c in CFGS}

rows = [dict(json.loads(l), rnd=k) for k in (1, 2, 3) for l in open(D / f"latency_r{k}.jsonl")]
imgs = sorted({r["image"] for r in rows})
T = {(r["config"], r["image"], r["rnd"]): r for r in rows}


def mean_read(c, i):
    return st.mean(T[c, i, k]["read_s"] for k in (1, 2, 3))


def p90(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(0.9 * (len(xs) - 1)))]


def boot(f, n=5000, seed=1):
    rng = random.Random(seed)
    vals = sorted(f([rng.choice(imgs) for _ in imgs]) for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def speedup(c, ref, sample):
    return st.median(mean_read(ref, i) / mean_read(c, i) for i in sample)


print(f"{len(imgs)} images, rounds {sorted({r['rnd'] for r in rows})}")
print(f"{'config':15s} {'median':>6s} {'p90':>6s} {'round medians':>20s} {'vs orig':>18s} {'vs orig_max':>18s} {'same date':>9s}")
for c in CFGS:
    reads = [mean_read(c, i) for i in imgs]
    rmed = [st.median(T[c, i, k]["read_s"] for i in imgs) for k in (1, 2, 3)]
    cols = []
    for ref in ("orig", "orig_max"):
        if ref == c:
            cols.append("-")
            continue
        s = speedup(c, ref, imgs)
        lo, hi = boot(lambda smp: speedup(c, ref, smp))
        cols.append(f"{s:.2f}x [{lo:.2f},{hi:.2f}]")
    same = sum(all(tuple(T[c, i, k]["date"]) == tuple(T[REF[c], i, k]["date"]) for k in (1, 2, 3)) for i in imgs)
    stable = sum(len({tuple(T[c, i, k]["date"]) for k in (1, 2, 3)}) == 1 for i in imgs)
    print(f"{c:15s} {st.median(reads):6.2f} {p90(reads):6.2f} {' / '.join(f'{x:.2f}' for x in rmed):>20s} "
          f"{cols[0]:>18s} {cols[1]:>18s} {same:4d}/{len(imgs)}  (stable across rounds {stable}/{len(imgs)})")

print("\nprefill / decode medians (s), tokens:")
for c in CFGS:
    pf = st.median(st.mean(T[c, i, k]["prompt_ms"] for k in (1, 2, 3)) for i in imgs) / 1e3
    dc = st.median(st.mean(T[c, i, k]["decode_ms"] for k in (1, 2, 3)) for i in imgs) / 1e3
    pn = st.median(T[c, i, 1]["prompt_n"] for i in imgs)
    gn = st.median(T[c, i, 1]["gen"] for i in imgs)
    print(f"  {c:15s} prefill {pf:5.2f}  decode {dc:5.2f}  prompt tokens {pn:6.0f}  generated {gn:4.0f}")
