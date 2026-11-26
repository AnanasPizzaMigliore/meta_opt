"""Product-level paired bootstrap between two live runs: difference (B - A) in wrong-or-invented dates and in correct dates.
  llm-env python live_paired.py A.tsv B.tsv [RULE_A RULE_B]
"""
import collections, sys
import numpy as np
import real_eval as re_
from live_eval import tri

def load(path, rule=None):
    out = {}
    for l in open(path):
        if not l.strip() or l[0] == "#" or l.startswith("clip"): continue
        clip, r, ans, ms, *_ = l.rstrip("\n").split("\t")
        if rule and r != rule: continue
        p, c = clip.split("_")[0], clip.split("_")[1]
        o = re_.outcome(tri(ans), None if c == "c" else tuple(re_.TRUTH[p]))
        out[clip] = (o, float(ms) / 1e3)
    return out

a = load(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else None)
b = load(sys.argv[2], sys.argv[4] if len(sys.argv) > 4 else None)
clips = sorted(set(a) & set(b)); prods = sorted({c.split("_")[0] for c in clips})
bad = lambda o: o in ("wrong", "false")
per = collections.defaultdict(lambda: np.zeros(2))
for c in clips:
    p = c.split("_")[0]
    per[p] += [bad(b[c][0]) - bad(a[c][0]), (b[c][0] == "correct") - (a[c][0] == "correct")]
M = np.array([per[p] for p in prods]); rng = np.random.default_rng(0)
bs = np.array([M[rng.integers(0, len(M), len(M))].sum(0) for _ in range(10000)])
for i, name in enumerate(["wrong-or-invented", "correct"]):
    lo, hi = np.percentile(bs[:, i], [2.5, 97.5])
    print(f"{name:18s} B-A {M[:, i].sum():+.0f}  95% CI [{lo:+.0f}, {hi:+.0f}]  ({len(clips)} clips, {len(prods)} products)")
both = [c for c in clips if a[c][0] == "correct" and b[c][0] == "correct"]
if both:
    ta, tb = np.median([a[c][1] for c in both]), np.median([b[c][1] for c in both])
    print(f"both correct on {len(both)} clips: median time A {ta:.1f} s, B {tb:.1f} s; same answer on {sum(a[c][0]==b[c][0] for c in clips)}/{len(clips)} clips")
