"""Multi-frame voting over simulated frames: does reading several frames buy accuracy and fewer false dates?

  python vote_analysis.py [cap]        (reads/cap<cap>_f<k>.json, k = 0..6; default cap 384)

Policies over the first n frames of each product (answer = the (year, month, day) triple of a read):
  single      one frame (mean over the 7 frames)
  vote n      plurality over n frames, ties -> the answer seen first
  agree m/n   answer only if some triple appears >= m times in n frames, otherwise abstain
  until 2     read frames until one answer has been seen twice (max 7); cost = mean number of reads
Metrics: exact = fully correct / images with a date (the paper's metric; an abstention counts as wrong);
false dates = no-date images given a date; coverage = share answered; selective = correct / answered.
"""
import importlib.util
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
CAP = sys.argv[1] if len(sys.argv) > 1 else "384"
PFX = sys.argv[2] if len(sys.argv) > 2 else ""
sys.argv = sys.argv[:1]
spec = importlib.util.spec_from_file_location("ev", "/home/user/qwen/eval_gguf.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
files = [e["filename"] for e in json.load(open(ev.TEST_JSON))]
gt = {e["filename"]: e for e in json.load(open(ev.GT_FILE))}
K = 7
reads = [json.load(open(HERE / "reads" / f"{PFX}cap{CAP}_f{k}.json")) for k in range(K)]
F = ("year", "month", "day")
tri = lambda r: tuple(r.get(k) for k in F)
has_date = lambda f: any(gt[f].get(k) is not None for k in F)
dated = [f for f in files if has_date(f)]
nodate = [f for f in files if not has_date(f)]
NONE = (None, None, None)
ABSTAIN = "abstain"


def score(answers, n_reads=1.0):
    """answers: file -> triple or ABSTAIN. Uses eval_gguf's own scorer for 'exact'."""
    res = {f: dict(zip(F, a if a != ABSTAIN else NONE)) for f, a in answers.items()}
    s = ev.compute_accuracy(res, gt, files)["ALL"]
    # an abstention on a dated image must not count as correct even for partially labelled dates
    exact = sum(1 for f in dated if answers[f] != ABSTAIN and
                ev.compute_accuracy({f: res[f]}, gt, [f])["ALL"]["full_correct"] == 1)
    answered = [f for f in dated if answers[f] != ABSTAIN]
    fd = sum(1 for f in nodate if answers[f] not in (ABSTAIN, NONE))
    return dict(exact=100 * exact / len(dated), ym=100 * s["ym_correct"] / s["gt_has_date"], false_dates=fd,
                coverage=100 * len(answered) / len(dated),
                selective=100 * exact / max(1, len(answered)), reads=n_reads)


def plurality(ts):
    c = Counter(ts)
    top = max(c.values())
    return next(t for t in ts if c[t] == top), top


rows = []
single = [score({f: tri(reads[k][f]) for f in files}) for k in range(K)]
rows.append(("single frame (mean of 7)", {k: st.mean(s[k] for s in single) for k in single[0]}))
rows.append(("single frame (min..max exact)", dict(exact=min(s["exact"] for s in single), ym=max(s["exact"] for s in single),
                                                   false_dates=float("nan"), coverage=100, selective=float("nan"), reads=1)))
for n in (3, 5, 7):
    rows.append((f"vote {n}", score({f: plurality([tri(reads[k][f]) for k in range(n)])[0] for f in files}, n)))
for m, n in ((2, 2), (2, 3), (3, 3), (3, 5), (4, 5), (4, 7), (5, 7)):
    ans = {}
    for f in files:
        t, top = plurality([tri(reads[k][f]) for k in range(n)])
        ans[f] = t if top >= m else ABSTAIN
    rows.append((f"agree {m}/{n}", score(ans, n)))
ans, used = {}, []
for f in files:
    seen, a = Counter(), ABSTAIN
    for k in range(K):
        t = tri(reads[k][f])
        seen[t] += 1
        if seen[t] == 2:
            a = t
            break
    ans[f], _ = a, used.append(k + 1)
rows.append(("until 2 agree (max 7)", score(ans, st.mean(used))))

print(f"cap {CAP}: {len(dated)} dated + {len(nodate)} no-date images, {K} frames each")
print(f"{'policy':30s} {'exact %':>8s} {'coverage %':>10s} {'selective %':>11s} {'false dates':>12s} {'reads':>6s}")
for name, s in rows:
    if "min..max" in name:
        print(f"{name:30s} {s['exact']:5.1f}..{s['ym']:4.1f}")
        continue
    print(f"{name:30s} {s['exact']:8.1f} {s['coverage']:10.1f} {s['selective']:11.1f} "
          f"{s['false_dates']:9.1f}/{len(nodate)} {s['reads']:6.2f}")

# how often do frames agree, and is agreement informative?
agree_hist = Counter(plurality([tri(reads[k][f]) for k in range(K)])[1] for f in files)
print("\nlargest agreeing group among 7 frames -> images:", dict(sorted(agree_hist.items())))
ever = sum(1 for f in dated if any(ev.compute_accuracy({f: reads[k][f]}, gt, [f])["ALL"]["full_correct"] for k in range(K)))
print(f"oracle (any of 7 frames correct): {100 * ever / len(dated):.1f}%")
