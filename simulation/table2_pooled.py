"""Table 2 of the NOSSDAV draft, recomputed with ONE abstention rule and pooled counts (revision, 2026-11-17).

Rules: an answer is a (year, month, day) triple; "no date" is an answer. Multi-frame policies that end without an answer
ABSTAIN (no plurality fallback). OCR + rules: "no date parsed" is treated as an abstention. Single-frame rows pool all
7 x 545 frame-level decisions (no averaging). Counts are given raw.

  python table2_pooled.py [cap]
"""
import collections
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
CAP = sys.argv[1] if len(sys.argv) > 1 else "1024"
src = (HERE / "ocr_only_baseline.py").read_text()
# reuse the baseline's loading and OCR parsing (everything before the report section)
pre = src[:src.index("def report(")]
g = {"__file__": str(HERE / "ocr_only_baseline.py"), "__name__": "t2"}
sys.argv = [sys.argv[0], CAP]
exec(compile(pre, "ocr_only_baseline_pre", "exec"), g)
files, dated, nodate, K, ocr, vlm, is_correct, tri, NONE = (g[k] for k in
    ("files", "dated", "nodate", "K", "ocr", "vlm", "is_correct", "tri", "NONE"))
V = [{f: tri(vlm[k][f]) for f in files} for k in range(K)]
O = [{f: (None if ocr[k][f] == NONE else ocr[k][f]) for f in files} for k in range(K)]   # no date parsed -> abstain


def until2(src_, f, extra_stop=None):
    seen = collections.Counter()
    for k in range(K):
        if extra_stop and extra_stop(k, f):
            return V[k][f], k + 1
        a = src_[k][f]
        if a is None:
            continue
        seen[a] += 1
        if seen[a] == 2:
            return a, k + 1
    return None, K


def row(name, decisions, frames):
    """decisions: list of (file, answer or None)."""
    d = [(f, a) for f, a in decisions if f in dated_set]
    n = [(f, a) for f, a in decisions if f in nodate_set]
    ok = sum(1 for f, a in d if a is not None and a != NONE and is_correct(f, a))
    ans = sum(1 for f, a in d if a is not None)
    false = sum(1 for f, a in n if a not in (None, NONE))
    print(f"{name:40s} exact {ok:4d}/{len(d):4d} = {100 * ok / len(d):5.1f}%  answered {100 * ans / len(d):5.1f}%  "
          f"right-if-answered {100 * ok / max(1, ans):5.1f}%  false {false:3d}/{len(n):3d}  frames {frames:.2f}")


dated_set, nodate_set = set(dated), set(nodate)
print(f"cap {CAP}: {len(dated)} dated + {len(nodate)} no-date products, {K} frames each; single-frame rows pool all frames")
row("OCR + rules, 1 frame (pooled)", [(f, O[k][f]) for k in range(K) for f in files], 1)
row("VLM, 1 frame (pooled)", [(f, V[k][f]) for k in range(K) for f in files], 1)
row("VLM confirmed by OCR, 1 frame (pooled)",
    [(f, V[k][f] if (O[k][f] is not None and V[k][f] == O[k][f]) else None) for k in range(K) for f in files], 1)
for name, fn in (("VLM, until 2 agree", lambda f: until2(V, f)),
                 ("VLM, until OCR confirms or 2 agree", lambda f: until2(V, f, lambda k, f_: O[k][f_] is not None and V[k][f_] == O[k][f_]))):
    res = {f: fn(f) for f in files}
    row(name, [(f, a) for f, (a, _) in res.items()], sum(n for _, n in res.values()) / len(files))
