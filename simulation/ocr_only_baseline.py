"""OCR-only baseline on the simulated headset frames, and what the OCR text adds to the VLM for free.

The on-device localiser already detects and recognises every text line of a frame (frames_m/f<k>_meta.json,
PP-OCRv5 mobile detector + Latin mobile recogniser). The baseline feeds those lines to the multilingual date
parser and expiry-keyword rule of qwen/ocr_baseline.py (unchanged) -- no VLM. Same 545 products, 7 frames,
same scorer as the VLM rows.

Also reported, because the OCR lines cost nothing extra once localisation has run:
  cross-check   answer the VLM's date only when the OCR rule parsed the same date from the same frame
  either        VLM date, but counted as 'confirmed' when OCR agrees (precision of confirmed answers)

  python ocr_only_baseline.py [cap]
"""
import importlib.util
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
CAP = sys.argv[1] if len(sys.argv) > 1 else "1024"
sys.argv = sys.argv[:1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ev = load("ev", "/home/user/qwen/eval_gguf.py")
ob = load("ob", "/home/user/qwen/ocr_baseline.py")
files = [e["filename"] for e in json.load(open(ev.TEST_JSON))]
gt = {e["filename"]: e for e in json.load(open(ev.GT_FILE))}
F = ("year", "month", "day")
K = 7
NONE = (None, None, None)
dated = [f for f in files if any(gt[f].get(k) is not None for k in F)]
nodate = [f for f in files if f not in dated]
metas = [json.load(open(HERE / "frames_m" / f"f{k}_meta.json")) for k in range(K)]
vlm = [json.load(open(HERE / "reads" / f"m_cap{CAP}_f{k}.json")) for k in range(K)]
tri = lambda r: tuple(r.get(k) for k in F)


def ocr_date(f, k):
    lines = sorted(((b[1] + b[3]) / 2, (b[0] + b[2]) / 2, t) for t, b, _ in metas[k][f]["all_lines"])
    chosen, _ = ob.select([t for _, _, t in lines])
    return tuple(chosen) if chosen else NONE


ocr = [{f: ocr_date(f, k) for f in files} for k in range(K)]


def is_correct(f, a):
    return ev.compute_accuracy({f: dict(zip(F, a))}, gt, [f])["ALL"]["full_correct"] == 1


def report(name, answers, reads=1.0):
    """answers: file -> triple, or None for abstain."""
    ok = sum(1 for f in dated if answers[f] is not None and is_correct(f, answers[f]))
    ans = [f for f in dated if answers[f] is not None]
    fd = sum(1 for f in nodate if answers[f] not in (None, NONE))
    return dict(name=name, exact=100 * ok / len(dated), answered=100 * len(ans) / len(dated),
                selective=100 * ok / max(1, len(ans)), false=fd, reads=reads)


def mean_rows(name, per_frame):
    return {k: (name if k == "name" else st.mean(r[k] for r in per_frame)) for k in per_frame[0]}


def plurality(ts):
    c = Counter(ts)
    top = max(c.values())
    return next(t for t in ts if c[t] == top)


def until2(src):
    ans, used = {}, []
    for f in files:
        seen, a, n = Counter(), plurality([src[k][f] for k in range(K)]), K
        for k in range(K):
            seen[src[k][f]] += 1
            if seen[src[k][f]] == 2:
                a, n = src[k][f], k + 1
                break
        ans[f] = a
        used.append(n)
    return ans, st.mean(used)


rows = []
# OCR only: "no date parsed" is an answer of no-date (the rule abstains on dated images -> wrong there)
rows.append(mean_rows("OCR + rules, 1 frame", [report("", ocr[k]) for k in range(K)]))
a, n = until2(ocr)
rows.append(report("OCR + rules, until 2 agree", a, n))
rows.append(report("OCR + rules, vote 7", {f: plurality([ocr[k][f] for k in range(K)]) for f in files}, 7))
any_ocr = sum(1 for f in dated if any(is_correct(f, ocr[k][f]) for k in range(K)))
# VLM
V = [{f: tri(vlm[k][f]) for f in files} for k in range(K)]
rows.append(mean_rows("VLM, 1 frame", [report("", V[k]) for k in range(K)]))
a, n = until2(V)
rows.append(report("VLM, until 2 agree", a, n))
# free cross-check: answer only when OCR parsed the same date from the same frame (else abstain)
rows.append(mean_rows("VLM confirmed by OCR, 1 frame",
                      [report("", {f: (V[k][f] if V[k][f] == ocr[k][f] and V[k][f] != NONE else None) for f in files})
                       for k in range(K)]))
# cascade: OCR-confirmed -> stop after one frame; otherwise keep reading until two VLM reads agree
ans, used = {}, []
for f in files:
    if V[0][f] == ocr[0][f] and V[0][f] != NONE:
        ans[f], n = V[0][f], 1
    else:
        seen, a, n = Counter(), plurality([V[k][f] for k in range(K)]), K
        for k in range(K):
            seen[V[k][f]] += 1
            if seen[V[k][f]] == 2 or (V[k][f] == ocr[k][f] and V[k][f] != NONE):
                a, n = V[k][f], k + 1
                break
        ans[f] = a
    used.append(n)
rows.append(report("VLM, stop when OCR confirms or 2 agree", ans, st.mean(used)))

print(f"cap {CAP}; {len(dated)} dated + {len(nodate)} no-date products, {K} frames, on-device detector")
print(f"{'method':40s} {'exact %':>8s} {'answered %':>10s} {'correct if ans %':>16s} {'false/25':>9s} {'frames':>7s}")
for r in rows:
    print(f"{r['name']:40s} {r['exact']:8.1f} {r['answered']:10.1f} {r['selective']:16.1f} {r['false']:9.1f} {r['reads']:7.2f}")
print(f"OCR oracle (any of 7 frames correct): {100 * any_ocr / len(dated):.1f}%")
parsed = st.mean(sum(1 for f in dated if ocr[k][f] != NONE) / len(dated) for k in range(K))
print(f"OCR parses some date on {100 * parsed:.1f}% of dated products per frame")
agree = st.mean(sum(1 for f in dated if V[k][f] == ocr[k][f] and V[k][f] != NONE) / len(dated) for k in range(K))
print(f"VLM and OCR agree on a date on {100 * agree:.1f}% of dated products per frame")
json.dump(rows, open(HERE / f"ocr_baseline_cap{CAP}.json", "w"), indent=1)
