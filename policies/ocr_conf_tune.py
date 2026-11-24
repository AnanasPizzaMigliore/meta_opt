"""Tuned OCR-only baseline, step 1 (development data only): choose an OCR-only stopping rule that uses the
recogniser's confidence, on the SIMULATED frames of the 545 test photos (7 frames each, PP-OCRv5 mobile detector,
per-line confidence stored in frames_m/f<k>_meta.json). The 37 held-out real products are not touched here.

Rule OCR(k, tau): frames in order (at most 7); on each frame the date rules choose a date from the recognised lines
(ocr_baseline.select, unchanged) and its confidence is the recogniser's score of the line it came from; a date
counts only if its confidence >= tau; answer the first date counted k times, otherwise abstain. It never answers
"no date".

Pre-registered selection (written before any held-out evaluation): among k in {1, 2, 3} and tau in TAUS, keep the
settings whose development precision is at least that of the OCR cross-check (VLM confirmed by OCR, same frames);
choose the one with the most correct dates; ties -> fewer frames read, then higher tau.

  llm-env python ocr_conf_tune.py      -> writes ocr_conf_choice.json
"""
import collections
import datetime
import importlib.util
import json
from pathlib import Path

V = Path("/home/user/qwen/quest_sim/voting")
K, TAUS = (1, 2, 3), (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99)
F = ("year", "month", "day")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ob = load("ob", "/home/user/qwen/ocr_baseline.py")


def select_conf(lines):
    """lines: [(text, conf)] in reading order -> (date triple or None, conf of its line). Same choice as ob.select."""
    cands = []
    for i, (line, conf) in enumerate(lines):
        near_kw = ob.has_kw(line, ob.EXPIRY) or (i > 0 and ob.has_kw(lines[i - 1][0], ob.EXPIRY))
        for dt in ob.parse_dates(line, compact=near_kw):
            score = (2 if near_kw else 0) - (2 if ob.has_kw(line, ob.PRODUCTION) else 0)
            cands.append((score, dt, conf))
    if not cands:
        return None, 0.0
    best = max(cands, key=lambda c: (c[0], c[1][0], c[1][1], c[1][2] or 0))
    return tuple(best[1]), best[2]


def norm(t):
    y, m, d = t
    return (y + 2000 if isinstance(y, int) and y < 100 else y, m, d)


def match(a, t):
    return a[0] == t[0] and a[1] == t[1] and (t[2] is None or a[2] == t[2])


def rule(frames, k, tau):
    seen = collections.Counter()
    for i, (dt, conf) in enumerate(frames):
        if dt is not None and conf >= tau:
            seen[dt] += 1
            if seen[dt] == k:
                return dt, i + 1
    return None, len(frames)


def outcome(a, truth):
    if truth is None:
        return "ok" if a is None else "false"
    if a is None:
        return "abstain"
    return "correct" if match(a, truth) else "wrong"


def stats(outs):
    c = collections.Counter(outs)
    em = c["correct"] + c["wrong"] + c["false"]
    return dict(correct=c["correct"], wrong=c["wrong"], false=c["false"], abstain=c["abstain"],
                prec=c["correct"] / em if em else float("nan"))


def main():
    ev = load("ev", "/home/user/qwen/eval_gguf.py")
    files = [e["filename"] for e in json.load(open(ev.TEST_JSON))]
    gt = {e["filename"]: e for e in json.load(open(ev.GT_FILE))}
    truth = {f: (tuple(gt[f].get(x) for x in F) if any(gt[f].get(x) is not None for x in F) else None) for f in files}
    metas = [json.load(open(V / "frames_m" / f"f{k}_meta.json")) for k in range(7)]
    reads = [json.load(open(V / "reads" / f"m_cap1024_f{k}.json")) for k in range(7)]
    ocr = {}
    for f in files:
        fr = []
        for k in range(7):
            lines = sorted(((b[1] + b[3]) / 2, (b[0] + b[2]) / 2, t, s) for t, b, s in metas[k][f]["all_lines"])
            dt, conf = select_conf([(t, s) for *_, t, s in lines])
            fr.append((norm(dt) if dt else None, conf))
        ocr[f] = fr
    # reference: the OCR cross-check on the same development frames
    xc = []
    for f in files:
        a = None
        for k in range(7):
            v = tuple(reads[k][f].get(x) for x in F) if isinstance(reads[k][f], dict) else (None, None, None)
            o = ocr[f][k][0]
            if o is not None and v != (None, None, None) and norm(v) == o:
                a = o
                break
        xc.append(outcome(a, truth[f]))
    ref = stats(xc)
    print(f"dev reference, OCR cross-check: {ref}")
    grid = []
    for k in K:
        for tau in TAUS:
            res = [rule(ocr[f], k, tau) for f in files]
            st = stats([outcome(a, truth[f]) for (a, _), f in zip(res, files)])
            st.update(k=k, tau=tau, frames=sum(n for _, n in res) / len(res))
            grid.append(st)
            print(f"OCR(k={k}, tau={tau:.2f}): {st}")
    ok = [g for g in grid if g["prec"] >= ref["prec"]]
    choice = max(ok, key=lambda g: (g["correct"], -g["frames"], g["tau"])) if ok else None
    out = dict(chosen_at=datetime.datetime.now().isoformat(timespec="seconds"), rule=__doc__.strip().splitlines()[0],
               dev_reference=ref, choice=choice, grid=grid)
    json.dump(out, open(Path(__file__).parent / "ocr_conf_choice.json", "w"), indent=1)
    print("CHOICE:", choice)


if __name__ == "__main__":
    main()
