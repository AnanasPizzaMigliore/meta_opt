"""The multi-frame reading policy, fixed on 2026-11-17 BEFORE any new real-video data is evaluated.

Do not change these rules after looking at the 40-product capture; report them as they are.

History: v1 (GATE + AGREE + BUDGET) was fixed in the morning of 2026-11-17 and evaluated on pilot 1 and pilot 2.
VALID was added on 2026-11-17 after pilot 2 (P2's "02.09.2027" read as day 32 in agreeing frames) and before the
40-product capture; on pilots 1-2 it is therefore post hoc, on the 40 products it is fixed in advance.
v1 is kept as reading_policy_v1_2026-11-17.py.

  GATE   a frame is read by the VLM only if the on-device detector found a date-like or digit line in it
         (meta level "date" or "digits"); otherwise the frame is skipped (localisation cost only).
  AGREE  read gated frames in time order; answer the first (year, month, day) triple seen twice.
         "no date" (all null) is an answer too and can agree with itself.
  BUDGET at most 7 localised frames; if nothing has agreed by then, abstain.
  VALID  a read whose date cannot exist on a calendar (month outside 1-12, or day outside the month's length,
         31 when the month is unknown) is discarded: it costs a read but never counts towards agreement.

Outcome per product:
  dated product:    correct | wrong (a different date) | missed (agreed on "no date") | abstain
  no-date product:  false_date (agreed on any date) | ok (agreed on "no date" or abstained)

  python reading_policy.py sim        simulated frames of the 545-image test set (on-device detector, cap 1024)
  python reading_policy.py real [set] real Quest clips processed by process_clip.sh (needs <set>/truth.json; default pilot)
"""
import calendar
import collections
import json
import sys
from pathlib import Path

NONE = (None, None, None)
BUDGET = 7
GATE_LEVELS = ("date", "digits")


def tri(r):
    j = r.get("json") if "json" in r else r
    j = j or {}
    return (j.get("year"), j.get("month"), j.get("day"))


def valid(t):
    """True unless the (year, month, day) triple names a date that cannot exist; unknown fields are allowed."""
    y, m, d = t
    if m is not None and not 1 <= m <= 12:
        return False
    if d is not None:
        last = calendar.monthrange(y if isinstance(y, int) and y > 0 else 2000, m)[1] if m is not None else 31
        if not 1 <= d <= last:
            return False
    return True


def run(frames, gated=True, budget=BUDGET, check_valid=True):
    """frames: [(detector_level, read_triple)] in time order -> (answer or 'abstain', localised, reads)."""
    seen = collections.Counter()
    reads = 0
    for i, (level, t) in enumerate(frames[:budget]):
        if gated and level not in GATE_LEVELS:
            continue
        reads += 1
        if check_valid and not valid(t):
            continue
        seen[t] += 1
        if seen[t] == 2:
            return t, i + 1, reads
    return "abstain", min(len(frames), budget), reads


def outcome(answer, truth, match):
    if truth is None:                               # no-date product
        return "ok" if answer in ("abstain", NONE) else "false_date"
    if answer == "abstain":
        return "abstain"
    if answer == NONE:
        return "missed"
    return "correct" if match(answer, truth) else "wrong"


def exact(a, t):
    return all(x == y for x, y in zip(a, t))


def table(name, outs, cost):
    c = collections.Counter(outs)
    dated = sum(v for k, v in c.items() if k != "ok" and k != "false_date")
    neg = c["ok"] + c["false_date"]
    emitted = c["correct"] + c["wrong"] + c["false_date"]
    prec = f"{100 * c['correct'] / emitted:.1f}%" if emitted else "-"
    print(f"{name:34s} dated {dated:3d}: correct {c['correct']:3d} wrong {c['wrong']:3d} missed {c['missed']:3d} "
          f"abstain {c['abstain']:3d} | no-date {neg:2d}: false {c['false_date']:2d} | date precision {prec} | {cost}")


def sim():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ev", "/home/user/qwen/eval_gguf.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    V = Path("/home/user/qwen/quest_sim/voting")
    files = [e["filename"] for e in json.load(open(ev.TEST_JSON))]
    gt = {e["filename"]: e for e in json.load(open(ev.GT_FILE))}
    metas = [json.load(open(V / f"frames_m/f{k}_meta.json")) for k in range(7)]
    reads = [json.load(open(V / f"reads/m_cap1024_f{k}.json")) for k in range(7)]

    def match(a, f):
        res = {f: dict(zip(("year", "month", "day"), a))}
        return ev.compute_accuracy(res, gt, [f])["ALL"]["full_correct"] == 1

    for gated, cv in ((False, False), (True, False), (True, True)):
        outs, loc, rd = [], [], []
        for f in files:
            frames = [(metas[k][f]["level"], tri(reads[k][f])) for k in range(7)]
            a, n, r = run(frames, gated, check_valid=cv)
            truth = None if all(gt[f].get(k) is None for k in ("year", "month", "day")) else f
            outs.append(outcome(a, truth, lambda a_, t_: match(a_, t_)))
            loc.append(n)
            rd.append(r)
        table(f"sim 545, until-2, {'GATED' if gated else 'ungated'}{' +VALID' if cv else ''}", outs,
              f"mean localised {sum(loc) / len(loc):.2f}, VLM reads {sum(rd) / len(rd):.2f}")
    # recurring wrong answers: which triples are emitted for no-date images and as wrong dates?
    wrong = collections.Counter()
    for k in range(7):
        for f in files:
            t = tri(reads[k][f])
            nod = all(gt[f].get(x) is None for x in ("year", "month", "day"))
            if t != NONE and (nod or not match(t, f)):
                wrong[t] += 1
    print("most frequent wrong/invented single-frame answers (sim, 7 x 545 frames):", wrong.most_common(6))


def real(set_="pilot"):
    """truth.json: {"p001": [2027, 7, null], "p002": "unknown", ...}; clips without a date side are clip c."""
    L = Path("/home/user/qwen/phone_bench/questcap_captures") / set_ / "loc"
    truth = json.load(open(L.parent / "truth.json"))
    for gated, cv in ((False, False), (True, False), (True, True)):
        outs, rows = [], []
        for p, t in sorted(truth.items()):
            for c in "abdc":
                d = L / f"{p}_{c}_rot0"
                if not (d / "reads_cap1024.json").exists():
                    continue
                m, r = json.load(open(d / "meta.json")), json.load(open(d / "reads_cap1024.json"))
                frames = [(m[f]["level"], tri(r[f])) for f in sorted(r)]
                a, n, nr = run(frames, gated, check_valid=cv)
                if c == "c":
                    o = outcome(a, None, exact)
                elif t == "unknown":
                    o = "abstain" if a == "abstain" else ("missed" if a == NONE else "wrong")  # unknown truth: any date counts against us
                else:
                    o = outcome(a, tuple(t), exact)
                outs.append(o)
                rows.append(f"{p}{c}:{o}")
        table(f"{set_}, until-2, {'GATED' if gated else 'ungated'}{' +VALID' if cv else ''}", outs, " ".join(rows))


if __name__ == "__main__":
    {"sim": sim, "real": real}[sys.argv[1]](*sys.argv[2:])
