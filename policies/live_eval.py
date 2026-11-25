"""Outcomes and wall-clock decision times of the live headset runs (liveloop on the Quest).

  python policies/live_eval.py <live tsv>...
For each run: correct / wrong / no date / abstain on the 111 dated clips, false dates on the 37 no-date sides,
coverage, precision, and the wall-clock time from the clip's first frame to the decision (median, p90), overall,
for correct answers and for abstentions; frames examined and VLM calls per clip.
"""
import collections
import statistics as st
import sys
from pathlib import Path

import numpy as np

import real_eval as re_


def tri(s):
    if s == "NONE":
        return re_.rp.NONE
    if s == "ABSTAIN":
        return "abstain"
    y, m, d = s.split("-")
    return tuple(None if "?" in v else int(v) for v in (y, m, d))


def q(v, p):
    return float(np.percentile(v, p)) if v else float("nan")


def main(paths):
    for path in paths:
        rows = [l.rstrip("\n").split("\t") for l in open(path) if l.strip() and not l.startswith("#") and not l.startswith("clip")]
        cnt, T, Tc, Ta, ex, calls = collections.Counter(), [], [], [], [], []
        for clip, rule, ans, ms, nexam, ncalls, last in rows:
            p, c = clip.split("_")[0], clip.split("_")[1]
            o = re_.outcome(tri(ans), None if c == "c" else tuple(re_.TRUTH[p]))
            cnt[o] += 1
            t = float(ms) / 1e3
            T.append(t)
            ex.append(int(nexam))
            calls.append(int(ncalls))
            if o == "correct":
                Tc.append(t)
            if o in ("abstain", "ok") and ans == "ABSTAIN":
                Ta.append(t)
        em = cnt["correct"] + cnt["wrong"] + cnt["false"]
        print(f"{Path(path).stem:28s} n={len(rows)} correct {cnt['correct']} wrong {cnt['wrong']} no-date {cnt['missed']} "
              f"abstain {cnt['abstain']} false {cnt['false']} | cov {100 * (cnt['correct'] + cnt['wrong']) / 111:.1f}% "
              f"prec {100 * cnt['correct'] / em if em else float('nan'):.1f}% | time median {st.median(T):.1f} p90 {q(T, 90):.1f} s; "
              f"to correct {st.median(Tc) if Tc else float('nan'):.1f} (p90 {q(Tc, 90):.1f}); to abstain {st.median(Ta) if Ta else float('nan'):.1f} "
              f"(p90 {q(Ta, 90):.1f}) | frames {st.mean(ex):.1f}, VLM calls {st.mean(calls):.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
