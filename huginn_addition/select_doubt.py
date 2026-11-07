"""Select prompts for the 'doubt as weak pull' experiment from the saved carry run (selection seed 0 there; the
experiment itself uses fresh init seeds). One primary answer position per prompt, one prompt per problem.
Uncertainty = top-1 minus top-2 of the digit-renormalised readout probabilities averaged over steps 48..64.
Cells: unsure (margin < 0.4) vs confident (margin > 0.9) x correct vs wrong (digit argmax at step 64 == truth and the
full-vocabulary argmax is a digit). Only answer positions >= 1 and digit mass > 0.5.
Usage: python -I huginn_addition/select_doubt.py results/<carry_run> results/doubt_selection.json
"""

from __future__ import annotations

import json
import sys

import numpy as np


def main(run, out):
    z = np.load(f"{run}/carry_traj.npz")
    probs = {p["uid"]: p for p in json.load(open(f"{run}/problems.json"))}
    lp, rows, fid = z["lp"], z["rows"], z["full_is_digit"]
    P = np.exp(lp[:, 48:65]).mean(1)
    mass = P.sum(1)
    Pn = P / mass[:, None]
    srt = np.sort(Pn, 1)
    margin = srt[:, -1] - srt[:, -2]  # margin of the time-averaged prediction (selection statistic)
    Ps = np.exp(lp[:, 48:65]); Ps = Ps / Ps.sum(-1, keepdims=True); o_ = np.sort(Ps, -1)
    margin_step = (o_[..., -1] - o_[..., -2]).mean(1)  # mean of per-step margins (alternative; saved, not used)
    truth = np.array([probs[int(u)]["digit_info"][int(j)]["digit"] for u, j in rows])
    correct = (lp[:, -1].argmax(-1) == truth) & fid[:, -1]
    ok = (rows[:, 1] >= 1) & (mass > 0.5)
    rng = np.random.default_rng(0)
    cells = {"unsure_correct": ok & (margin < 0.4) & correct, "unsure_wrong": ok & (margin < 0.4) & ~correct,
             "confident_correct": ok & (margin > 0.9) & correct, "confident_wrong": ok & (margin > 0.9) & ~correct}
    n_per = {"unsure_correct": 8, "unsure_wrong": 8, "confident_correct": 8, "confident_wrong": 8}
    used_pid = set()
    chosen = []
    for cell in ("confident_wrong", "unsure_correct", "unsure_wrong", "confident_correct"):  # scarcest first
        idx = rng.permutation(np.where(cells[cell])[0])
        k = 0
        for i in idx:
            u, j = int(rows[i, 0]), int(rows[i, 1])
            p = probs[u]
            if p["pid"] in used_pid:
                continue
            used_pid.add(p["pid"])
            q = {k_: v for k_, v in p.items() if k_ not in ("ids", "pred_pos")}
            q.update(ipos=j, cell=cell, sel_margin=float(margin[i]), sel_margin_perstep=float(margin_step[i]), sel_mass=float(mass[i]),
                     sel_top=int(Pn[i].argmax()), sel_second=int(np.argsort(Pn[i])[-2]), truth=int(truth[i]), multi=k < 4)
            chosen.append(q)
            k += 1
            if k == n_per[cell]:
                break
    json.dump(chosen, open(out, "w"), indent=1)
    for c in cells:
        sub = [q for q in chosen if q["cell"] == c]
        print(c, len(sub), "multi-start", sum(q["multi"] for q in sub), "margins", [round(q["sel_margin"], 2) for q in sub])


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2])
