"""Pointer-chasing behavioural SCREEN (intervene_app.py `mcq`, mode ptr_screen). Exploratory, to choose the depth range
for a prespecified procedure experiment.
Usage: python -I huginn_addition/analyze_pointer_screen.py results/<ptr_screen_run>
Per depth n (20 prompts each): accuracy (argmax over " 0".." 9" == f^n(x)) at loops 4, 8, 16, 32, 64, 128 (chance 10%);
CORRECT-ATTAINMENT loop = first loop after which the argmax equals the answer through the last loop (censored if not);
digit mass; COUNT of prompts whose argmax at 128 equals an intermediate f^j(x), j < n (stopped early) or f^{n+1}(x)
(overshoot); a match with an intermediate is compatible with early stopping but has chance rate ~n/10. Attainment
quantiles are conditional on final correctness (censored excluded).
"""
import json
import sys

import numpy as np


def main(run):
    z = np.load(f"{run}/mcq.npz", allow_pickle=True)
    P = {p["uid"]: p for p in json.load(open(f"{run}/problems.json"))}
    assert z["done"].all()
    sched = list(z["sched"])
    out = {}
    for n in range(1, 7):
        idx = [i for i, u in enumerate(z["uid"]) if P[int(u)]["depth"] == n]
        acc = {b: float(np.mean([z["ro_lp"][i, sched.index(b)].argmax() == z["gold"][i] for i in idx])) for b in (4, 8, 16, 32, 64, 96)}
        att = []
        for i in idx:
            ok = z["ro_lp"][i].argmax(-1) == z["gold"][i]
            if not ok[-1]:
                att.append(None); continue
            bad = np.where(~ok)[0]
            att.append(sched[bad[-1] + 1] if len(bad) else sched[0])
        fin = [int(z["ro_lp"][i, -1].argmax()) for i in idx]
        inter = [P[int(z["uid"][i])]["intermediates"] for i in idx]
        fvec = [P[int(z["uid"][i])]["f"] for i in idx]
        early = sum(fi in it[:-1] for fi, it in zip(fin, inter))
        over = sum(fi == fv[it[-1]] for fi, it, fv in zip(fin, inter, fvec))
        a_ok = [a for a in att if a is not None]
        out[n] = dict(n=len(idx), acc=acc, attain_median=float(np.median(a_ok)) if a_ok else None,
                      attain_q75=float(np.quantile(a_ok, .75)) if a_ok else None, censored=sum(a is None for a in att),
                      final_is_intermediate=early, final_is_overshoot=over,
                      median_digit_mass=float(np.median([z["ro_mass"][i, -1] for i in idx])))
    json.dump(out, open(f"{run}/screen_summary.json", "w"), indent=1)
    for n, v in out.items():
        print(n, v)


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
