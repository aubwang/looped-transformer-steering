"""Carry-chain procedure experiment, stage 1 behaviour (intervene_app.py `mcq`, mode carry_chain). Prespecified.
Usage: python -I huginn_addition/analyze_carry_chain.py results/<carry_chain_run>
Prompts: results/carry_chain_prompts.json (make_carry_chain_prompts.py): chain length k = 0..4 propagate columns between
the target column and the carry-deciding column; 30 prompts per k; everything else matched (54 tokens, same answer index).
Readout every loop 1..96 over bare digits "0".."9" at the target position.
Definitions: correct at loop t = digit argmax == gold. ATTAINMENT loop = first loop after which the argmax equals gold
through loop 96; prompts not correct at 96 are CENSORED (set to 97 for rank statistics).
PRIMARY (H1, procedure signature): Spearman(k, attainment loop with censoring at 97) > 0; permutation p-value (10000
  label permutations; undefined if the statistic is nonfinite). This composite mixes correctness and latency, so the
  latency among prompts correct at 96 (Spearman k vs attainment) and the censoring counts are always reported alongside;
  per-k median attainment treats censored as +inf (simple median, not Kaplan-Meier).
  Carry-flip 1/9 is a uniform-wrong-digit reference, not a calibrated baseline. A positive result is a screening
  association (k also moves the decider position and the fixed/random column composition; demo set is coupled to the
  column-5 type identically across k).
SECONDARY: H2 accuracy at 96 decreases with k (Spearman over prompts of k vs correct); H3 accuracy gain from loop 16 to
  96 per k (does extra recurrence help longer chains?); H4 error type: among wrong final answers, the fraction equal to
  the carry-flipped digit (digit_no_carry if carry_in = 1 else digit_with_carry), chance ~1/9, per k; and split by
  decider (G vs K: is the carry missed, or invented?).
Descriptive: accuracy curves by k at loops 4, 8, 12, 16, 24, 32, 48, 64, 96; digit mass.
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr


def main(run):
    z = np.load(f"{run}/mcq.npz", allow_pickle=True)
    P = {p["uid"]: p for p in json.load(open(f"{run}/problems.json"))}
    assert z["done"].all(), "incomplete"
    sched = list(z["sched"]); K = sched[-1]
    rows = []
    for i, u in enumerate(z["uid"]):
        p = P[int(u)]
        am = z["ro_lp"][i].argmax(-1)
        ok = am == p["gold"]
        att = None
        if ok[-1]:
            bad = np.where(~ok)[0]
            att = sched[bad[-1] + 1] if len(bad) else sched[0]
        fin = int(am[-1])
        flip = p["digit_no_carry"] if p["carry_in"] else p["digit_with_carry"]
        rows.append(dict(k=p["k"], decider=p["decider"], att=att, correct=bool(ok[-1]), final=fin, carry_flip_error=bool(fin == flip and not ok[-1]),
                         acc_at={t: bool(ok[sched.index(t)]) for t in (4, 8, 12, 16, 24, 32, 48, 64, 96)}, mass=float(z["ro_mass"][i, -1])))
    k = np.array([r["k"] for r in rows]); att = np.array([r["att"] if r["att"] is not None else K + 1 for r in rows], float)
    rho = spearmanr(k, att)[0]
    rng = np.random.default_rng(0)
    if np.isfinite(rho):
        perm = np.array([spearmanr(rng.permutation(k), att)[0] for _ in range(10000)])
        p_val = float((np.sum(perm >= rho) + 1) / (len(perm) + 1))
    else:
        p_val = None  # undefined statistic (e.g. constant attainment): test undefined
    okm = np.array([r["att"] is not None for r in rows])
    rho_lat = spearmanr(k[okm], att[okm])[0] if okm.sum() > 5 else float("nan")
    by_k = {}
    for kk in range(5):
        rs = [r for r in rows if r["k"] == kk]
        a_ = sorted([r["att"] if r["att"] is not None else np.inf for r in rs])
        med = a_[len(a_) // 2] if len(a_) % 2 else 0.5 * (a_[len(a_) // 2 - 1] + a_[len(a_) // 2])
        wrong = [r for r in rs if not r["correct"]]
        by_k[kk] = dict(n=len(rs), acc_curve={t: float(np.mean([r["acc_at"][t] for r in rs])) for t in (4, 8, 12, 16, 24, 32, 48, 64, 96)},
                        median_attainment_censored_inf=None if not np.isfinite(med) else float(med), censored=sum(r["att"] is None for r in rs),
                        gain_16_to_96=float(np.mean([r["acc_at"][96] for r in rs]) - np.mean([r["acc_at"][16] for r in rs])),
                        n_wrong=len(wrong), carry_flip_errors=sum(r["carry_flip_error"] for r in wrong),
                        carry_flip_errors_by_decider={d: (sum(r["carry_flip_error"] for r in wrong if r["decider"] == d), sum(1 for r in wrong if r["decider"] == d)) for d in "GK"},
                        median_digit_mass=float(np.median([r["mass"] for r in rs])))
    out = dict(PRIMARY_spearman_k_attainment=dict(rho=None if not np.isfinite(rho) else float(rho), p_perm_one_sided=p_val, n=len(rows),
                                                  censored=int((~okm).sum())),
               latency_among_correct_at_96=dict(spearman_k_attainment=None if not np.isfinite(rho_lat) else float(rho_lat), n=int(okm.sum())),
               H2_spearman_k_correct=float(spearmanr(k, [r["correct"] for r in rows])[0]), by_k=by_k)
    json.dump(dict(summary=out, rows=rows), open(f"{run}/analysis_summary.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
