"""Budget robustness on ARC-Challenge zero-shot multiple choice, baseline vs global relaxation (intervene_app.py `mcq`).
Prespecified before the runs.
Usage: python -I huginn_addition/analyze_mcq.py results/<base_shard_a>,results/<base_shard_b> results/<relax_a>,results/<relax_b> out_dir

Both runs: same 100 prompts, same seed-0 init, K = 256 loops; relax run uses s <- 0.15 s + 0.85 F(s) at every position.
Answer = argmax over the four letter tokens; p = probabilities renormalised over the four letters.
Per prompt and run:
  BLINKING (settled period-2 somewhere): some position with mean drift over the last 64 loops > 1e-4, median lag-2
    closure over the last 32 loops < 0.01, and drift ratio (last 32 / previous 32 loops) within 2%.
  STATIONARY: max-over-positions mean drift over the last 8 loops < 2e-5. Otherwise UNRESOLVED.
  Tail metrics (last 64 loops, every loop): answer FLIP = the letter argmax changes at least once; PARITY SWING
    S = max over letters |mean p(even loops) - mean p(odd loops)|; adjacent-budget total variation TV(255, 256).
  Practical-range metrics: number of answer changes between consecutive read loops 16..64; answer at budgets 16, 32,
    64, 128, 256; accuracy vs gold at those budgets; letter mass.
BASELINE questions (descriptive): prevalence of BLINKING (count, cluster-free binomial 95% CI); prompts with tail FLIP;
  S > 0.01 and S > 0.05; median TV(255, 256) among blinking vs stationary prompts.
PRIMARY PRACTICAL METRICS (budgets people actually use, 16..64 loops; both conditions): mean adjacent-budget TV over
  consecutive loops 16..64; answer disagreement between budgets 16 / 32 / 64 and 256; accuracy at 16, 32, 64 (chance
  25%); letter-mass distribution. Tail blinking establishes a dynamical phenomenon; useful-budget harm needs these.
RELAXATION hypotheses (motivated by theory, tested EMPIRICALLY; not guarantees: relaxation preserves the fixed-point
  set but can change basins and finite-budget trajectories; a real mode lam is stabilised iff -1.353 < lam < 1, and
  new prompts may carry other unstable modes):
  R1 no BLINKING prompts under relaxation (UNRESOLVED counts reported alongside);
  R2 for prompts STATIONARY in both runs, the loop-256 four-letter distributions agree within TV < 1e-3 and the answers
     are identical (operational tolerance, fixed now, not to be loosened; disagreements are split into
     endpoint disagreement vs either run unresolved);
  R3 every baseline-BLINKING prompt has S < 1e-3 under relaxation;
  R4 answers at loop 256 differ only on baseline non-stationary prompts.
  COST: OBSERVED ANSWER STABILITY = first read loop after which the answer never changes, CONFIRMED only if the answer is
  constant over at least the last 32 read loops (else censored, counted); median and 90th percentile over confirmed
  prompts, both conditions; accuracy at 16 and 32 in both. Sparse checkpoints (every 4 loops in 68..192) can miss
  intermediate changes.
Also reported: parity TV (mean TV between consecutive tail loops); S is a parity effect for sampling CONDITIONED on
  A-D (letter mass varies), not a full-vocabulary sampling assessment. With 100 prompts, zero observed events still
  allows ~3-4% prevalence; claims apply to this length-filtered four-option subset and this initialisation.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def load(runs):
    """Comma-separated run dirs (shards); keep completed rows, merge by uid (sorted). Every uid must be complete once."""
    parts = [np.load(f"{r}/mcq.npz", allow_pickle=True) for r in runs.split(",")]
    sched = parts[0]["sched"]
    assert all((p["sched"] == sched).all() for p in parts)
    rows = {}
    for p in parts:
        for i in np.where(p["done"])[0]:
            u = int(p["uid"][i])
            assert u not in rows, f"duplicate uid {u}"
            rows[u] = (p, i)
    uids = sorted(rows)
    Lm = max(p["drift"].shape[2] for p in parts)
    def stack(key, pad=None):
        arrs = []
        for u in uids:
            p, i = rows[u]
            a = p[key][i]
            if pad is not None and a.shape[-1] < Lm:
                a = np.concatenate([a, np.full(a.shape[:-1] + (Lm - a.shape[-1],), pad, a.dtype)], -1)
            arrs.append(a)
        return np.stack(arrs)
    return dict(uid=np.array(uids), sched=sched, ro_lp=stack("ro_lp"), ro_am=stack("ro_am"), ro_mass=stack("ro_mass"),
                drift=stack("drift", 0.0), closure=stack("closure", np.nan), lens=stack("lens"), gold=stack("gold"),
                done=np.ones(len(uids), bool))


def classify(z, i):
    L = int(z["lens"][i])
    d = z["drift"][i, :, :L]
    late64 = d[-64:].mean(0)
    cl = np.nanmedian(z["closure"][i, :, :L], 0)
    ratio = d[-32:].mean(0) / np.maximum(d[-64:-32].mean(0), 1e-30)
    p2 = (late64 > 1e-4) & (cl < 0.01) & (np.abs(ratio - 1) < 0.02)
    if p2.any():
        return "BLINKING", int(np.where(p2)[0][0])
    if d[-8:].mean(0).max() < 2e-5:
        return "STATIONARY", None
    return "UNRESOLVED", None


def metrics(z, i):
    sched = list(z["sched"])
    K = sched[-1]
    lp = z["ro_lp"][i]
    p = np.exp(lp); p = p / p.sum(-1, keepdims=True)
    ans = p.argmax(-1)
    tail = [sched.index(k) for k in range(K - 63, K + 1)]
    pt, at = p[tail], ans[tail]
    loops = np.array([sched[t] for t in tail])
    S = float(np.abs(pt[loops % 2 == 0].mean(0) - pt[loops % 2 == 1].mean(0)).max())
    tv = float(0.5 * np.abs(p[sched.index(K)] - p[sched.index(K - 1)]).sum())
    prac = [sched.index(k) for k in sched if 16 <= k <= 64]
    changes_16_64 = int((np.diff(ans[prac]) != 0).sum())
    bad = np.where(ans != ans[-1])[0]
    t_stab = sched[bad[-1] + 1] if len(bad) else sched[0]
    if (ans[-32:] != ans[-1]).any():
        t_stab = None  # unconfirmed (answer still changing within the last 32 read loops): censored
    prac = [sched.index(k) for k in sched if 16 <= k <= 64]
    adj_tv = float(np.mean(0.5 * np.abs(np.diff(p[prac], axis=0)).sum(-1)))
    parity_tv = float(np.mean(0.5 * np.abs(np.diff(pt, axis=0)).sum(-1)))
    at_b = {b: int(ans[sched.index(b)]) for b in (16, 32, 64, 128, 256) if b in sched}
    return dict(flip=bool((at != at[0]).any()), S=S, tv_255_256=tv, changes_16_64=changes_16_64, t_stab=t_stab,
                adj_tv_16_64=adj_tv, parity_tv_tail=parity_tv,
                answer_at=at_b, p256=p[sched.index(K)], mass256=float(z["ro_mass"][i, sched.index(K)]))


def main(run_b, run_r, out_dir):
    zb, zr = load(run_b), load(run_r)
    assert len(zb["uid"]) == len(zr["uid"]) and (zb["uid"] == zr["uid"]).all(), "conditions cover different prompts"
    gold = zb["gold"]
    n = len(gold)
    rows = []
    for i in range(n):
        cb, pb = classify(zb, i); cr_, pr_ = classify(zr, i)
        mb, mr = metrics(zb, i), metrics(zr, i)
        rows.append(dict(i=i, uid=int(zb["uid"][i]), gold=int(gold[i]), base_cls=cb, base_p2_pos=pb, relax_cls=cr_, base=mb, relax=mr,
                         tv_base_relax_256=float(0.5 * np.abs(mb["p256"] - mr["p256"]).sum())))
    def ci(k, n_):
        from math import sqrt
        p_ = k / n_; z_ = 1.96; den = 1 + z_ ** 2 / n_
        c = (p_ + z_ ** 2 / (2 * n_)) / den; hw = z_ * sqrt(p_ * (1 - p_) / n_ + z_ ** 2 / (4 * n_ ** 2)) / den
        return [round(c - hw, 3), round(c + hw, 3)]
    nb = sum(r["base_cls"] == "BLINKING" for r in rows)
    acc = lambda key, b: float(np.mean([r[key]["answer_at"][b] == r["gold"] for r in rows]))
    base_summary = dict(
        classes={c: sum(r["base_cls"] == c for r in rows) for c in ("BLINKING", "STATIONARY", "UNRESOLVED")},
        blinking_prevalence=dict(k=nb, n=n, wilson95=ci(nb, n)),
        tail_flip=sum(r["base"]["flip"] for r in rows), S_gt_0p01=sum(r["base"]["S"] > 0.01 for r in rows),
        S_gt_0p05=sum(r["base"]["S"] > 0.05 for r in rows),
        median_tv_255_256={c: (float(np.median([r["base"]["tv_255_256"] for r in rows if r["base_cls"] == c])) if any(r["base_cls"] == c for r in rows) else None)
                           for c in ("BLINKING", "STATIONARY", "UNRESOLVED")},
        max_S_blinking=max([r["base"]["S"] for r in rows if r["base_cls"] == "BLINKING"], default=None),
        prompts_with_changes_16_64=sum(r["base"]["changes_16_64"] > 0 for r in rows),
        accuracy={b: acc("base", b) for b in (16, 32, 64, 128, 256)}, median_letter_mass=float(np.median([r["base"]["mass256"] for r in rows])),
        letter_mass_q10_q90=[float(np.quantile([r["base"]["mass256"] for r in rows], q)) for q in (.1, .9)])
    def practical(key):
        return dict(mean_adj_tv_16_64=float(np.mean([r[key]["adj_tv_16_64"] for r in rows])),
                    disagree_vs_256={b: int(sum(r[key]["answer_at"][b] != r[key]["answer_at"][256] for r in rows)) for b in (16, 32, 64)},
                    accuracy={b: acc(key, b) for b in (16, 32, 64)}, mean_parity_tv_tail=float(np.mean([r[key]["parity_tv_tail"] for r in rows])))
    primary_practical = dict(base=practical("base"), relax=practical("relax"))
    stat_both = [r for r in rows if r["base_cls"] == "STATIONARY" and r["relax_cls"] == "STATIONARY"]
    blink_b = [r for r in rows if r["base_cls"] == "BLINKING"]
    R1 = sum(r["relax_cls"] == "BLINKING" for r in rows) == 0
    R2_fail = [r["uid"] for r in stat_both if not (r["tv_base_relax_256"] < 1e-3 and r["base"]["answer_at"][256] == r["relax"]["answer_at"][256])]
    R2_unresolved_pairs = sum(1 for r in rows if "UNRESOLVED" in (r["base_cls"], r["relax_cls"]))
    R3_fail = [r["uid"] for r in blink_b if not r["relax"]["S"] < 1e-3]
    R4_diff = [r["uid"] for r in rows if r["base"]["answer_at"][256] != r["relax"]["answer_at"][256]]
    R4_ok = all(next(x for x in rows if x["uid"] == u)["base_cls"] != "STATIONARY" for u in R4_diff)
    relax_summary = dict(
        classes={c: sum(r["relax_cls"] == c for r in rows) for c in ("BLINKING", "STATIONARY", "UNRESOLVED")},
        R1_no_blinking=R1, R2_n_stationary_both=len(stat_both), R2_failures=R2_fail, R2_prompts_unresolved_in_either=R2_unresolved_pairs, R3_n_baseline_blinking=len(blink_b), R3_failures=R3_fail,
        R4_answer_changes_at_256=R4_diff, R4_ok=R4_ok,
        tail_flip=sum(r["relax"]["flip"] for r in rows), S_gt_0p01=sum(r["relax"]["S"] > 0.01 for r in rows),
        accuracy={b: acc("relax", b) for b in (16, 32, 64, 128, 256)},
        t_stab_confirmed_median={k: float(np.median([r[k]["t_stab"] for r in rows if r[k]["t_stab"] is not None])) for k in ("base", "relax")},
        t_stab_confirmed_q90={k: float(np.quantile([r[k]["t_stab"] for r in rows if r[k]["t_stab"] is not None], .9)) for k in ("base", "relax")},
        t_stab_censored={k: sum(r[k]["t_stab"] is None for r in rows) for k in ("base", "relax")},
        median_tv_base_relax_256_stationary=float(np.median([r["tv_base_relax_256"] for r in stat_both])) if stat_both else None)
    out = dict(primary_practical=primary_practical, baseline=base_summary, relaxation=relax_summary)
    for r in rows:
        for k in ("base", "relax"):
            r[k]["p256"] = r[k]["p256"].round(4).tolist()
    import os
    os.makedirs(out_dir, exist_ok=True)
    json.dump(dict(summary=out, rows=rows), open(f"{out_dir}/analysis_summary.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
