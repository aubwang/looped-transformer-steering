"""Is uncertainty a weak pull? Analysis of intervene_app.py `doubt`.
Usage: python -I huginn_addition/analyze_doubt.py results/<doubt_run>

Per prompt (27; one primary teacher-forced answer position p): selection margin (seed-0 run), fresh-seed and fixed-point
margins; multi-start agreement (answer argmax across 8 fresh seeds; terminal p-state spread); end-of-run motion at p;
recovery after small pushes at the fixed point x* (relative 1e-3): per-loop contraction of the distance to the control
trajectory at p for random pushes at p (mean of 2), the gap-shrinking push, and prefix pushes (max over positions <= p),
fitted over loops 2..T where the distance is > 20x the duplicate-control floor; readout recovery = per-loop decay of
|gap_push - gap_control| (fixed top-2 pair) over loops 2..24; leading Ritz |eigenvalue| of the answer block J_pp.
Association with uncertainty: Spearman rho vs selection margin across prompts with a permutation p-value, and medians by
unsure vs confident. Exploratory (n = 27, one prompt per problem).
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_gradients import CAT, INK2  # noqa: E402

RNG = np.random.default_rng(0)


def margin(lp):
    P = np.exp(lp)
    P = P / P.sum(-1, keepdims=True)
    o = np.sort(P, -1)
    return o[..., -1] - o[..., -2]


def rate(d, floor=None, t0=2, tmax=None, k=10):
    """Per-loop factor from a log-linear fit over loops >= t0 where d > k * floor; floor = plateau (median of the last 16
    loops) unless given. (The duplicate control is bit-identical to the control, so it cannot set the floor.)"""
    d = np.asarray(d, float)
    floor = float(np.median(d[-16:])) if floor is None else floor
    t = np.arange(len(d))
    tmax = len(d) if tmax is None else tmax
    ok = (t >= t0) & (t < tmax) & (d > k * max(floor, 1e-12))
    if ok.sum() < 4:
        return np.nan
    return float(np.exp(np.polyfit(t[ok], np.log(d[ok]), 1)[0]))


def avg_rank(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); r[o] = np.arange(len(x), dtype=float)
    for v in np.unique(x):  # average ranks for ties
        k = x == v
        r[k] = r[k].mean()
    return r


def spearman(x, y, B=20000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rx, ry = avg_rank(x), avg_rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    b = sum(abs(np.corrcoef(rx, RNG.permutation(ry))[0, 1]) >= abs(rho) for _ in range(B))
    return {"rho": round(rho, 3), "p_perm": round(float((b + 1) / (B + 1)), 5), "n": int(ok.sum())}


def main(run):
    out = f"{run}/analysis"
    os.makedirs(out, exist_ok=True)
    z = np.load(f"{run}/doubt.npz", allow_pickle=True)
    probs = json.load(open(f"{run}/problems.json"))
    n = len(probs)
    cell = z["cell"]
    unsure = np.array([c.startswith("unsure") for c in cell])
    correct = np.array([c.endswith("_correct") for c in cell])
    pr, sd = z["dyn__prompt"], z["dyn__seed"]
    fpi = {int(i): j for j, i in enumerate(z["fp_prompt"])}
    kinds = list(z["push_kinds"])
    nk = len(kinds)
    D, Dpre, LP = z["push_dist_p"], z["push_dist_pre"], z["push_lp"]
    ge = int(z["push_gap_every"])
    rows = []
    for i in range(n):
        j = fpi[i]
        r1 = np.where((pr == i) & (sd == 1))[0][0]
        allr = np.where(pr == i)[0]
        rec = {"i": i, "uid": int(z["uid"][i]), "cell": str(cell[i]), "problem": f"{probs[i]['a']} + {probs[i]['b']}",
               "digit_pos": int(probs[i]["ipos"]), "carry_in": int(probs[i]["digit_info"][probs[i]["ipos"]]["carry_in"]),
               "n_digits": int(probs[i]["n_digits"]), "sel_margin": probs[i]["sel_margin"],
               "margin_seed1_terminal": float(margin(z["dyn__ro_lp"][r1][-1])), "margin_fixed_point": float(margin(z["fp_lp"][j])),
               "fp_valid": bool(z["fp_valid"][j]), "fp_resid_prefix": float(z["fp_resid_prefix"][j])}
        # multi-start
        ams = [int(np.exp(z["dyn__ro_lp"][r][-1]).argmax()) for r in allr]
        Tst = np.stack([z["dyn__ck_state_p"][r][-1] for r in allr])
        sp = max(np.linalg.norm(Tst[a] - Tst[b]) / np.linalg.norm(Tst[a]) for a in range(len(allr)) for b in range(len(allr))) if len(allr) > 1 else np.nan
        rec.update(n_seeds=len(allr), n_distinct_answers=len(set(ams)), answers=ams, terminal_state_spread_p=float(sp),
                   end_drift_p_seed1=float(z["dyn__drift_p"][r1][-16:].mean()), steps_seed1=int(z["dyn__steps_done"][r1]),
                   still_moving_end=bool(z["dyn__drift_p"][r1][-16:].mean() > 1e-4))
        # settling time (seed 1): first loop after which drift at p stays < 1e-4
        d = z["dyn__drift_p"][r1]
        above = np.where(d >= 1e-4)[0]
        rec["settle_loop_p"] = int(above[-1] + 2) if len(above) and above[-1] < len(d) - 1 else (1 if not len(above) else np.nan)
        # pushes
        base = j * nk
        rec["dup_control_max_dist"] = float(np.max(D[base + kinds.index("control_dup")]))
        rec["push_floor_plateau"] = float(np.median(D[base + kinds.index("rand1")][-16:]))
        rr = [rate(D[base + kinds.index(k)]) for k in ("rand1", "rand2")]
        rec["recovery_rate_rand_p"] = float(np.nanmean(rr))
        rec["recovery_rate_gap_p"] = rate(D[base + kinds.index("gap")])
        dpre = Dpre[base + kinds.index("prefix_rand")]
        rec["prefix_push_nonrecovering"] = bool(dpre[-16:].mean() > 2 * dpre[0])  # grows instead of returning
        rec["prefix_push_dist_end_over_start"] = float(dpre[-16:].mean() / dpre[0])
        rec["recovery_rate_prefix"] = np.nan if rec["prefix_push_nonrecovering"] else rate(dpre)
        dg_ = D[base + kinds.index("gap")]
        rec["gap_push_peak_amplification"] = float(dg_.max() / dg_[0])
        rec["rand_push_peak_amplification"] = float(np.mean([D[base + kinds.index(k)].max() / D[base + kinds.index(k)][0] for k in ("rand1", "rand2")]))
        t2 = z["push_top2"][j]
        g = lambda kk: LP[base + kinds.index(kk)][:, t2[0]] - LP[base + kinds.index(kk)][:, t2[1]]
        dgap = np.abs(g("gap") - g("control"))
        rec["gap_push_initial_gap_change"] = float(dgap[0])
        rec["gap_readout_recovery_rate"] = rate(dgap, 1e-7, t0=1, tmax=13, k=1) ** (1 / ge) if np.isfinite(rate(dgap, 1e-7, t0=1, tmax=13, k=1)) else np.nan
        rec["gap0"] = float(z["push_gap0"][j])
        rec["lead_abs_eig_Jpp"] = float(np.abs(z["spec_lead"][j]))  # UNCONVERGED leading Ritz estimate (see fd resid)
        rec["lead_eig_half_full_diff"] = float(abs(z["spec_half_lead_abs"][j] - np.abs(z["spec_lead"][j])))
        rec["lead_eig_fd_resid"] = float(z["spec_fd_resid"][j, 1])
        rows.append(rec)

    metrics = ["recovery_rate_rand_p", "recovery_rate_gap_p", "recovery_rate_prefix", "gap_readout_recovery_rate",
               "lead_abs_eig_Jpp", "settle_loop_p", "gap_push_peak_amplification"]  # random-push peak amplification is 1.0 everywhere (monotone decay), omitted
    sel_m = np.array([r["sel_margin"] for r in rows])
    assoc = {}
    for mt in metrics:
        v = np.array([r[mt] for r in rows], float)
        assoc[mt] = {"spearman_vs_margin": spearman(sel_m, v),
                     "median_unsure": float(np.nanmedian(v[unsure])), "median_confident": float(np.nanmedian(v[~unsure])),
                     "range_unsure": [float(np.nanmin(v[unsure])), float(np.nanmax(v[unsure]))],
                     "range_confident": [float(np.nanmin(v[~unsure])), float(np.nanmax(v[~unsure]))]}
    multi = [r for r in rows if r["n_seeds"] > 1]
    # Holm correction over the reported metrics (excluding the withdrawn readout-decay fit)
    tested = [m_ for m_ in metrics if m_ != "gap_readout_recovery_rate"]
    ps = sorted((assoc[m_]["spearman_vs_margin"]["p_perm"], m_) for m_ in tested)
    holm = {}
    run_max = 0.0
    for k_, (pv, m_) in enumerate(ps):
        run_max = max(run_max, min(1.0, (len(ps) - k_) * pv))
        holm[m_] = round(run_max, 4)
    for m_ in tested:
        assoc[m_]["p_holm"] = holm[m_]
    assoc["gap_readout_recovery_rate"]["NOTE"] = "WITHDRAWN: window-sensitive, ~half the fit window below the plateau; signed curves reverse"
    assoc["lead_abs_eig_Jpp"]["NOTE"] = "unconverged Ritz estimates (FD resid median ~0.08; half/full diff ~0.06): not evidence of equal stability"
    # exploratory Welch intervals for unsure - confident differences
    def welch(m_, lvl=1.645):
        v = np.array([r[m_] for r in rows], float)
        a_, b_ = v[unsure & np.isfinite(v)], v[~unsure & np.isfinite(v)]
        d = a_.mean() - b_.mean(); se = np.sqrt(a_.var(ddof=1) / len(a_) + b_.var(ddof=1) / len(b_))
        return [round(d - lvl * se, 4), round(d + lvl * se, 4)]
    summ_ci = {m_: welch(m_) for m_ in ("recovery_rate_rand_p", "settle_loop_p")}
    summ = {"welch90_unsure_minus_confident": summ_ci,
            "nonrecovering_prefix_pushes": [(r["i"], r["cell"], round(r["prefix_push_dist_end_over_start"], 2)) for r in rows if r["prefix_push_nonrecovering"]],"n_prompts": n, "all_fp_valid": all(r["fp_valid"] for r in rows),
            "multistart": {"prompts": len(multi), "seeds_each": 8, "prompts_with_disagreeing_answers": sum(r["n_distinct_answers"] > 1 for r in multi),
                           "max_terminal_state_spread_p_excluding_moving": float(max(r["terminal_state_spread_p"] for r in multi if not r["still_moving_end"]))},
            "margin_persistence": {"spearman_sel_vs_fixed_point_margin": spearman(sel_m, [r["margin_fixed_point"] for r in rows])},
            "still_moving_at_end": [(r["i"], r["cell"], r["end_drift_p_seed1"]) for r in rows if r["still_moving_end"]],
            "association_with_uncertainty": assoc, "rows": rows}
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=1, default=float))

    # figure
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.3))
    col = [CAT[1] if u else CAT[0] for u in unsure]
    mk = ["o" if c else "x" for c in correct]
    for ax, mt, lab in ((axs[0], "recovery_rate_rand_p", "state: remaining fraction per loop\n(random nudges at the answer position)"),
                        (axs[1], "recovery_rate_gap_p", "state: remaining fraction per loop\n(nudge that shrinks the top-2 gap)"),
                        (axs[2], "lead_abs_eig_Jpp", "largest |eigenvalue| of one loop step\n(unconverged estimate; answer position)"),
                        (axs[3], "gap_push_peak_amplification", "peak growth of a nudge that shrinks the gap\n(max distance / initial)")):
        v = np.array([r[mt] for r in rows], float)
        for k in range(n):
            ax.scatter(sel_m[k], v[k], color=col[k], marker=mk[k], s=36)
        a_ = assoc[mt]["spearman_vs_margin"]
        ax.set_xlabel("certainty: top-1 minus top-2 probability"); ax.set_ylabel(lab)
        ax.set_title(f"Spearman ρ = {a_['rho']:+.2f} (p = {a_['p_perm']:.3f}, Holm {a_.get('p_holm', float('nan')):.2f}, n = {a_['n']})", fontsize=9.5)
    axs[0].scatter([], [], color=CAT[1], label="unsure"); axs[0].scatter([], [], color=CAT[0], label="confident")
    axs[0].scatter([], [], color=INK2, marker="o", label="correct"); axs[0].scatter([], [], color=INK2, marker="x", label="wrong")
    axs[0].legend(fontsize=7)
    fig.suptitle("Does an unsure answer have a weaker pull back to its resting point?  (higher = slower recovery)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out}/doubt.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
