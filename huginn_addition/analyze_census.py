"""Prevalence + convergence-mode census on fresh, unselected prompts (intervene_app.py `census`).
Usage: HUGINN_TOKENIZER=... python -I huginn_addition/analyze_census.py out_dir results/<census_shard_run> [...]

Rules fixed BEFORE looking at the results (written while the runs were in progress):
Positions: 0 .. pred_pos[-1] (prompt plus teacher-forced answer prefix; the final answer token is excluded).
Floor f: median over all such positions of all prompts of the late drift (mean relative change over loops 480..511).
Per position, late drift D1 = mean over loops 480..511, D0 = mean over loops 448..479:
  stationary  : D1 < 2e-5
  period-2    : D1 > 1e-4 and median lag-2/lag-1 over loops 480..511 < 0.01 and |D1/D0 - 1| < 0.02
  unresolved  : otherwise
Prompt: 'contains settled period-2' if any position is period-2; 'stationary' if all positions are stationary; else
'unresolved'. Problem: 'cycles' if at least one of its two demo prompts contains a settled period-2 position.
Prevalence CIs: bootstrap over problems within digit-count strata (both demos kept together), 4000 draws.
Mode census (positions not classified period-2): eligible windows = mean relative drift over the window's triple starts
> 10 f and window start >= 16; use the latest eligible window. Fits from the window's Gram sums:
  real    : lam = S01/S00, relative residual rr = (S11 - S01^2/S00)/S11
  two-step: d_{k+2} = a d_{k+1} + b d_k (2x2 normal equations), relative residual r2; roots of z^2 - a z - b.
  class   : 'real' if rr < 0.05 (lam); elif r2 < 0.05 and cond < 1e8: 'complex' if a^2 + 4b < 0 (modulus sqrt(-b),
            angle) else 'real2' (dominant real root); else 'unresolved'. Positions without an eligible window: 'fast'
            (converged below 10 f before loop 16+), retained in the denominator.
  consistency: the previous eligible window (if any) gives the same class and |lam| within 0.02.
These are TRAJECTORY modes at each position (they include upstream forcing; a downstream position can inherit an
upstream slow mode). 'Own-mode candidate' = position whose |lam| exceeds the max |lam| of all earlier fitted positions
of the same prompt by > 0.005. Primary statistic: fraction of fitted real-negative modes in [-1, -0.95] vs neighbouring
equal-width bins, with all classes and exclusions reported.
Token roles: demo (before the query line), query operand digit, query ' +', query ' =', answer prefix (after '='), BOS.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

RNG = np.random.default_rng(0)


def tok(ids):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.environ["HUGINN_TOKENIZER"])
    return [tk.decode([int(i)]) for i in ids]


def roles(labels, pred_pos):
    eq = int(pred_pos[0])
    nl = [i for i, t in enumerate(labels[:eq]) if "\n" in t]
    q0 = (nl[-1] + 1) if nl else 1
    r = []
    for i, t in enumerate(labels):
        if i == 0:
            r.append("bos")
        elif i < q0:
            r.append("demo")
        elif i < eq:
            r.append("query_plus" if t.strip() == "+" else "query_digit" if t.strip().isdigit() else "query_other")
        elif i == eq:
            r.append("query_eq")
        else:
            r.append("answer_prefix")
    return r


def fit_window(g):
    S00, S11, S22, S01, S02, S12, c = g
    out = {}
    if c < 3 or S00 <= 0 or S11 <= 0:
        return None
    lam = S01 / S00
    rr = (S11 - S01 ** 2 / S00) / S11
    M = np.array([[S11, S01], [S01, S00]])
    cond = np.linalg.cond(M)
    a, b = np.linalg.solve(M, [S12, S02]) if np.isfinite(cond) and cond < 1e12 else (np.nan, np.nan)
    r2 = (S22 - 2 * a * S12 - 2 * b * S02 + a * a * S11 + 2 * a * b * S01 + b * b * S00) / S22 if S22 > 0 else np.nan
    if rr < 0.05:
        out = {"cls": "real", "lam": float(lam), "mod": abs(lam)}
    elif np.isfinite(r2) and r2 < 0.05 and cond < 1e8:
        disc = a * a + 4 * b
        if disc < 0:
            out = {"cls": "complex", "mod": float(np.sqrt(-b)), "ang": float(np.degrees(np.arccos(np.clip(a / (2 * np.sqrt(-b)), -1, 1))))}
        else:
            r_ = np.roots([1, -a, -b])
            r_ = r_[np.argmax(np.abs(r_))].real
            out = {"cls": "real2", "lam": float(r_), "mod": abs(float(r_))}
    else:
        out = {"cls": "unresolved"}
    out.update(rr=float(rr), r2=float(r2) if np.isfinite(r2) else None, cond=float(cond))
    return out


def main(out_dir, runs):
    os.makedirs(out_dir, exist_ok=True)
    data = []
    for run in runs:
        z = np.load(f"{run}/census.npz", allow_pickle=True)
        probs = {p["uid"]: p for p in json.load(open(f"{run}/problems.json"))}
        W = z["windows"]
        for i in range(len(z["uid"])):
            data.append(dict(run=run, uid=int(z["uid"][i]), p=probs[int(z["uid"][i])], ids=z["ids"][i], pp=z["pred_pos"][i],
                             drift=z["drift"][i], l2=z["lag2_tail"][i], gram=z["gram"][i], steps=int(z["steps"][i]), W=W,
                             ro_am=z["ro_argmax"][i], ro_lp=z["ro_lp"][i]))
    K = 512
    late_all = np.concatenate([d["drift"][480:512, : int(d["pp"][-1]) + 1].mean(0) for d in data if d["steps"] >= K])
    f = float(np.median(late_all))
    prompts, positions = [], []
    for d in data:
        lim = int(d["pp"][-1]) + 1
        complete = d["steps"] >= K
        labels = tok(d["ids"])
        rl = roles(labels, d["pp"])
        D1 = d["drift"][480:512, :lim].mean(0) if complete else np.full(lim, np.nan)
        D0 = d["drift"][448:480, :lim].mean(0) if complete else np.full(lim, np.nan)
        L2 = np.nanmedian(d["l2"][-32:, :lim], 0) if complete else np.full(lim, np.nan)
        pcls = np.where(D1 < 2e-5, "stationary", np.where((D1 > 1e-4) & (L2 < 0.01) & (np.abs(D1 / D0 - 1) < 0.02), "period2", "unresolved"))
        if not complete:
            pcls[:] = "incomplete"
        prompt_cls = ("incomplete" if not complete else "period2" if (pcls == "period2").any() else
                      "stationary" if (pcls == "stationary").all() else "unresolved")
        am_changes = int((np.diff(d["ro_am"], axis=0) != 0).sum()) if len(d["ro_am"]) > 1 else 0
        truth = [x["digit"] for x in d["p"]["digit_info"]]
        prompts.append(dict(uid=d["uid"], pid=d["p"]["pid"], demo=d["p"]["demo"], n_digits=d["p"]["n_digits"], cls=prompt_cls,
                            p2_positions=[int(j) for j in np.where(pcls == "period2")[0]],
                            p2_tokens=[labels[j] for j in np.where(pcls == "period2")[0]][:3],
                            earliest_p2_role=(rl[int(np.where(pcls == "period2")[0][0])] if (pcls == "period2").any() else None),
                            unresolved_positions=int((pcls == "unresolved").sum()), answer_argmax_changes_tail=am_changes,
                            correct=bool(np.array_equal(np.array(d["ro_lp"][-1].argmax(-1)), np.array(truth)))))
        # mode census
        fitted_prev_max = 0.0
        for j in range(lim):
            rec = dict(uid=d["uid"], pid=d["p"]["pid"], pos=j, role=rl[j], tok=labels[j], pos_cls=str(pcls[j]))
            if pcls[j] == "period2" or not complete:
                rec["mode"] = {"cls": "excluded_" + str(pcls[j])}
                positions.append(rec); continue
            elig = []
            for wi, (a_, b_) in enumerate(d["W"]):
                g = d["gram"][wi, j]
                if a_ < 16 or g[6] < 3:
                    continue
                mrel = d["drift"][a_ : b_ - 2, j].mean()
                if mrel > 10 * f:
                    elig.append(wi)
            if not elig:
                rec["mode"] = {"cls": "fast"}
                positions.append(rec); continue
            w = elig[-1]
            m = fit_window(d["gram"][w, j]) or {"cls": "unresolved"}
            m["window"] = [int(x) for x in d["W"][w]]
            if len(elig) > 1:
                mp = fit_window(d["gram"][elig[-2], j]) or {"cls": "unresolved"}
                m["consistent"] = bool(mp.get("cls") == m.get("cls") and "mod" in m and "mod" in mp and abs(mp["mod"] - m["mod"]) < 0.02)
            else:
                m["consistent"] = None
            if "mod" in m:
                m["own_candidate"] = bool(m["mod"] > fitted_prev_max + 0.005)
                fitted_prev_max = max(fitted_prev_max, m["mod"])
            rec["mode"] = m
            positions.append(rec)

    # prevalence
    pids = sorted({p["pid"] for p in prompts})
    strata = {}
    for pid in pids:
        nd = next(p["n_digits"] for p in prompts if p["pid"] == pid)
        strata.setdefault(nd, []).append(pid)
    byp = {pid: [p for p in prompts if p["pid"] == pid] for pid in pids}

    def stats(sel_pids):
        ps = [p for pid in sel_pids for p in byp[pid]]
        return (np.mean([p["cls"] == "period2" for p in ps]), np.mean([any(p["cls"] == "period2" for p in byp[pid]) for pid in sel_pids]),
                np.mean([p["cls"] == "unresolved" for p in ps]))

    est = stats(pids)
    bs = np.array([stats([pid for nd, lst in strata.items() for pid in RNG.choice(lst, len(lst))]) for _ in range(4000)])
    ci = np.percentile(bs, [2.5, 97.5], 0)
    prev = {"prompt_level_period2": [round(est[0], 3), round(ci[0, 0], 3), round(ci[1, 0], 3)],
            "problem_level_period2_any_demo": [round(est[1], 3), round(ci[0, 1], 3), round(ci[1, 1], 3)],
            "prompt_level_unresolved": [round(est[2], 3), round(ci[0, 2], 3), round(ci[1, 2], 3)],
            "counts": {c: sum(p["cls"] == c for p in prompts) for c in ("period2", "stationary", "unresolved", "incomplete")},
            "by_n_digits": {nd: round(float(np.mean([p["cls"] == "period2" for pid in lst for p in byp[pid]])), 3) for nd, lst in strata.items()},
            "both_demos_period2": sum(all(p["cls"] == "period2" for p in byp[pid]) for pid in pids),
            "earliest_p2_role": {r: sum(p["earliest_p2_role"] == r for p in prompts) for r in sorted({p["earliest_p2_role"] for p in prompts if p["earliest_p2_role"]})},
            "answer_flicker_prompts": [(p["pid"], p["demo"], p["answer_argmax_changes_tail"]) for p in prompts if p["answer_argmax_changes_tail"] > 0]}

    # census summary
    cls_counts = {}
    for r in positions:
        cls_counts[r["mode"]["cls"]] = cls_counts.get(r["mode"]["cls"], 0) + 1
    realneg = np.array([r["mode"]["lam"] for r in positions if r["mode"].get("cls") in ("real", "real2") and r["mode"]["lam"] < 0])
    realpos = np.array([r["mode"]["lam"] for r in positions if r["mode"].get("cls") in ("real", "real2") and r["mode"]["lam"] > 0])
    cmod = np.array([r["mode"]["mod"] for r in positions if r["mode"].get("cls") == "complex"])
    bins = np.arange(-1.10, -0.40, 0.05)
    hist_neg = {f"[{bins[k]:.2f},{bins[k+1]:.2f})": int(((realneg >= bins[k]) & (realneg < bins[k + 1])).sum()) for k in range(len(bins) - 1)}
    pbins = np.arange(0.40, 1.10, 0.05)
    hist_pos = {f"[{pbins[k]:.2f},{pbins[k+1]:.2f})": int(((realpos >= pbins[k]) & (realpos < pbins[k + 1])).sum()) for k in range(len(pbins) - 1)}
    by_role = {}
    for role in sorted({r["role"] for r in positions}):
        rs = [r for r in positions if r["role"] == role]
        neg = [r["mode"]["lam"] for r in rs if r["mode"].get("cls") in ("real", "real2") and r["mode"]["lam"] < 0]
        by_role[role] = {"n": len(rs), "classes": {c: sum(r["mode"]["cls"] == c for r in rs) for c in sorted({r["mode"]["cls"] for r in rs})},
                         "real_neg_n": len(neg), "real_neg_in_[-1,-0.95]": int(sum(-1 <= x < -0.95 for x in neg)),
                         "real_neg_median": float(np.median(neg)) if neg else None}
    own = [r for r in positions if r["mode"].get("own_candidate")]
    own_neg = np.array([r["mode"]["lam"] for r in own if r["mode"].get("cls") in ("real", "real2") and r["mode"]["lam"] < 0])
    summ = {"floor_late_drift_median": f, "n_prompts": len(prompts), "n_positions": len(positions), "prevalence": prev,
            "mode_classes": cls_counts, "real_negative_hist": hist_neg, "real_positive_hist": hist_pos,
            "complex_modulus_quantiles": np.percentile(cmod, [10, 50, 90]).round(3).tolist() if len(cmod) else None,
            "own_candidates": {"n": len(own), "real_neg_n": len(own_neg),
                               "real_neg_hist": {f"[{bins[k]:.2f},{bins[k+1]:.2f})": int(((own_neg >= bins[k]) & (own_neg < bins[k + 1])).sum()) for k in range(len(bins) - 1)},
                               "roles": {ro: sum(r["role"] == ro for r in own) for ro in sorted({r["role"] for r in own})}},
            "by_role": by_role, "consistency_rate_fitted": float(np.mean([r["mode"]["consistent"] for r in positions if r["mode"].get("consistent") is not None])) if positions else None}
    json.dump({"summary": summ, "prompts": prompts}, open(f"{out_dir}/summary.json", "w"), indent=1, default=float)
    json.dump(positions, open(f"{out_dir}/positions.json", "w"), default=float)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], [r.rstrip("/") for r in sys.argv[2:]])
