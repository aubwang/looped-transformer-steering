"""Settling-speed experiment (intervene_app.py `settle`). Prespecified before the run; revised after Astra's code review
(still before data).
Usage: python -I huginn_addition/analyze_settle.py results/<settle_run>

Unit = an answer-prediction position (teacher-forced: a controlled token-position proxy for one generation step, not
actual generation), 217 positions in 60 prompts. Loops are 1-based (array index k = after loop k+1). The collection must
be complete (k_done == K) or the analysis aborts.
SPLIT: prompts are split into CALIBRATION and EVALUATION halves (seed 7, stratified by n_digits: 10 + 10 per digit
length). Anything chosen from data (exit-rule settings, the constant baseline) is chosen on calibration prompts only and
reported on evaluation prompts.
OUTCOMES per position:
  t_settle = first loop after which the full-vocab argmax does not change again through loop K (finite-horizon
             stability);
  t_conv   = first loop after which the position's relative state drift stays < 1e-4 through K; CENSORED if drift at
             K >= 1e-4 (coverage and censored count reported; analyses of t_conv are COMPLETER analyses);
  t_kl     = first loop >= 2 with KL(p_t || p_{t-1}) < 5e-4.
PREDICTORS:
  rho_J    = validated Ritz-modulus estimate of the own-block Jacobian J_pp at the fixed point (Arnoldi m=24, one
             start; usable only if Arnoldi completed all m steps, the prompt's fixed point is valid and the leading pair's
             FD residual < 0.05 at both eps). Not a certified spectral radius.
  rho_hat8 = online drift-ratio estimate at loop 8 (geometric mean of d_{k+1}/d_k over loops 4..8).
  rho_24   = landmark drift-ratio estimate over loops 8..24 (descriptive only; uses information through loop 24).
PRIMARY test (one): Spearman(rho_J, t_conv) over converged positions, prompt-cluster bootstrap 95% CI (2000 resamples
  of prompts). All other correlations are EXPLORATORY (reported with CIs, no confirmatory claim): rho_J vs t_settle,
  t_kl; rho_hat8 and rho_24 vs t_conv, t_settle, t_kl; Spearman stratified by n_digits for the primary pair.
PREDICTION of t_conv (evaluation prompts, completers): t_hat = 8 + max(0, log(1e-4 / d_8) / log(rho)) with rho = rho_J
  or rho_hat8 (both available at loop 8 in principle; rho_J is not online); constant baseline = median t_conv of
  calibration completers. MAE and the bootstrap CI of MAE(model) - MAE(constant) over evaluation prompts.
EARLY-EXIT SIMULATION (target: agreement = argmax at exit == argmax at loop K, a fidelity target; true-token accuracy
  also reported). Families and settings: FIXED budgets {4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128}; KL rule (exit at the
  first loop >= 2 with KL < tau, tau in {1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 5e-4, 1e-4}); RATE rule (at loop 8 with
  rho_hat8 < 1, exit at 8 + ceil(log(eps / d_8) / log(rho_hat8)) clipped to [8, K], else K; eps in {3e-2, 1e-2, 3e-3,
  1e-3, 3e-4, 1e-4}); JACOBIAN rule (same with rho_J; reference, not online). For each family, the setting chosen on
  CALIBRATION = smallest mean loops with calibration agreement >= 0.99 (else the most-agreeing setting). On EVALUATION:
  mean loops, agreement, accuracy; prompt-cluster bootstrap CIs of the paired differences RATE - KL and RATE - FIXED in
  mean loops and in agreement. PRACTICALLY USEFUL only if, on evaluation, the RATE rule uses fewer mean loops than both
  KL and FIXED with the 95% CI of each difference entirely below 0, and its agreement is no more than 0.01 below
  theirs. Loop savings ignore the readout/KL overhead (stated).
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.stats import spearmanr

THR = 1e-4
KL_EXIT = 5e-4
K0 = 8


def first_stable(mask):
    if not mask[-1]:
        return None
    bad = np.where(~mask)[0]
    return int(bad[-1] + 1) if len(bad) else 0


def geo_ratio(d, lo, hi):
    hi = min(hi, len(d))
    if hi - lo < 2:
        return None
    rr = d[lo + 1:hi] / np.maximum(d[lo:hi - 1], 1e-30)
    return float(np.exp(np.mean(np.log(np.maximum(rr, 1e-30)))))


def main(run):
    z = np.load(f"{run}/settle.npz", allow_pickle=True)
    P = json.load(open(f"{run}/problems.json"))
    meta = json.load(open(f"{run}/meta.json"))
    am, kl, dp, truth = z["argmax"], z["kl"], z["drift_pos"], z["truth"]
    n, K, _ = am.shape
    assert int(z["k_done"]) == K, "incomplete collection"
    m_full = int(meta["cfg"]["m"])
    arn_complete = int(z["arn_dim"]) == m_full
    lead = z["lead"] if "lead" in z.files else None
    fdr = z["lead_fd_resid"] if "lead_fd_resid" in z.files else None
    # split
    rng = np.random.default_rng(7)
    calib = set()
    for nd in sorted({p["n_digits"] for p in P}):
        idx = [i for i, p in enumerate(P) if p["n_digits"] == nd]
        calib |= set(rng.permutation(idx)[: len(idx) // 2].tolist())
    R = []
    for b, (i, j, q) in enumerate(z["arn_rows"]):
        a_ = am[i, :, j]
        tc = first_stable(dp[i, :, j] < THR)
        klv = np.nan_to_num(kl[i, :, j], nan=9.0)
        kk = np.where(klv < KL_EXIT)[0]
        d = dp[i, :, j]
        ok_j = bool(arn_complete and lead is not None and z["fp_ok"][i] and np.all(fdr[b] < 0.05))
        R.append(dict(i=int(i), j=int(j), split="calib" if int(i) in calib else "eval", n_digits=P[int(i)]["n_digits"],
                      t_settle=first_stable(a_ == a_[-1]) + 1, t_conv=None if tc is None else tc + 1,
                      t_kl=int(kk[0] + 1) if len(kk) else None, rho_J=float(abs(lead[b])) if ok_j else None,
                      rho_hat8=geo_ratio(d, 3, K0), rho_24=geo_ratio(d, 7, 24), d8=float(d[K0 - 1]),
                      correct_final=bool(a_[-1] == truth[i, j])))
    brng = np.random.default_rng(0)

    def boot(fn, rows, B=2000):
        by = {}
        for r in rows:
            by.setdefault(r["i"], []).append(r)
        keys = list(by)
        vals = []
        for _ in range(B):
            v = fn([x for k in brng.choice(keys, len(keys)) for x in by[k]])
            if v is not None and np.isfinite(v):
                vals.append(v)
        return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if len(vals) > 100 else None

    def sp(rows, p, o):
        rs = [r for r in rows if r[p] is not None and r[o] is not None]
        if len(rs) < 8:
            return dict(n=len(rs), spearman=None, ci95=None)
        f = lambda xs: spearmanr([x[p] for x in xs], [x[o] for x in xs])[0] if len(xs) > 5 else None
        return dict(n=len(rs), n_prompts=len({r["i"] for r in rs}), spearman=float(f(rs)), ci95=boot(f, rs))

    out = dict(skipped=meta["skipped"], n_positions=len(R), n_prompts=n, fp_ok_prompts=int(z["fp_ok"].sum()), arnoldi_complete=arn_complete,
               n_rhoJ_usable=sum(r["rho_J"] is not None for r in R), n_conv_censored=sum(r["t_conv"] is None for r in R),
               final_accuracy=float(np.mean([r["correct_final"] for r in R])),
               quantiles={k: np.quantile([r[k] for r in R if r[k] is not None], [.1, .25, .5, .75, .9]).round(4).tolist()
                          for k in ("t_settle", "t_conv", "t_kl", "rho_J", "rho_hat8", "rho_24") if any(r[k] is not None for r in R)})
    out["PRIMARY_rhoJ_tconv"] = sp(R, "rho_J", "t_conv")
    out["primary_by_n_digits"] = {nd: sp([r for r in R if r["n_digits"] == nd], "rho_J", "t_conv") for nd in (2, 3, 4)}
    out["exploratory"] = {f"{p}~{o}": sp(R, p, o) for p in ("rho_J", "rho_hat8", "rho_24") for o in ("t_conv", "t_settle", "t_kl")
                          if not (p == "rho_J" and o == "t_conv")}
    cal = [r for r in R if r["split"] == "calib"]; ev = [r for r in R if r["split"] == "eval"]
    med = float(np.median([r["t_conv"] for r in cal if r["t_conv"] is not None]))
    models = {}
    for pred in ("rho_J", "rho_hat8"):
        rows = [r for r in ev if r[pred] is not None and r["t_conv"] is not None and 0 < r[pred] < 1 and r["d8"] > 0]
        if not rows:
            continue
        that = lambda r, p=pred: K0 + max(0.0, np.log(THR / r["d8"]) / np.log(r[p]))
        mae_m = lambda rs: float(np.mean([abs(that(r) - r["t_conv"]) for r in rs]))
        mae_c = lambda rs: float(np.mean([abs(med - r["t_conv"]) for r in rs]))
        models[pred] = dict(n_eval=len(rows), coverage_of_eval_positions=len(rows) / len(ev), mae_model=mae_m(rows), mae_constant=mae_c(rows),
                            constant_from_calibration=med, mae_diff_ci95=boot(lambda rs: mae_m(rs) - mae_c(rs), rows))
    out["t_conv_prediction_eval"] = models

    # early exit
    def exits(fam, setting, rows):
        ex = []
        for r in rows:
            if fam == "fixed":
                ex.append(min(setting, K)); continue
            if fam == "kl":
                v = np.nan_to_num(kl[r["i"], :, r["j"]], nan=9.0); w = np.where(v < setting)[0]
                ex.append(int(w[0] + 1) if len(w) else K); continue
            rho = r["rho_hat8"] if fam == "rate" else r["rho_J"]
            if rho is None or not (0 < rho < 1):
                ex.append(K); continue
            extra = 0 if r["d8"] <= setting else int(np.ceil(np.log(setting / r["d8"]) / np.log(rho)))
            ex.append(int(min(K, max(K0, K0 + extra))))
        return ex

    def perf(rows, ex):
        if not rows:
            return None
        ag = [am[r["i"], e - 1, r["j"]] == am[r["i"], -1, r["j"]] for r, e in zip(rows, ex)]
        ac = [am[r["i"], e - 1, r["j"]] == truth[r["i"], r["j"]] for r, e in zip(rows, ex)]
        return dict(mean_loops=float(np.mean(ex)), agreement=float(np.mean(ag)), accuracy=float(np.mean(ac)))
    grid = {"fixed": [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128], "kl": [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 5e-4, 1e-4],
            "rate": [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4], "jacobian": [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]}
    table, chosen = {}, {}
    for fam, settings in grid.items():
        table[fam] = {str(s_): dict(calib=perf(cal, exits(fam, s_, cal)), eval=perf(ev, exits(fam, s_, ev))) for s_ in settings}
        ok = [s_ for s_ in settings if table[fam][str(s_)]["calib"]["agreement"] >= 0.99]
        chosen[fam] = min(ok, key=lambda s_: table[fam][str(s_)]["calib"]["mean_loops"]) if ok else \
            max(settings, key=lambda s_: table[fam][str(s_)]["calib"]["agreement"])
    out["early_exit_table"] = table
    out["early_exit_chosen_on_calibration"] = {fam: dict(setting=chosen[fam], eval=table[fam][str(chosen[fam])]["eval"]) for fam in grid}

    def diff(fa, fb, key):
        def f(rs):
            pa, pb = perf(rs, exits(fa, chosen[fa], rs)), perf(rs, exits(fb, chosen[fb], rs))
            return pa[key] - pb[key]
        return dict(point=f(ev), ci95=boot(f, ev, B=1000))
    cmp_ = {f"rate_minus_{o}": dict(mean_loops=diff("rate", o, "mean_loops"), agreement=diff("rate", o, "agreement")) for o in ("kl", "fixed")}
    out["early_exit_paired_eval"] = cmp_
    useful = all(c["mean_loops"]["ci95"] is not None and c["mean_loops"]["ci95"][1] < 0 and c["agreement"]["point"] >= -0.01
                 for c in cmp_.values())
    out["PRACTICALLY_USEFUL_rate_rule"] = bool(useful)
    out["note"] = "loop savings ignore readout/KL overhead; teacher-forced positions are a proxy for generation steps"
    json.dump(dict(summary=out, rows=R), open(f"{run}/analysis_summary.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "early_exit_table"}, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
