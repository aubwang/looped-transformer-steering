"""Linear-response (resolvent) test of steerability, digits vs truth (intervene_app.py `respond`). Prespecified before the
runs (revised after Astra's review, still before data).
Usage: python -I huginn_addition/analyze_respond.py results/<respond_digits_run> results/<respond_truth_run>

Per item at the settled answer-position state x* (q = last token; d = dose-1 push): S0 = g . d (immediate readout
coupling), S_T = g . sum_{k=1..48} J^k d (prediction at the observed horizon T = 48), S_inf = g . sum_{k=1..64} J^k d,
retention = S_inf / S0, amplification |delta_lin| / |d|; observed per-unit-dose forced responses at doses 0.1 and 1.
VALIDITY per item: cumulative sham drift over the 48 forced loops at q < 1e-3 (relative) AND Neumann tail < 1e-3 AND FD
  step sensitivity of J d (eps vs eps/2) < 0.05. Task-level medians reported. >= 20 valid items per task required.
L1 LINEAR PREDICTION (per task): at dose 0.1, |obs - S_T| <= 0.3 |S_T| + 0.05 for >= 70% of valid items; Spearman.
ELIGIBLE for ratios: valid items with S0 >= 0.02 (the direction pushes the readout the intended way). Reported: exclusion
  counts, reversal fraction (retention < 0), and sensitivity of medians to the cutoff (0.01, 0.05).
H_att (recurrent retention explains the weaker truth steering): ratio R_att = median retention(digits) / median
  retention(truth) on eligible items; HOLDS if R_att > 2 and the bootstrap 95% CI lower bound of R_att > 1.
H_coup (immediate coupling explains it): R_coup = median S0(digits) / median S0(truth) on eligible items; same rule. Also
  S0 / |d| (coupling per unit push norm) reported.
GATE: H_att / H_coup conclusions are drawn only if L1 holds in both tasks and both tasks have >= 20 valid items.
EXACT DECOMPOSITION (items eligible with S_inf > 0): mean log S_inf = mean log S0 + mean log retention per task; the
  digits-minus-truth difference of mean log S_inf splits exactly into the coupling and retention terms.
Bootstrap: 5000 resamples of units (truth: city/pair units, both statements together; digits: items), within task.
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr


def load(run):
    z = np.load(f"{run}/respond.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def main(rd, rt):
    rng = np.random.default_rng(0)
    out = {}
    T = {"digits": load(rd), "truth": load(rt)}
    sets = {}
    for name, z in T.items():
        valid = (z["sham_drift_q"] < 1e-3) & (z["neumann_tail"] < 1e-3) & (z["fd_rel_eps_half"] < 0.05)
        obs01, ST, S0, Sinf = z["obs"][:, 0], z["ST"], z["S0"], z["Sinf"]
        ok = np.abs(obs01 - ST) <= 0.3 * np.abs(ST) + 0.05
        elig = valid & (S0 >= 0.02)
        ret = Sinf / np.where(S0 == 0, np.nan, S0)
        sens = {c: (float(np.median(ret[valid & (S0 >= c)])) if (valid & (S0 >= c)).any() else None) for c in (0.01, 0.02, 0.05)}
        out[name] = dict(n=int(len(S0)), valid=int(valid.sum()), median_sham_drift_q=float(np.median(z["sham_drift_q"])),
                         median_sham_drift_prefix=float(np.median(z["sham_drift_prefix"])), median_tail=float(np.median(z["neumann_tail"])),
                         median_fd_rel=float(np.median(z["fd_rel_eps_half"])),
                         L1_frac_accurate=float(ok[valid].mean()) if valid.any() else None,
                         L1_holds=bool(valid.sum() >= 20 and ok[valid].mean() >= 0.7),
                         L1_spearman=float(spearmanr(obs01[valid], ST[valid])[0]) if valid.sum() > 3 else None,
                         eligible=int(elig.sum()), reversal_fraction=float((ret[elig] < 0).mean()) if elig.any() else None,
                         median_S0=float(np.median(S0[elig])) if elig.any() else None,
                         median_S0_per_norm=float(np.median(S0[elig] / z["dnorm"][elig])) if elig.any() else None,
                         median_ST=float(np.median(ST[valid])) if valid.any() else None, median_Sinf=float(np.median(Sinf[elig])) if elig.any() else None,
                         median_obs_dose01=float(np.median(obs01[valid])) if valid.any() else None,
                         median_obs_dose1=float(np.median(z["obs"][valid, 1])) if valid.any() else None,
                         median_retention=float(np.median(ret[elig])) if elig.any() else None, retention_cutoff_sensitivity=sens,
                         median_amplification=float(np.median(z["dlin_norm"][valid] / z["dnorm"][valid])) if valid.any() else None,
                         median_cos_g_d=float(np.median(z["cos_g_d"][valid])) if valid.any() else None,
                         median_cos_d_dlin=float(np.median(z["cos_d_dlin"][valid])) if valid.any() else None,
                         median_oneshot_immediate=float(np.median(z["oneshot_imm"][valid])) if valid.any() else None)
        sets[name] = dict(units=z["unit"][elig], S0=S0[elig], ret=ret[elig], Sinf=Sinf[elig])

    def boot_ratio(fn_a, fn_b):
        def draw(s):
            uu = np.unique(s["units"]); pick = rng.choice(uu, len(uu))
            return np.concatenate([np.where(s["units"] == u)[0] for u in pick])
        vals = []
        for _ in range(5000):
            ia, ib = draw(sets["digits"]), draw(sets["truth"])
            a, b = fn_a(ia), fn_b(ib)
            if b != 0 and np.isfinite(a / b):
                vals.append(a / b)
        return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))] if len(vals) > 100 else None
    gate = out["digits"]["L1_holds"] and out["truth"]["L1_holds"]
    sd, st = sets["digits"], sets["truth"]
    if len(sd["ret"]) and len(st["ret"]):
        r_att = float(np.median(sd["ret"]) / np.median(st["ret"])) if np.median(st["ret"]) != 0 else None
        ci_att = boot_ratio(lambda i: np.median(sd["ret"][i]), lambda i: np.median(st["ret"][i]))
        r_cp = float(np.median(sd["S0"]) / np.median(st["S0"]))
        ci_cp = boot_ratio(lambda i: np.median(sd["S0"][i]), lambda i: np.median(st["S0"][i]))
        out["H_att"] = dict(ratio=r_att, ci95=ci_att, holds=bool(gate and r_att is not None and r_att > 2 and ci_att and ci_att[0] > 1), gated_by_L1=not gate)
        out["H_coup"] = dict(ratio=r_cp, ci95=ci_cp, holds=bool(gate and r_cp > 2 and ci_cp and ci_cp[0] > 1), gated_by_L1=not gate)
    dec = {}
    for name in ("digits", "truth"):
        s = sets[name]; m = s["Sinf"] > 0
        dec[name] = dict(n=int(m.sum()), mean_log_Sinf=float(np.mean(np.log(s["Sinf"][m]))) if m.any() else None,
                         mean_log_S0=float(np.mean(np.log(s["S0"][m]))) if m.any() else None,
                         mean_log_ret=float(np.mean(np.log(s["ret"][m]))) if m.any() else None)
    if all(dec[k]["n"] for k in dec):
        dec["difference"] = {k: dec["digits"][k] - dec["truth"][k] for k in ("mean_log_Sinf", "mean_log_S0", "mean_log_ret")}
    out["exact_log_decomposition"] = dec
    json.dump(out, open(f"{rt}/analysis_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/"))
