"""Digit lever experiment: do loop-aware levers predict the full 10-digit response, and does the context-specific loop
operator matter? Prespecified before the runs.
Usage: python -I huginn_addition/analyze_dglever.py results/<train> results/<pred> results/<eval>

Eval items: 20 held-out digit prompts (2 per answer digit, test cells, both contexts). Response Dz[j, c] = z(push c) -
z(sham) after 48 loops of persistent forcing (centered digit log-probs, 10-vector), push norm 3.
CHECKS: Neumann tails < 1e-3 (train VJP and pred JVP); adjoint consistency on TRAIN items: (A^T g_k) . P_m vs g_k . (A P_m)
  relative error (median < 0.05 required to interpret operator analyses).
P1 EFFICACY & SELECTIVITY: target effect Dz_k under push F_k for families F in {L, G, C}; per item the mean over k of the
  target effect and of selectivity (target minus mean off-target). Paired item-level differences L - G and L - C with item
  bootstrap 95% CIs (resampling items; items are distinct cells). Random pushes: median |Dz|.
P2 CIRCULAR STRUCTURE of off-target effects: per item and push F_k, Spearman between Dz_m (m != k) and minus the circular
  distance |m - k|_10; median over items x k per family (positive = neighbours co-raised).
P3 OPERATOR SPECIFICITY (L pushes): predicted Dz for eval item j and push L_k under
  FULL      G_j (A_j L_k)            own operator
  AVERAGED  G_j (mean_train A_i L_k) operator averaged over training contexts
  SWAPPED   G_j (A_j' L_k)           operator of another eval item j' = (j + 2) mod 20 (different answer digit)
  IDENTITY  G_j L_k                  no loop accumulation (readout geometry only)
  SWAPPED_SAME_DIGIT G_j (A_j'' L_k) with j'' = the other eval item with the same answer digit (other context, other cell)
  Score per item = Pearson correlation between predicted and observed 10 x 10 response matrices (flattened, scale-free);
  also the least-squares slope and the uncalibrated normalised error |pred - obs| / |obs|. PRIMARY: FULL - SWAPPED and
  FULL - AVERAGED paired CIs > 0 -> the item's own operator matters (SWAPPED changes the answer digit too; SWAPPED_SAME_DIGIT
  holds it fixed); FULL - IDENTITY CI > 0 -> the loop's operator adds predictive structure beyond readout geometry.
REVISIONS (Astra, before data): eval/train items are one prompt per context per digit from distinct cells (bootstrap over
  items = cells); GATES: all arrays finite, train and pred Neumann tails median < 1e-3 and 95th percentile < 1e-2, adjoint
  median relative error < 0.05 — positive P3 flags are reported only if the gates pass; selectivity is (10/9) x target
  effect for centered responses (redundant, reported once); C uses more fit data than L/G (secondary reference; L - G is the
  matched-data comparison); norm 3 linearity is unverified; sham last-step drift and endpoint displacement reported.
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr


def main(rt, rp, re):
    zt, zp, ze = np.load(f"{rt}/dglever.npz"), np.load(f"{rp}/dglever.npz"), np.load(f"{re}/dglever.npz")
    pit = json.load(open(f"{rp}/items.json")); split = np.array([p["split"] for p in pit])
    names = [str(x) for x in np.load("results/dglever_pushes.npz")["names"]]
    PL = np.load("results/dglever_pushes_L.npz")["P"].astype(np.float64)
    Z = ze["Z"].astype(np.float64); n = Z.shape[0]
    Dz = Z[:, 1:] - Z[:, :1]  # (n, 32, 10)
    out = dict(train_tail_median=float(np.median(zt["tail"])), train_tail_q95=float(np.quantile(zt["tail"], .95)),
               pred_tail_median=float(np.median(zp["tail"])), pred_tail_q95=float(np.quantile(zp["tail"], .95)),
               sham_last_step_drift_median=float(np.nanmedian(ze["last_step_max_drift"][:, 0])),
               endpoint_displacement_L_median=float(np.nanmedian(ze["endpoint_displacement_q"][:, 1:11])))
    finite = all(np.isfinite(a).all() for a in (zt["AtG"], zt["G"], zp["G"], zp["AP"], ze["Z"]))
    # adjoint check on train items present in pred run
    tr_idx = np.where(split == "train")[0]; ev_idx = np.where(split == "eval")[0]
    uid_tr = list(zt["uid"])
    errs = []
    for ii in tr_idx:
        t = uid_tr.index(int(zp["uid"][ii]))
        lhs = zt["AtG"][t].astype(np.float64) @ PL.T  # (10 k, 10 m)
        rhs = zp["G"][ii].astype(np.float64) @ zp["AP"][ii].astype(np.float64).T
        errs.append(np.abs(lhs - rhs) / (np.abs(lhs) + np.abs(rhs) + 1e-9))
    out["adjoint_rel_err_median"] = float(np.median(errs))
    assert list(zp["uid"][ev_idx]) == list(ze["uid"]), "eval item order mismatch"
    rng = np.random.default_rng(0)
    def ci(v):
        b = [np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(5000)]
        return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
    fam = {"L": 0, "G": 10, "C": 20}
    eff, sel, circ = {}, {}, {}
    for f, o in fam.items():
        T = np.array([[Dz[j, o + k, k] for k in range(10)] for j in range(n)])
        OFF = np.array([[np.mean([Dz[j, o + k, m] for m in range(10) if m != k]) for k in range(10)] for j in range(n)])
        eff[f] = T.mean(1); sel[f] = (T - OFF).mean(1)
        cd = lambda a, b: min((a - b) % 10, (b - a) % 10)
        circ[f] = float(np.median([spearmanr([Dz[j, o + k, m] for m in range(10) if m != k], [-cd(m, k) for m in range(10) if m != k])[0]
                                   for j in range(n) for k in range(10)]))
    out["P1"] = dict(mean_target_effect={f: float(eff[f].mean()) for f in fam},
                     L_minus_G_target_ci=ci(eff["L"] - eff["G"]), L_minus_C_target_ci=ci(eff["L"] - eff["C"]),
                     random_median_abs=float(np.median(np.abs(Dz[:, 30:]))))
    out["P2_circular_spearman_median"] = circ
    # P3 operator specificity
    G_ev = zp["G"][ev_idx].astype(np.float64); AP_ev = zp["AP"][ev_idx].astype(np.float64); AP_tr = zp["AP"][tr_idx].astype(np.float64)
    APbar = AP_tr.mean(0)
    obs = Dz[:, :10, :]  # (n, k, m)
    preds = {"FULL": np.einsum("jmd,jkd->jkm", G_ev, AP_ev), "AVERAGED": np.einsum("jmd,kd->jkm", G_ev, APbar),
             "SWAPPED": np.einsum("jmd,jkd->jkm", G_ev, AP_ev[(np.arange(n) + 2) % n]),
             "SWAPPED_SAME_DIGIT": np.einsum("jmd,jkd->jkm", G_ev, AP_ev[np.arange(n) ^ 1]), "IDENTITY": np.einsum("jmd,kd->jkm", G_ev, PL)}
    ans_ev = ze["ans"]; assert (ans_ev[np.arange(n) ^ 1] == ans_ev).all() and (ans_ev[(np.arange(n) + 2) % n] != ans_ev).all()
    sc = {k: np.array([np.corrcoef(v[j].ravel(), obs[j].ravel())[0, 1] for j in range(n)]) for k, v in preds.items()}
    slope = {k: float(np.median([np.polyfit(v[j].ravel(), obs[j].ravel(), 1)[0] for j in range(n)])) for k, v in preds.items()}
    nerr = {k: float(np.median([np.linalg.norm(v[j] - obs[j]) / np.linalg.norm(obs[j]) for j in range(n)])) for k, v in preds.items()}
    out["P3"] = dict(median_corr={k: float(np.median(v)) for k, v in sc.items()}, median_slope=slope, median_norm_error=nerr,
                     FULL_minus_SWAPPED_SAME_DIGIT_ci=ci(sc["FULL"] - sc["SWAPPED_SAME_DIGIT"]),
                     FULL_minus_SWAPPED_ci=ci(sc["FULL"] - sc["SWAPPED"]), FULL_minus_AVERAGED_ci=ci(sc["FULL"] - sc["AVERAGED"]),
                     FULL_minus_IDENTITY_ci=ci(sc["FULL"] - sc["IDENTITY"]))
    out["P3"]["context_specific_operator_matters"] = bool(out["P3"]["FULL_minus_SWAPPED_ci"][0] > 0 and out["P3"]["FULL_minus_AVERAGED_ci"][0] > 0)
    out["P3"]["loop_operator_beyond_readout"] = bool(out["P3"]["FULL_minus_IDENTITY_ci"][0] > 0)
    gate = bool(finite and out["adjoint_rel_err_median"] < 0.05 and out["pred_tail_median"] < 1e-3 and out["train_tail_median"] < 1e-3
                and out["pred_tail_q95"] < 1e-2 and out["train_tail_q95"] < 1e-2)
    out["P3"]["gates_pass"] = gate
    if not gate:
        out["P3"]["context_specific_operator_matters"] = None; out["P3"]["loop_operator_beyond_readout"] = None
    json.dump(out, open(f"{re}/analysis_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/"), sys.argv[3].rstrip("/"))
