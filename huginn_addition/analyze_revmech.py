"""Mechanism of the readout reversal (intervene_app.py `revmech`). Prespecified before the run (revised after Astra's
review, before data).
Usage: python -I huginn_addition/analyze_revmech.py results/<revmech_run> results/<dglever_pred_run> results/<dglever_eval_run>

Rows = 20 held-out digit prompts x 21 pushes (L0..L9 loop-aware levers, G0..G9 readout-gradient directions, R0 random;
norm 3; frozen population pushes from the dglever runs). Arnoldi from each push, m = 50 >= T + 1 with horizon T = 48.
For readout m (centered digit logit z_m, gradient g_m): exact impulse response c_t^m = |P| (V^T g_m) . y_t, y_0 = e1,
y_{t+1} = H y_t[:m] (exact for t <= 50); immediate effect c_0^m = g_m . P (the IDENTITY prediction); persistent response
S^m = sum_{t=1..48} c_t^m.
MODES (Ritz approximations of J restricted to the push's Krylov space; push-dependent): H[:m, :m] = W diag(lambda) W^-1;
a_j^m = |P| (V[:m]^T g_m . W)_j (W^-1 e1)_j; immediate contribution Re(a_j) (summed over conjugates = 2 Re for a pair) and
forced contribution Re(a_j f48(lambda_j)), f48(lambda) = sum_{t=1..48} lambda^t computed directly.
GATES (per row; quantiles reported): (G1) L-row S^m vs the independent pred run g_m . (A L_k): median relative error < 0.05
  and 90th percentile < 0.2 (10 readouts per row); (G2) modal reconstruction of S^m and of c_0^m: 90th percentile relative
  error < 0.1; (G3) Arnoldi orthogonality max |V^T V - I| < 1e-6 (median); FD eps/2 relative difference of J P < 0.02
  (median). Also reported: cond(W), cancellation ratios sum_j |a_j f48_j| / |S| and sum_j |a_j| / |c_0|, Ritz residuals of
  the 3 dominant contributing pairs, and sensitivity of the PRIMARY statistic to the Krylov size (m' = 40 using the
  leading block of H). Gates failing or Krylov-size sensitivity flipping the PRIMARY conclusion -> INCONCLUSIVE.
PRIMARY (the full-matrix reversal of L pushes, cf. dglever IDENTITY corr -0.59): per held-out item, the 10 x 10 matrices
  (push L_k, readout m) of c_0 (immediate) and S (accumulated); r_obs = corr(c_0, S) per item (reproduces the reversal if
  negative; also corr(S, observed eval dz) as a check). COUNTERFACTUAL S_noflip = sum_j Re(a_j) |f48(lambda_j)|: each mode
  keeps its immediate sign but is weighted by its gain magnitude. MECHANISM:
  GAIN-PHASE REVERSAL if median_items corr(c_0, S) < 0 AND median_items corr(c_0, S_noflip) > 0 (removing the gains' signs
    / phases removes the reversal);
  REWEIGHTING if both medians < 0 (the reversal survives without sign flips: it comes from re-weighting modes);
  INCONCLUSIVE otherwise or if gates fail.
  Item-bootstrap CIs (5000) of the two medians reported.
SECONDARY: forced-contribution shares of mode classes FORWARD (Re lambda >= |lambda|^2), OVERSHOOT (|Im| <= 0.1 |lambda|,
  Re < 0), ROTATING (other) for L and G pushes (target readout and full matrix); readout-weighted modal amplitude shares
  sum |a_j| by class; target-readout sign reversal fraction (sign S != sign c_0) for G, L, R; median c_1 / c_0; first
  sign change. Paired G - L contrasts per item; R0 descriptive.
"""
import json
import sys

import numpy as np


def f48(lam, T=48):
    return np.array([np.sum(l ** np.arange(1, T + 1)) for l in np.atleast_1d(lam)])


def row_quantities(H, vg, pn, mk, T=48):
    """vg: (m+1, 10) projections; returns c (T+1, 10), modal a (mk, 10), lam (mk,), W cond, residuals."""
    m = H.shape[1]
    y = np.zeros(m + 1); y[0] = 1.0
    cs = [pn * vg.T @ y]
    for t in range(T):
        y = H @ y[:m]
        cs.append(pn * vg.T @ y)
    c = np.array(cs)  # (T+1, 10)
    Hs = H[:mk, :mk]
    lam, W = np.linalg.eig(Hs)
    b = np.linalg.solve(W, np.eye(mk)[:, 0])
    a = pn * (vg[:mk].T @ W) * b[None, :]  # (10, mk)
    res = np.abs(H[mk, mk - 1] * W[-1, :]) / np.linalg.norm(W, axis=0) if mk < H.shape[0] else np.zeros(mk)
    return c, a, lam, float(np.linalg.cond(W)), res


def main(rr, rp, re):
    z = np.load(f"{rr}/revmech.npz")
    H, VG, PN = z["H"], z["VG"], z["PN"]
    ri, rpi, names = z["row_item"], z["row_push"], [str(x) for x in z["push_names"]]
    n = int(ri.max()) + 1; m = H.shape[2]; T = 48
    zp = np.load(f"{rp}/dglever.npz"); pit = json.load(open(f"{rp}/items.json"))
    ev = [k for k, p in enumerate(pit) if p["split"] == "eval"]
    assert list(zp["uid"][ev]) == list(z["uid"])
    G_ev, AP_ev = zp["G"][ev].astype(np.float64), zp["AP"][ev].astype(np.float64)  # (n,10,d), (n,10 pushes,d)
    ze = np.load(f"{re}/dglever.npz"); assert list(ze["uid"]) == list(z["uid"])
    Dz = ze["Z"][:, 1:] - ze["Z"][:, :1]  # (n, 32, 10)
    out = {}
    per = {}
    g1, g2S, g2c, conds, canc, canc0 = [], [], [], [], [], []
    cls_acc = {"L": [], "G": [], "R": []}
    tgt = {"L": [], "G": [], "R": []}
    for r in range(len(ri)):
        i, p = int(ri[r]), int(rpi[r]); nm = names[p]; fam = nm[0]
        c, a, lam, cw, res = row_quantities(H[r], VG[r], PN[r], m)
        F = f48(lam)
        S = c[1:].sum(0); c0 = c[0]
        S_modal = np.real(a @ F); c0_modal = np.real(a.sum(1))
        g2S += list(np.abs(S_modal - S) / (np.abs(S) + 1e-9)); g2c += list(np.abs(c0_modal - c0) / (np.abs(c0) + 1e-9))
        conds.append(cw); canc.append(float(np.median(np.abs(a * F[None]).sum(1) / (np.abs(S) + 1e-9))))
        canc0.append(float(np.median(np.abs(a).sum(1) / (np.abs(c0) + 1e-9))))
        S_noflip = np.real(a) @ np.abs(F)
        k = int(nm[1:]) if nm[1:].isdigit() else None
        if fam == "L":
            pred = G_ev[i] @ AP_ev[i, k]  # (10,)
            g1 += list(np.abs(S - pred) / (np.abs(pred) + 1e-9))
        fwd = lam.real >= np.abs(lam) ** 2
        ovs = (~fwd) & (np.abs(lam.imag) <= 0.1 * np.abs(lam)) & (lam.real < 0)
        rot = (~fwd) & (~ovs)
        contrib = np.real(a * F[None])  # (10, mk)
        per[(i, nm)] = dict(c0=c0, S=S, S_noflip=S_noflip, fwd=contrib[:, fwd].sum(1), ovs=contrib[:, ovs].sum(1), rot=contrib[:, rot].sum(1),
                            amp_fwd=np.abs(a[:, fwd]).sum(1), amp_rev=np.abs(a[:, ~fwd]).sum(1), c1=c[1])
        if k is not None:
            tgt[fam].append(dict(c0=c0[k], S=S[k], c1=c[1, k], first=next((t for t in range(1, T + 1) if np.sign(c[t, k]) != np.sign(c[0, k])), None),
                                 fwd=contrib[k, fwd].sum(), ovs=contrib[k, ovs].sum(), rot=contrib[k, rot].sum()))
        elif fam == "R":
            tgt["R"].append(dict(c0=c0, S=S))
    out["gates"] = dict(G1_median=float(np.median(g1)), G1_q90=float(np.quantile(g1, .9)),
                        G2_S_q90=float(np.quantile(g2S, .9)), G2_c0_q90=float(np.quantile(g2c, .9)),
                        G3_orth_median=float(np.median(z["orth_err"])), G3_fd_eps_median=float(np.median(z["fd_eps_rel"])),
                        condW_median=float(np.median(conds)), condW_q90=float(np.quantile(conds, .9)),
                        cancellation_S_median=float(np.median(canc)), cancellation_c0_median=float(np.median(canc0)))
    gt = out["gates"]
    gates_ok = gt["G1_median"] < 0.05 and gt["G1_q90"] < 0.2 and gt["G2_S_q90"] < 0.1 and gt["G2_c0_q90"] < 0.1 and gt["G3_orth_median"] < 1e-6 and gt["G3_fd_eps_median"] < 0.02
    out["gates"]["pass"] = bool(gates_ok)

    def primary(mk):
        r_id, r_nf, r_obs = [], [], []
        for i in range(n):
            C0 = np.array([per[(i, f"L{k}")]["c0"] for k in range(10)])
            if mk == m:
                SS = np.array([per[(i, f"L{k}")]["S"] for k in range(10)])
                NF = np.array([per[(i, f"L{k}")]["S_noflip"] for k in range(10)])
            else:  # recompute BOTH S and S_noflip from the smaller Krylov block (sensitivity check)
                NF, SS = [], []
                for k in range(10):
                    r = [rr_ for rr_ in range(len(ri)) if int(ri[rr_]) == i and names[int(rpi[rr_])] == f"L{k}"][0]
                    c, a, lam, _, _ = row_quantities(H[r][: mk + 1, :mk], VG[r][: mk + 1], PN[r], mk)
                    NF.append(np.real(a) @ np.abs(f48(lam))); SS.append(np.real(a @ f48(lam)))
                NF, SS = np.array(NF), np.array(SS)
            r_id.append(np.corrcoef(C0.ravel(), SS.ravel())[0, 1]); r_nf.append(np.corrcoef(C0.ravel(), NF.ravel())[0, 1])
            r_obs.append(np.corrcoef(SS.ravel(), Dz[i, :10].ravel())[0, 1])
        return np.array(r_id), np.array(r_nf), np.array(r_obs)
    rid, rnf, robs = primary(m)
    rng = np.random.default_rng(0)
    ci = lambda v: [float(np.quantile([np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(5000)], q)) for q in (.025, .975)]
    rid40, rnf40, _ = primary(40)
    def verdict(a_, b_):
        return "GAIN-PHASE REVERSAL" if np.median(a_) < 0 and np.median(b_) > 0 else ("REWEIGHTING" if np.median(a_) < 0 and np.median(b_) < 0 else "INCONCLUSIVE")
    v50, v40 = verdict(rid, rnf), verdict(rid40, rnf40)
    out["PRIMARY"] = dict(median_corr_c0_S=float(np.median(rid)), ci_c0_S=ci(rid), median_corr_c0_S_noflip=float(np.median(rnf)), ci_c0_noflip=ci(rnf),
                          median_corr_S_observed=float(np.median(robs)), krylov40=dict(c0_S=float(np.median(rid40)), c0_noflip=float(np.median(rnf40)), verdict=v40),
                          verdict=(v50 if (gates_ok and v50 == v40) else "INCONCLUSIVE"), verdict_before_gates=v50)
    sec = {}
    for fam in ("L", "G"):
        R = tgt[fam]
        rev = [x for x in R if np.sign(x["S"]) != np.sign(x["c0"])]
        sec[fam] = dict(target_reversal_fraction=len(rev) / len(R), median_c1_over_c0=float(np.median([x["c1"] / x["c0"] for x in R if x["c0"] != 0])),
                        first_sign_change=dict(sorted({t: sum(1 for x in R if x["first"] == t) for t in {x["first"] for x in R}}.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))),
                        target_median_contrib=dict(forward=float(np.median([x["fwd"] for x in R])), overshoot=float(np.median([x["ovs"] for x in R])),
                                                   rotating=float(np.median([x["rot"] for x in R]))),
                        fullmatrix_median_contrib=dict(forward=float(np.median([per[(i, f"{fam}{k}")]["fwd"] for i in range(n) for k in range(10)])),
                                                       overshoot=float(np.median([per[(i, f"{fam}{k}")]["ovs"] for i in range(n) for k in range(10)])),
                                                       rotating=float(np.median([per[(i, f"{fam}{k}")]["rot"] for i in range(n) for k in range(10)]))),
                        readout_weighted_amplitude_share_reversing=float(np.median([per[(i, f"{fam}{k}")]["amp_rev"] / (per[(i, f"{fam}{k}")]["amp_rev"] + per[(i, f"{fam}{k}")]["amp_fwd"])
                                                                                    for i in range(n) for k in range(10)])))
    sec["R0_median_abs_S"] = float(np.median([np.abs(x["S"]).mean() for x in tgt["R"]]))
    out["SECONDARY"] = sec
    for k_ in ("first_sign_change",):
        for fam in ("L", "G"):
            out["SECONDARY"][fam][k_] = {str(kk): vv for kk, vv in out["SECONDARY"][fam][k_].items()}
    json.dump(out, open(f"{rr}/analysis_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/"), sys.argv[3].rstrip("/"))
