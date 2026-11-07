"""Local readout transfer operator (intervene_app.py `rtransfer`). Prespecified before the run.
Usage: python -I huginn_addition/analyze_rtransfer.py results/<rtransfer_run>

Per held-out digit prompt (the 20 dglever eval items, x* after 96 loops): Q = orthonormal basis of the span of the 10
centered digit-logit gradients g_k (rank 9 expected: they sum to zero; rank cut at 1e-6 relative). One-step readout
transfer B = Q^T J (rows: readout directions; computed as (J^T Q)^T from VJPs). Decomposition J^T Q = Q A_in + B_perp,
A_in = Q^T J^T Q (within-readout transfer), B_perp = (I - Q Q^T) J^T Q (readout input arriving from outside the span).
GATE: adjoint consistency Q^T (J Q) [FD JVPs] vs A_in^T [VJPs]: median relative Frobenius error < 0.05.
R1 WITHIN-READOUT SHARE rho = ||A_in||_F / ||J^T Q||_F per item (1 = the readout is fed only from inside its span; small
   = fed from outside). Reported: median and item-bootstrap CI; 'FED FROM OUTSIDE' if the CI upper bound < 0.5.
R2 PUSH GEOMETRY (frozen population pushes from results/dglever_pushes.npz): for L_k, G_k, C_k: in-span fraction
   ||Q^T P|| / ||P|| and one-step readout gain ||Q^T J P|| / ||P|| = ||(J^T Q)^T P|| / ||P||; target one-step effect
   g_k . J P vs immediate g_k . P. Reported per family; L vs G paired item comparisons (bootstrap CIs).
R3 WITHIN-READOUT MAP: eigenvalues of A_in^T (= Q^T J Q): median real part, fraction with negative real part, spectral
   radius; and how much of a readout-aligned push survives one step inside the readout span: ||Q^T J Q u|| for u in span
   (singular values of A_in).
REVISION (after the run, before interpretation, flagged by Astra): the original 1e-6 relative rank cut admitted float
noise (10th singular value 1.1e-6 relative) on 2 items, which then failed the adjoint check (errors 76, 66). The cut is
now 1e-4 (all items rank 9, the 10th singular value is <= 1.1e-6 on every item); the original output is kept as
analysis_summary_cut1e-6.json.
R4 TOP INPUT DIRECTIONS: SVD of B = Q^T J = U S V^T; cosine of each push family with span(V) (fraction of ||P||^2 in the
   top-9 right singular vectors), L vs G.
"""
import json
import sys

import numpy as np


def main(run):
    z = np.load(f"{run}/rtransfer.npz")
    G, JTG, JG = (z[k].astype(np.float64) for k in ("G", "JTG", "JG"))
    Pz = np.load("results/dglever_pushes.npz"); P = Pz["P"].astype(np.float64); names = [str(x) for x in Pz["names"]]
    n = G.shape[0]
    rows = []
    for i in range(n):
        Gi = G[i]  # (10, d)
        U_, S_, Vt_ = np.linalg.svd(Gi, full_matrices=False)
        r = int((S_ > 1e-4 * S_[0]).sum())  # revised from 1e-6 (see docstring)
        Q = Vt_[:r].T  # (d, r) orthonormal span of the gradients
        # express J^T Q: J^T is linear, J^T Q = J^T Gi^T C where Q = Gi^T C  => C = pinv(Gi^T) Q
        Cm = np.linalg.lstsq(Gi.T, Q, rcond=None)[0]  # (10, r)
        JTQ = JTG[i].T @ Cm  # (d, r)
        JQ = JG[i].T @ Cm    # (d, r)  (J g_k combined linearly)
        A_in = Q.T @ JTQ     # (r, r) = Q^T J^T Q
        adj_err = np.linalg.norm(Q.T @ JQ - A_in.T) / np.linalg.norm(A_in)
        B_perp = JTQ - Q @ A_in
        rho = np.linalg.norm(A_in) / np.linalg.norm(JTQ)
        ev = np.linalg.eigvals(A_in.T)
        sv_in = np.linalg.svd(A_in, compute_uv=False)
        Ub, Sb, Vtb = np.linalg.svd(JTQ.T, full_matrices=False)  # B = Q^T J = JTQ^T (r x d)
        Vtop = Vtb[:r].T
        fam = {}
        for f in ("L", "G", "C"):
            vals = []
            for k in range(10):
                p = P[names.index(f"{f}{k}")]
                vals.append(dict(inspan=np.linalg.norm(Q.T @ p) / np.linalg.norm(p), gain=np.linalg.norm(JTQ.T @ p) / np.linalg.norm(p),
                                 topV=np.linalg.norm(Vtop.T @ p) ** 2 / np.linalg.norm(p) ** 2,
                                 c0=Gi[k] @ p, c1=JTG[i][k] @ p))
            fam[f] = {kk: float(np.median([v[kk] for v in vals])) for kk in vals[0]}
        rows.append(dict(rank=r, adj_err=float(adj_err), rho=float(rho), ev_re_median=float(np.median(ev.real)), ev_neg_frac=float((ev.real < 0).mean()),
                         ev_radius=float(np.abs(ev).max()), sv_in=sv_in.round(3).tolist(), outside_norm_share=float(np.linalg.norm(B_perp) / np.linalg.norm(JTQ)),
                         fam=fam))
    rng = np.random.default_rng(0)
    ci = lambda v: [float(np.quantile([np.median(np.array(v)[rng.integers(0, len(v), len(v))]) for _ in range(5000)], q)) for q in (.025, .975)]
    rho = [r["rho"] for r in rows]
    gate = float(np.median([r["adj_err"] for r in rows]))
    out = dict(gate_adjoint_median=gate, gate_pass=bool(gate < 0.05), rank=[r["rank"] for r in rows],
               R1=dict(median_rho=float(np.median(rho)), ci=ci(rho), fed_from_outside=bool(ci(rho)[1] < 0.5),
                       median_outside_norm_share=float(np.median([r["outside_norm_share"] for r in rows]))),
               R3=dict(median_eig_real=float(np.median([r["ev_re_median"] for r in rows])), median_neg_frac=float(np.median([r["ev_neg_frac"] for r in rows])),
                       median_radius=float(np.median([r["ev_radius"] for r in rows])), median_top_sv_in=float(np.median([r["sv_in"][0] for r in rows]))))
    R2 = {}
    for f in ("L", "G", "C"):
        R2[f] = {kk: float(np.median([r["fam"][f][kk] for r in rows])) for kk in ("inspan", "gain", "topV", "c0", "c1")}
    R2["L_minus_G_gain_ci"] = ci([r["fam"]["L"]["gain"] - r["fam"]["G"]["gain"] for r in rows])
    R2["L_minus_G_inspan_ci"] = ci([r["fam"]["L"]["inspan"] - r["fam"]["G"]["inspan"] for r in rows])
    R2["L_minus_G_topV_ci"] = ci([r["fam"]["L"]["topV"] - r["fam"]["G"]["topV"] for r in rows])
    out["R2"] = R2
    json.dump(dict(summary=out, rows=rows), open(f"{run}/analysis_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
