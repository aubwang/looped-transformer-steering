"""Local recurrence dynamics around the step-K state, at one answer position (from intervene_app.py `jacobian`).
Usage: python -I huginn_addition/analyze_jacobian.py results/<jac_run> results/<collect_run>

- stationarity of the reference trajectory (position p and earlier positions)
- in-plane 2x2 block B = Q^T J_pp Q (column convention): eigenvalues (contraction vs rotation), max plane gain, leak,
  FD check (eps vs small eps), time variation along the reference trajectory
- Arnoldi spectrum of J_pp: spectral radius, complex fraction, leading-mode overlap with the plane, residuals (Arnoldi
  estimate and fresh FD)
- nonlinear propagation of circle +-1 pushes and a norm-matched random push: in-plane / out-of-plane decay, linearity
  (alpha 0.1 vs 1), signed in-plane rotation of the displacement in circle coordinates (+ = toward higher digits)
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
from analyze_gradients import CAT, GRID, INK, INK2  # noqa: E402  (also sets rcParams)

med = lambda x: float(np.median(x))
q = lambda x, a: float(np.percentile(x, a))


def main(run, collect_run):
    out = f"{run}/analysis"
    os.makedirs(out, exist_ok=True)
    z = np.load(f"{run}/jacobian.npz")
    U = np.load(f"{collect_run}/directions.npz")["U"].astype(np.float64)
    Qb, _ = np.linalg.qr(U.T)
    Gm = U @ U.T
    C = np.linalg.solve(Gm, U @ Qb)  # Q coords -> least-squares circle coords
    if np.linalg.det(C) < 0:
        print("note: Q frame is reflected relative to circle coords (handled by C)")
    el = z["base_correct"].astype(bool)
    n = len(el)
    S = {"rows": int(n), "eligible_rows": int(el.sum()), "problems": int(len(np.unique(z["pid"]))),
         "dir_names": z["dir_names"].tolist(), "alphas": z["alphas"].tolist()}

    # stationarity
    S["stationarity"] = {"drift_at_K_median": med(z["drift"][el]), "drift_at_K_p90": q(z["drift"][el], 90),
                         "prefix_drift_median": med(z["prefix_drift"][el]),
                         "ref_drift_median_by_step": [float(f"{x:.3g}") for x in np.median(z["ref_drift"][el], 0)[sorted({0, min(7, z["ref_drift"].shape[1] - 1), z["ref_drift"].shape[1] - 1})]],
                         "n_rows_drift_gt_1e-4_all": int((z["drift"] > 1e-4).sum()),
                         "n_problems_drift_gt_1e-4_all": int(len(np.unique(z["pid"][z["drift"] > 1e-4]))),
                         "drift_at_K_max": float(z["drift"].max())}

    # in-plane block
    B, Bs, G2 = z["B"][el], z["B_small"][el], z["JQ_gram"][el]
    lamB = np.linalg.eigvals(B)
    cpx = np.abs(lamB.imag).max(1) > 1e-6
    rot = np.abs(np.angle(lamB)).max(1)  # rad per step of the complex pair (0 if real positive)
    # signed rotation in circle coordinates: B in circle coords = C B C^-1; rotation part via polar decomposition
    Bc = np.einsum("ij,njk,kl->nil", C, B, np.linalg.inv(C))
    rot_signed = []
    for M in Bc:
        Uu, _, Vt = np.linalg.svd(M)
        R = Uu @ Vt
        rot_signed.append(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    rot_signed = np.array(rot_signed)
    S["plane_block"] = {
        "B_median": np.median(B, 0).round(4).tolist(),
        "eig_modulus_median": [med(np.abs(lamB).max(1)), med(np.abs(lamB).min(1))],
        "frac_complex_eigs": float(cpx.mean()), "rotation_deg_per_step_median_abs": float(np.degrees(np.median(rot))),
        "polar_rotation_deg_per_step_circle_coords": {"median": med(rot_signed), "p10": q(rot_signed, 10), "p90": q(rot_signed, 90)},
        "max_plane_gain_median": med(np.sqrt(np.linalg.eigvalsh(G2).max(1))),
        "leak_frac_median": med(z["leakQ"][el]),
        "fd_check_max_abs_diff_eps_vs_small": float(np.abs(B - Bs).max()),
    }
    Bck = z["B_ck"][el]
    S["plane_block"]["time_variation"] = {
        f"K+{c}": {"eig_modulus_max_median": med(np.abs(np.linalg.eigvals(Bck[:, i])).max(1)),
                   "median_abs_diff_from_K": med(np.abs(Bck[:, i] - B).max((1, 2)))} for i, c in enumerate(z["block_checkpoints"])}

    # Arnoldi spectrum
    rz = z["ritz"][el]
    rho = np.abs(rz[:, 0])
    S["spectrum"] = {
        "spectral_radius_median": med(rho), "spectral_radius_p10_p90": [q(rho, 10), q(rho, 90)],
        "frac_leading_complex": float((np.abs(rz[:, 0].imag) > 1e-6).mean()),
        "leading_rotation_deg_median": float(np.degrees(np.median(np.abs(np.angle(rz[:, 0]))))),
        "n_eigs_modulus_gt_half_median": med((np.abs(rz) > 0.5).sum(1)),
        "leading_mode_plane_overlap_median": med(z["ritz_plane_overlap"][el][:, 0]),
        "top8_plane_overlap_max_median": med(z["ritz_plane_overlap"][el].max(1)),
        "random_2d_overlap_expectation": float(np.sqrt(2 / U.shape[1])),
        "arnoldi_resid_top2_median": np.median(z["ritz_resid"][el][:, :2], 0).round(4).tolist(),
        "fresh_fd_resid_top2_median_eps_small": np.median(z["ritz_fd_resid"][el][:, :, :], 0).round(4).tolist(),
        "plane_quotient_exceeds_radius_frac": float((np.abs(lamB).max(1) > rho).mean()),
    }

    # propagation
    pq, po, push_q, pn = z["prop_q"][el], z["prop_out"][el], z["push_q"][el], z["push_norm"][el]
    T = pq.shape[3] - 1
    tt = np.arange(T + 1)
    alphas = z["alphas"]
    prop = {}
    curves = {}
    for di, dn in enumerate(z["dir_names"]):
        for ai, al in enumerate(alphas):
            n0 = al * pn[:, di]
            inpl = np.linalg.norm(pq[:, di, ai], axis=-1) / n0[:, None]
            outp = po[:, di, ai] / n0[:, None]
            tot = np.sqrt(inpl ** 2 + outp ** 2)
            # signed in-plane rotation of the displacement relative to t=0, circle coords
            a = np.einsum("ij,ntj->nti", C, pq[:, di, ai])
            ang = np.degrees(np.unwrap(np.arctan2(a[..., 1], a[..., 0]), axis=1))
            dang = ang - ang[:, :1]
            curves[(dn, float(al))] = (np.median(inpl, 0), np.median(outp, 0), np.median(tot, 0))
            k8 = min(8, T)
            rate = np.exp(np.median(np.log(np.maximum(inpl[:, k8], 1e-12)) / k8))
            prop[f"{dn}@{al:g}"] = {
                "inplane_ratio_median_t1_t4_t8_tT": np.median(inpl, 0)[[1, 4, k8, T]].round(4).tolist(),
                "outplane_ratio_median_t1_t4_t8_tT": np.median(outp, 0)[[1, 4, k8, T]].round(4).tolist(),
                "total_ratio_median_t1_t4_t8_tT": np.median(tot, 0)[[1, 4, k8, T]].round(4).tolist(),
                "inplane_geometric_rate_per_step_first8": float(rate),
                "inplane_rotation_deg_median_t1_t4_t8": np.median(dang, 0)[[1, 4, k8]].round(2).tolist(),
            }
    S["propagation"] = prop
    # linear plane-only prediction B^t q0 vs observed in-plane (circle+1, alpha 0.1)
    q0 = pq[:, 0, 0, 0]
    pred = [q0]
    for _ in range(T):
        pred.append(np.einsum("nij,nj->ni", B, pred[-1]))
    pred = np.stack(pred, 1)
    S["plane_only_linear_prediction_vs_observed_inplane_ratio_t1_t4_t8"] = {
        "predicted": np.median(np.linalg.norm(pred, axis=-1) / np.linalg.norm(q0, axis=-1)[:, None], 0)[[1, 4, 8]].round(4).tolist(),
        "observed": curves[("circle+1", float(alphas[0]))][0][[1, 4, 8]].round(4).tolist()}
    # carry split (exploratory)
    ci = z["carry_in"][el].astype(bool)
    rB = np.abs(lamB).max(1)
    S["exploratory_carry_in_split"] = {"n_carry": int(ci.sum()), "n_nocarry": int((~ci).sum()),
                                       "plane_eig_modulus_carry": med(rB[ci]) if ci.any() else None,
                                       "plane_eig_modulus_nocarry": med(rB[~ci]) if (~ci).any() else None,
                                       "spectral_radius_carry": med(rho[ci]) if ci.any() else None,
                                       "spectral_radius_nocarry": med(rho[~ci]) if (~ci).any() else None}

    # ---------------- figures
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
    ax = axs[0]
    for (dn, al), c in zip([("circle+1", float(alphas[1])), ("circle+1", float(alphas[0]))], (CAT[0], CAT[2])):
        i_, o_, t_ = curves[(dn, al)]
        ax.plot(tt, i_, color=c, label=f"circle +1 push ×{al:g}: in-plane part")
        ax.plot(tt, o_, color=c, ls="--", label=f"circle +1 push ×{al:g}: leaked out of plane")
    i_, o_, t_ = curves[("random_0", float(alphas[1]))]
    ax.plot(tt, t_, color=CAT[1], label="random push ×1: total")
    ax.plot(tt, np.median(np.linalg.norm(pred, axis=-1) / np.linalg.norm(q0, axis=-1)[:, None], 0), color=INK2, ls=":",
            label="2×2 plane block, iterated")
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1.5)
    ax.set_xlabel("loop iterations after the push"); ax.set_ylabel("displacement / initial push size (median)")
    ax.set_title("A pushed state returns: in-plane vs leaked parts"); ax.legend(fontsize=6.8)

    ax = axs[1]
    t_ = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t_), np.sin(t_), color=GRID, lw=1.2)
    r_all = rz[:, :12].ravel()
    ax.scatter(r_all.real, r_all.imag, s=5, color=CAT[0], alpha=0.25, edgecolor="none", label="leading eigenvalues of J (Arnoldi)")
    ax.scatter(lamB.real.ravel(), lamB.imag.ravel(), s=8, color=CAT[1], alpha=0.6, edgecolor="none", label="2×2 circle-plane block")
    ax.set_aspect("equal"); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("real"); ax.set_ylabel("imaginary"); ax.set_title("One loop step, linearized\n(unit circle = no decay)")
    ax.legend(fontsize=7, loc="lower left")

    ax = axs[2]
    ax.hist(rot_signed, bins=30, color=CAT[0], alpha=0.85)
    ax.axvline(0, color=INK2, lw=1, ls=":")
    ax.set_xlabel("rotation per loop step in circle plane\n(deg; + = toward higher digit)")
    ax.set_ylabel("rows"); ax.set_title("Does the loop rotate the circle plane?")
    fig.tight_layout(); fig.savefig(f"{out}/jacobian_dynamics.png", dpi=150); plt.close(fig)

    json.dump(S, open(f"{out}/summary.json", "w"), indent=1, default=float)
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/"))
