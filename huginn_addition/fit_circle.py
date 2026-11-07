"""Fit the answer-digit circle in full fp32 state space and build control directions.

Usage: python -I huginn_addition/fit_circle.py results/<collect_run>
Reads states.npz + problems.json from intervene_app.py `collect`; writes directions.npz and fit_summary.json.

1. Replication: held-out (grouped by problem) encoding comparison in the full 5280-d space — nuisance only vs number
   line vs circle vs categorical lookup — plus a scrambled-order null (same procedure as shape_analysis.py).
2. Circle plane: Frisch-Waugh — residualize states and [cos 2πd/10, sin 2πd/10] on the nuisance design, then least
   squares gives U (2 x 5280): the cos and sin coefficient vectors. One notch d -> d+1 is
   Δ = U^T [cos θ_{d+1} − cos θ_d, sin θ_{d+1} − sin θ_d].
3. Controls: U_perm for circles fitted with scrambled digit orders; random Gaussian directions; random combinations of
   the top-50 PCs of the (pos >= 1) states.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __import__("os").path.dirname(__file__))
from shape_analysis import N_FOLDS, cv_sse, digit_features, group_folds, nuisance_design, ridge_fit, ridge_pred  # noqa: E402

rng = np.random.default_rng(2026)


def main(run):
    z = np.load(f"{run}/states.npz")
    X, LP, rows = z["states"], z["digit_logp"], z["rows"]
    probs = {p["uid"]: p for p in json.load(open(f"{run}/problems.json"))}
    df = pd.DataFrame(rows, columns=["uid", "seed", "pos"])
    df["row"] = np.arange(len(df))
    recs = []
    for (uid, pos), g in df.groupby(["uid", "pos"]):
        p = probs[uid]
        di = p["digit_info"][pos]
        prev = "eq" if pos == 0 else (f"sp{p['digit_info'][0]['digit']}" if pos == 1 else f"d{p['digit_info'][pos - 1]['digit']}")
        recs.append(dict(uid=uid, pid=p["pid"], pos=pos, digit=di["digit"], col=di["col"], n_answer=len(p["digit_info"]),
                         n_digits=p["n_digits"], overflow=p["overflow"], demo=p["demo"], family=p["family"], prev=prev,
                         rows=g["row"].tolist(), correct=float(np.mean([LP[r].argmax() == di["digit"] for r in g["row"]]))))
    S = pd.DataFrame(recs)
    S = S[S["pos"] >= 1].reset_index(drop=True)
    Y = np.stack([X[r].mean(0) for r in S["rows"]]).astype(np.float64)
    d = S["digit"].to_numpy()
    groups = S["pid"].to_numpy()
    folds = group_folds(groups, N_FOLDS, seed=0)
    N = nuisance_design(S)
    sst = ((Y - Y.mean(0)) ** 2).sum()
    summary = {"rows": int(len(S)), "problems": int(S["pid"].nunique()), "dim": int(Y.shape[1]),
               "readout_accuracy_pos_ge1": float(S["correct"].mean()),
               "state_norm_mean": float(np.linalg.norm(Y, axis=1).mean())}

    # 1. replication in full space
    sse = {k: cv_sse(np.column_stack([N, digit_features(d, k)]), Y, groups, folds) for k in ("none", "line", "circle", "categorical")}
    cat_gain = sse["none"].sum() - sse["categorical"].sum()
    summary["full_space_encoding"] = {
        "nuisance_R2": float(1 - sse["none"].sum() / sst), "categorical_gain_R2": float(cat_gain / sst),
        "line_share": float((sse["none"].sum() - sse["line"].sum()) / cat_gain),
        "circle_share": float((sse["none"].sum() - sse["circle"].sum()) / cat_gain),
    }
    true_gain = sse["none"].sum() - sse["circle"].sum()
    perm_gain = np.array([sse["none"].sum() - cv_sse(np.column_stack([N, digit_features(rng.permutation(10)[d], "circle")]), Y, groups, folds).sum()
                          for _ in range(50)])
    summary["full_space_encoding"]["circle_vs_scrambled_p_50"] = float((1 + (perm_gain >= true_gain).sum()) / 51)
    print(json.dumps(summary, indent=1), flush=True)

    # 2. circle plane (exact OLS Frisch-Waugh, all rows): residual maker is idempotent (min-norm lstsq on [1, N])
    def resid(A, Nmat):
        N1 = np.column_stack([np.ones(len(Nmat)), Nmat])
        coef, *_ = np.linalg.lstsq(N1, A, rcond=None)
        return A - N1 @ coef

    Yr_base = resid(Y, N)

    def fit_plane(labels, Nmat=N, Yr=None):
        Fr = resid(digit_features(labels, "circle"), Nmat)
        Yr = Yr_base if Yr is None else Yr
        U, *_ = np.linalg.lstsq(Fr, Yr, rcond=None)
        return U, Fr  # (2, dim)
    U, Fr = fit_plane(d)
    ev = np.linalg.eigvalsh(Fr.T @ Fr / len(Fr))
    summary["residual_feature_eigs"] = [float(x) for x in ev]
    # sensitivity: plane fitted with strong nuisance (prev x pos / col / family interactions)
    from shape_sensitivity import design as strong_design
    Ns = strong_design(S, True)
    Us, _ = fit_plane(d, Ns, resid(Y, Ns))
    Qa, _ = np.linalg.qr(U.T)
    Qb, _ = np.linalg.qr(Us.T)
    summary["plane_vs_strong_nuisance_principal_cosines"] = [float(x) for x in np.linalg.svd(Qa.T @ Qb, compute_uv=False)]
    summary["plane_vs_strong_nuisance_norm_ratio"] = float(np.linalg.norm(Us) / np.linalg.norm(U))
    th = 2 * np.pi * np.arange(10) / 10
    deltas = np.stack([np.array([np.cos(th[(k + 1) % 10]) - np.cos(th[k]), np.sin(th[(k + 1) % 10]) - np.sin(th[k])]) @ U for k in range(10)])
    # natural scale: distance between nuisance-adjusted class means of neighboring digits
    R = Yr_base
    means = np.stack([R[d == k].mean(0) for k in range(10)])  # descriptive only
    nb = np.linalg.norm(means - np.roll(means, -1, 0), axis=1)
    summary["circle_plane"] = {
        "norm_u_cos": float(np.linalg.norm(U[0])), "norm_u_sin": float(np.linalg.norm(U[1])),
        "cos_angle_u_cos_u_sin": float(U[0] @ U[1] / np.linalg.norm(U[0]) / np.linalg.norm(U[1])),
        "one_notch_norm_mean": float(np.linalg.norm(deltas, axis=1).mean()),
        "one_notch_rel_to_state_norm": float(np.linalg.norm(deltas, axis=1).mean() / summary["state_norm_mean"]),
        "neighbor_class_mean_dist_mean": float(nb.mean()),
    }

    # 3. controls
    perms = []
    while len(perms) < 3:
        p = rng.permutation(10)
        # skip orders equivalent to the true cycle (rotations / reflections)
        steps = (np.roll(p, -1) - p) % 10
        if np.all(steps == steps[0]) and steps[0] in (1, 9):
            continue
        perms.append(p)
    perms = np.array(perms)
    U_perm = np.stack([fit_plane(p[d])[0] for p in perms])
    Yc = Y - Y.mean(0)
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
    pcs = Vt[:50]
    pca_dirs = np.stack([(rng.standard_normal(50) @ pcs) for _ in range(3)])
    pca_dirs /= np.linalg.norm(pca_dirs, axis=1, keepdims=True)
    rand = rng.standard_normal((3, Y.shape[1]))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    # how much of each control lies in the circle plane (should be small)
    Q, _ = np.linalg.qr(U.T)
    summary["controls_in_circle_plane_frac"] = {
        "random": [float(np.linalg.norm(Q.T @ v) ** 2) for v in rand],
        "pca": [float(np.linalg.norm(Q.T @ v) ** 2) for v in pca_dirs],
        "scr_plane_principal_cosines_with_true": [[float(x) for x in np.linalg.svd(Q.T @ np.linalg.qr(Up.T)[0], compute_uv=False)] for Up in U_perm],
    }
    np.savez(f"{run}/directions.npz", U=U.astype(np.float32), U_perm=U_perm.astype(np.float32), perms=perms,
             random=rand.astype(np.float32), pca_dirs=pca_dirs.astype(np.float32))
    json.dump(summary, open(f"{run}/fit_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
