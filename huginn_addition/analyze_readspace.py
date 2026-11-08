"""EXPLORATORY (post hoc, written after the ctrlprobe smoke runs were seen): how the loop acts on the coda's read space.
Uses only the saved Grams of ctrlprobe runs (no new GPU work). Per context, with R = the sampled read directions (task
readouts + C1 + C2 + C3; 25 for truth, 34 for digits) and P = the 16 Gaussian probes:
  1. one-step norm |J^T u| (sqrt gram3[1, 1]) for read directions vs probes: shrinking vs turning away;
  2. alpha(u) = <u, J^T u> and alpha_2(u) = <u, (J^T)^2 u> (does the signal come back after two steps?);
  3. retention: the share of |J^T u|^2 inside span(R) for u in R, against the share inside span(P) for probes and the
     isotropic benchmark rank(R) / d (numerical rank; revised after review: the centred digit gradients are dependent);
  4. compression of J^T onto span(R) (orthonormalized; from x1 and the start Gram): normalized trace, the largest
     eigenvalue of its symmetric part, and its spectral radius, against the same for span(P).
Usage: python -I analyze_readspace.py <ctrlprobe run dir> [<ctrlprobe run dir> ...]
"""
import json
import sys

import numpy as np


def compress(ug, x1, idx):
    """Orthonormal compression B = Qn^T J^T Qn of J^T onto span{u_a, a in idx}: Qn = U G^{-1/2}, so
    B = G^{-1/2} (U^T J^T U) G^{-1/2}, with (U^T J^T U)[a, b] = x1[a, b] = <u_a, J^T u_b>."""
    G = ug[np.ix_(idx, idx)]; X = x1[np.ix_(idx, idx)]
    ev, V = np.linalg.eigh(G); keep = ev > 1e-6 * ev.max()
    Gm = (V[:, keep] / np.sqrt(ev[keep])) @ V[:, keep].T
    B = Gm @ X @ Gm
    Wk = V[:, keep]; B = Wk.T @ B @ Wk                       # restrict to the numerically spanned subspace
    return B


def retention(ug, x1, n1, idx_span, idx_u):
    """share of |J^T u|^2 inside span{u_a, a in idx_span}, for u = u_b, b in idx_u."""
    G = ug[np.ix_(idx_span, idx_span)]; Gp = np.linalg.pinv(G, rcond=1e-8, hermitian=True)
    return np.array([x1[idx_span, b] @ Gp @ x1[idx_span, b] / n1[b] for b in idx_u])


def run(rd):
    z = np.load(f"{rd}/ctrlprobe.npz"); done = z["done"]; fam = z["fam"]
    g3, ug, x1 = z["gram3"][done], z["ugram"][done], z["x1"][done]
    d = z["G"].shape[-1]; n = len(g3)
    R = np.where(fam <= 3)[0]; P = np.where((fam == 4) | (fam == 5))[0]
    al = g3[:, :, 0, 1]; n1 = g3[:, :, 1, 1]; a2 = g3[:, :, 0, 2]
    out = dict(run=rd, n=n, dim_R=len(R), dim_P=len(P))
    med = lambda v: float(np.median(v))
    for name, f in (("task", fam == 0), ("C1", fam == 1), ("C2", fam == 2), ("C3", fam == 3), ("probes", (fam == 4) | (fam == 5))):
        out[name] = dict(norm1=med(np.sqrt(n1[:, f])), alpha=med(al[:, f]), alpha2=med(a2[:, f]),
                         cos1=med(al[:, f] / np.sqrt(n1[:, f])))
    retR = np.array([retention(ug[i], x1[i], n1[i], R, R) for i in range(n)])
    retP = np.array([retention(ug[i], x1[i], n1[i], P, P) for i in range(n)])
    # numerical rank with the cutoff of retention()'s pseudoinverse (rcond 1e-8): the centred digit gradients are dependent
    rk = lambda G: int(np.sum(np.linalg.eigvalsh(G) > 1e-8 * np.linalg.eigvalsh(G).max()))
    rankR = np.array([rk(ug[i][np.ix_(R, R)]) for i in range(n)])
    out["retention"] = dict(read_in_R=med(retR), probes_in_P=med(retP), rank_R=sorted(set(rankR.tolist())),
                            baseline_R=med(rankR / d), baseline_P=len(P) / d)
    cr, cp = [], []
    for i in range(n):
        for idx, acc in ((R, cr), (P, cp)):
            B = compress(ug[i], x1[i], idx)
            acc.append((np.trace(B) / len(B), np.linalg.eigvalsh((B + B.T) / 2).max(), np.abs(np.linalg.eigvals(B)).max(),
                        np.linalg.norm(B, 2)))
    cr, cp = np.array(cr), np.array(cp)
    out["compression"] = {k: dict(read=med(cr[:, j]), probes=med(cp[:, j]))
                          for j, k in enumerate(("trace_over_dim", "max_sym_eig", "spectral_radius", "op_norm"))}
    return out


if __name__ == "__main__":
    for rd in sys.argv[1:]:
        print(json.dumps(run(rd), indent=1))
