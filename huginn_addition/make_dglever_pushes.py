"""Population pushes for the digit lever experiment (frozen on TRAINING items; prespecified).
Usage: python -I huginn_addition/make_dglever_pushes.py results/<dglever_train_run>
Families (unit vectors, then scaled to norm C_NORM = 3.0, a small dose in the linear regime):
  L_k = normalise(mean_i normalise(A_i^T grad z_k(x_i)))   loop-aware lever for raising digit k (k = 0..9)
  G_k = normalise(mean_i normalise(grad z_k(x_i)))          immediate readout gradient
  C_k = normalise(C[k] - mean_j C[j])                       representation centroid (adjusted answer-level centroids from
                                                            results/steer_dirs.npz, fit cells)
  R_0, R_1 = seeded random unit vectors.
Order: L0..L9, G0..G9, C0..C9, R0, R1 (32 pushes). Also writes the combined pred item list (eval + train).
"""
import json
import sys

import numpy as np

C_NORM = 3.0


def nz(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main(run):
    z = np.load(f"{run}/dglever.npz")
    AtG, G = z["AtG"].astype(np.float64), z["G"].astype(np.float64)
    L = nz(nz(AtG).mean(0)); Gp = nz(nz(G).mean(0))
    C = np.load("results/steer_dirs.npz")["C"].astype(np.float64); Cc = nz(C - C.mean(0))
    rng = np.random.default_rng(55); R = nz(rng.normal(size=(2, AtG.shape[-1])))
    P = C_NORM * np.vstack([L, Gp, Cc, R])
    names = [f"L{k}" for k in range(10)] + [f"G{k}" for k in range(10)] + [f"C{k}" for k in range(10)] + ["R0", "R1"]
    np.savez("results/dglever_pushes.npz", P=P.astype(np.float32), names=np.array(names))
    np.savez("results/dglever_pushes_L.npz", P=(C_NORM * L).astype(np.float32), names=np.array(names[:10]))
    ev = json.load(open("results/dglever_eval.json")); tr = json.load(open("results/dglever_train.json"))
    json.dump([dict(p, split="eval") for p in ev] + [dict(p, split="train") for p in tr], open("results/dglever_pred_items.json", "w"), indent=1)
    cos = lambda a, b: float(np.mean(np.sum(a * b, -1)))
    print(json.dumps(dict(cos_L_G=cos(L, Gp), cos_L_C=cos(L, Cc), cos_G_C=cos(Gp, Cc),
                          L_gram_offdiag_mean=float((L @ L.T)[~np.eye(10, dtype=bool)].mean()), tail_median=float(np.median(z["tail"])))))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
