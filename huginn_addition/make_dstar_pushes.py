"""Build held-out steering pushes for the d* evaluation (prespecified before the evaluation run).
Usage: python -I huginn_addition/make_dstar_pushes.py results/<dstar_vjp_run>
Inputs: dstar.npz (per item g = dR/dx_q and a = A^T g, R = log p(True) - log p(False), A = sum_{k<=48} J^k), items.json
(split train / eval by unit), results/dstar_mm.npy (train-unit mass-mean answer-site truth direction, true minus false).
Population directions (TRAIN items only, frozen): u_dstar = normalise(mean_i normalise(a_i)); u_g = normalise(mean_i
normalise(g_i)); u_mm = mass-mean. Item-specific (label-free, computed on the held-out item itself): u_item = normalise(a_i).
All '+' = toward TRUE (increase R to first order). Push norm c = 12.5 for every direction (matched; = the earlier
answer-site dose-1 median). Conditions per EVAL item: sham; dstar_pop +/-; g_pop +/-; mm_pop +/-; dstar_item +/-;
random + (seeded unit vector). Writes results/dstar_eval_items.json and results/dstar_pushes.npz.
"""
import json
import sys

import numpy as np

C_NORM = 12.5


def nz(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main(run):
    z = np.load(f"{run}/dstar.npz"); items = json.load(open(f"{run}/items.json"))
    split = np.array([it["split"] for it in items])
    A, G = z["A"].astype(np.float64), z["G"].astype(np.float64)
    tr, ev = split == "train", split == "eval"
    u_dstar = nz(nz(A[tr]).mean(0)); u_g = nz(nz(G[tr]).mean(0)); u_mm = nz(np.load("results/dstar_mm.npy").astype(np.float64))
    names = ["sham", "dstar_pop+", "dstar_pop-", "g_pop+", "g_pop-", "mm_pop+", "mm_pop-", "dstar_item+", "dstar_item-", "random+"]
    ev_idx = np.where(ev)[0]
    P = np.zeros((len(ev_idx), len(names), A.shape[1]), np.float32)
    rng = np.random.default_rng(123)
    for j, i in enumerate(ev_idx):
        ui = nz(A[i])
        r = rng.normal(size=A.shape[1]); r = r / np.linalg.norm(r)
        for k, (vec, sgn) in enumerate([(None, 0), (u_dstar, 1), (u_dstar, -1), (u_g, 1), (u_g, -1), (u_mm, 1), (u_mm, -1), (ui, 1), (ui, -1), (r, 1)]):
            if vec is not None:
                P[j, k] = sgn * C_NORM * vec
    cos = dict(dstar_g=float(u_dstar @ u_g), dstar_mm=float(u_dstar @ u_mm), g_mm=float(u_g @ u_mm),
               item_vs_pop_dstar_median=float(np.median(nz(A[ev]) @ u_dstar)))
    np.savez_compressed("results/dstar_pushes.npz", P=P, names=np.array(names))
    json.dump([items[i] for i in ev_idx], open("results/dstar_eval_items.json", "w"), indent=1)
    json.dump(dict(cosines=cos, n_train=int(tr.sum()), n_eval=int(ev.sum()), c_norm=C_NORM), open("results/dstar_pushes_meta.json", "w"), indent=1)
    print(json.dumps(cos), int(tr.sum()), int(ev.sum()))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
