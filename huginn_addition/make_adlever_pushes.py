"""Fixed population levers from the pulled-back readout covectors on TRAINING items (prespecified; same construction as
the L levers in make_dglever_pushes.py).
Usage: python -I huginn_addition/make_adlever_pushes.py results/<adaptcov_run>
  A_k = normalise(mean_train normalise(a_k))   adapter pullback W_s^T R_adapter^T g_k (one block, adapter path only)
  W_k = normalise(mean_train normalise(w_k))   W_s^T g_k (adapter weights, no norm chain)
  B_k = normalise(mean_train normalise(b_k))   full one-step pullback J^T g_k
Scaled to norm 3. Order: L0..L9 (copied from results/dglever_pushes.npz), A0..A9, W0..W9, B0..B9 (40 pushes).
"""
import json
import sys

import numpy as np


def nz(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main(run):
    assert json.load(open(f"{run}/analysis_summary.json"))["gate_pass"], "stage-1 gate failed"
    z = np.load(f"{run}/adaptcov.npz"); tr = z["split"] == "train"
    assert tr.sum() == 20 and all(np.isfinite(z[f]).all() for f in ("A", "W", "B"))
    Pz = np.load("results/dglever_pushes.npz"); names0 = [str(x) for x in Pz["names"]]
    L = Pz["P"][[names0.index(f"L{k}") for k in range(10)]].astype(np.float64)
    for f in ("A", "W", "B"):
        assert np.linalg.norm(z[f][tr], axis=-1).min() > 0 and np.linalg.norm(nz(z[f][tr].astype(np.float64)).mean(0), axis=-1).min() > 1e-6
    fam = {f: nz(nz(z[f][tr].astype(np.float64)).mean(0)) for f in ("A", "W", "B")}
    P = np.vstack([L] + [3.0 * fam[f] for f in ("A", "W", "B")])
    names = [f"{f}{k}" for f in ("L", "A", "W", "B") for k in range(10)]
    np.savez("results/adlever_pushes.npz", P=P.astype(np.float32), names=np.array(names))
    import hashlib
    open("results/adlever_pushes.sha256", "w").write(hashlib.sha256(open("results/adlever_pushes.npz", "rb").read()).hexdigest() + f" built_from={run}\n")
    cos = {f: float(np.mean(np.sum(nz(L) * fam[f], -1))) for f in fam}
    print(json.dumps(dict(cos_with_L=cos, norms=np.linalg.norm(P, axis=1).round(3).tolist()[::10])))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
