"""Overlap-matched controls for the cheap levers (prespecified; revised per Astra before any run).
Usage: python -I huginn_addition/make_overlap_pushes.py
From results/adlever_pushes.npz (hash-checked): unit L_k; A_k = the full adapter-path pullback family (A ~ W, cos 0.999,
but not identical) and B_k = the one-step pullback. For F in {A, B}, each k, c = cos(F_k, L_k):
  Fctrl{s}_k = 3 (c L_k + sqrt(1 - c^2) r_k), r_k isotropic Gaussian orthogonalised to L_k, unit; two independent draws
               s in {a, b} (seed 101) -> same L overlap and norm as F_k, random remainder.
  Fperp_k    = 3 normalise(F_k - (F_k . L_k) L_k) -> F_k's own remainder with its L component removed.
Two 30-push files (eval budget): ovl_pushes_1.npz = Actrla, Bctrla, Aperp; ovl_pushes_2.npz = Actrlb, Bctrlb, Bperp.
Checks (asserted on the saved float32 pushes): norms 3 +- 1e-4; control cos with L equal to F's within 1e-5; perp cos
with L < 1e-5.
"""
import hashlib
import json

import numpy as np


def nz(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main():
    assert open("results/adlever_pushes.sha256").read().split()[0] == hashlib.sha256(open("results/adlever_pushes.npz", "rb").read()).hexdigest()
    z = np.load("results/adlever_pushes.npz"); names = [str(x) for x in z["names"]]; P = z["P"].astype(np.float64)
    get = lambda f: P[[names.index(f"{f}{k}") for k in range(10)]]
    assert all(np.allclose(np.linalg.norm(get(f), axis=-1), 3.0, atol=1e-4) for f in "LAB")
    L = nz(get("L")); fam = {f: nz(get(f)) for f in ("A", "B")}; c = {f: np.sum(fam[f] * L, -1) for f in fam}
    rng = np.random.default_rng(101)
    D = {}
    for s in ("a", "b"):
        for f in ("A", "B"):
            r = rng.normal(size=L.shape); r -= np.sum(r * L, -1, keepdims=True) * L; r = nz(r)
            D[f"{f}ctrl{s}"] = 3.0 * (c[f][:, None] * L + np.sqrt(1 - c[f][:, None] ** 2) * r)
    for f in ("A", "B"):
        D[f"{f}perp"] = 3.0 * nz(fam[f] - c[f][:, None] * L)
    info = dict(cos_A_L=c["A"].round(4).tolist(), cos_B_L=c["B"].round(4).tolist())
    for i, fams in enumerate((("Actrla", "Bctrla", "Aperp"), ("Actrlb", "Bctrlb", "Bperp")), 1):
        Pn = np.vstack([D[f] for f in fams]).astype(np.float32); nm = [f"{f}{k}" for f in fams for k in range(10)]
        fn = f"results/ovl_pushes_{i}.npz"; np.savez(fn, P=Pn, names=np.array(nm))
        Q = np.load(fn)["P"].astype(np.float64)
        assert np.allclose(np.linalg.norm(Q, axis=-1), 3.0, atol=1e-4)
        for j, f in enumerate(fams):
            cq = np.sum(nz(Q[j * 10:(j + 1) * 10]) * L, -1)
            assert (np.max(np.abs(cq - c[f[0]])) < 1e-5) if "ctrl" in f else (np.max(np.abs(cq)) < 1e-5), f
        info[fn] = hashlib.sha256(open(fn, "rb").read()).hexdigest()
    json.dump(info, open("results/ovl_pushes.json", "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
