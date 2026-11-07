"""Stage 1 of the adapter-pullback test (intervene_app.py `adaptcov`). Prespecified before the run.
Usage: python -I huginn_addition/analyze_adaptcov.py results/<adaptcov_run> results/<blockdecomp_run> results/<dglever_train_run>

Covectors at x* for 20 TRAIN and 20 held-out EVAL digit prompts, readouts k = 0..9: g (coda readout gradient),
a = W_s^T R_adapter^T g (adapter pullback), w = W_s^T g (no norm chain), b = J^T g (full one-step pullback); and, from the
dglever train run (train items only), the 48-step resolvent pullback r = (sum_t J^t)^T g that defines the L levers.
GATES (cross-run identities on eval items, same x*): a_k . P vs blockdecomp's adapter contribution (FD-based; median
relative < 0.02) and b_k . P vs blockdecomp's VJP c1 (median relative < 1e-3), over all 32 pushes x 10 readouts; uid
alignment asserted. (Revised per Astra, before data: the statistic is a RATIO OF MEDIANS median|diff| / median|ref|;
per-item maxima of the same ratio are also reported (adapter < 0.05, one-step < 5e-3 required); finite arrays, unique and
disjoint TRAIN/EVAL uids and complete blockdecomp output are asserted. R stability is compared only with the other
families' TRAIN LOO values.)
S1 STABILITY ACROSS STATES: population direction X_bar_k = normalise(mean_train normalise(X_k)); held-out cosine
   cos(X_bar_k, X_k(x_j)) over eval items j and k (median, item-bootstrap CI of per-item means). Also leave-one-out
   cosines within TRAIN items for g, a, w, b and r (the only comparison available for r). A state-stable adapter
   explanation predicts held-out stability of a well above that of g and comparable to or above b and r.
S2 ALIGNMENT WITH THE LEVERS: cos(X_bar_k, L_k) for X in {g, a, w, b} (mean over k). A_bar ~ L would mean the
   resolvent lever is approximately the pulled-back adapter readout.
Descriptive only; the decisive test is stage 2 (analyze_adlever.py).
"""
import json
import sys

import numpy as np


def nz(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def boot(v, B=5000, seed=0):
    v = np.asarray(v); rng = np.random.default_rng(seed)
    b = [np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(B)]
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def main(ra, rb, rt):
    z = np.load(f"{ra}/adaptcov.npz"); zb = np.load(f"{rb}/blockdecomp.npz"); zt = np.load(f"{rt}/dglever.npz")
    split = z["split"]; tr, ev = split == "train", split == "eval"
    assert (z["uid"][ev] == zb["uid"]).all() and (z["uid"][tr] == zt["uid"]).all()
    assert len(set(z["uid"])) == len(z["uid"]) and not set(z["uid"][tr]) & set(z["uid"][ev]) and tr.sum() == 20 and ev.sum() == 20
    assert int(zb["n_done_push"]) == 32 and all(np.isfinite(z[f]).all() for f in ("G", "A", "W", "B"))
    P = np.load("results/dglever_pushes.npz")["P"].astype(np.float64)
    X = {f: z[f].astype(np.float64) for f in ("G", "A", "W", "B")}
    rel = lambda a, b: float(np.median(np.abs(a - b)) / np.median(np.abs(b)))
    aP = np.einsum("ikd,pd->ipk", X["A"][ev], P); bP = np.einsum("ikd,pd->ipk", X["B"][ev], P)
    gates = dict(adapter_vs_blockdecomp_ratio_of_medians=rel(aP, zb["comp"][..., 0]), onestep_vs_blockdecomp_ratio_of_medians=rel(bP, zb["c1vjp"]),
                 adapter_worst_item=max(rel(aP[i], zb["comp"][i, ..., 0]) for i in range(aP.shape[0])),
                 onestep_worst_item=max(rel(bP[i], zb["c1vjp"][i]) for i in range(bP.shape[0])))
    gate_pass = bool(gates["adapter_vs_blockdecomp_ratio_of_medians"] < 0.02 and gates["onestep_vs_blockdecomp_ratio_of_medians"] < 1e-3
                     and gates["adapter_worst_item"] < 0.05 and gates["onestep_worst_item"] < 5e-3)
    out = dict(gates=gates, gate_pass=gate_pass)
    s1 = {}
    for f, v in X.items():
        bar = nz(nz(v[tr]).mean(0))
        c = np.sum(nz(v[ev]) * bar[None], -1)  # (n_eval, 10)
        s1[f] = dict(heldout_median=float(np.median(c)), heldout_item_mean_ci=boot(c.mean(1)))
    Xt = dict({f: v[tr] for f, v in X.items()}, R=zt["AtG"].astype(np.float64))
    for f, v in Xt.items():
        u = nz(v); ntr = len(u); loo = []
        for i in range(ntr):
            bar = nz(nz(np.delete(u, i, 0)).mean(0)); loo.append(np.sum(u[i] * bar, -1))
        s1.setdefault(f, {})["train_loo_median"] = float(np.median(loo))
    out["S1_stability"] = s1
    Pz = np.load("results/dglever_pushes.npz"); names = [str(x) for x in Pz["names"]]
    L = nz(Pz["P"][[names.index(f"L{k}") for k in range(10)]].astype(np.float64))
    out["S2_cos_with_L"] = {f: float(np.mean(np.sum(nz(nz(v[tr]).mean(0)) * L, -1))) for f, v in X.items()}
    out["S2_cos_with_L"]["R_bar(train)"] = float(np.mean(np.sum(nz(nz(Xt["R"]).mean(0)) * L, -1)))
    json.dump(out, open(f"{ra}/analysis_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*[a.rstrip("/") for a in sys.argv[1:4]])
