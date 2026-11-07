"""Cheap truth levers: per-item push files for the held-out truth evaluation (prespecified).
Usage: python -I huginn_addition/make_tlever_pushes.py results/<dstar_vjp_run> results/<truthcov_run>
Covectors of R = log p(True) - log p(False) at x* (loop 64): a = (sum_{k<=48} J^k)^T g (dstar_vjp, 48 VJPs), b = J^T g
(one VJP), w = W_s^T g (no VJP), g (truthcov). Population directions from TRAIN items (same construction as before):
X_pop = normalise(mean_train normalise(X_i)), X in {L := a, B := b, W := w, G := g}. Item-specific (label-free, the
held-out item's own covector): L_item = normalise(a_i), B_item = normalise(b_i). Every push points toward the item's
OPPOSITE label (true items: -X; false items: +X). Doses (push norm): 1.5625, 3.125, 12.5 (12.5 = the earlier matched dose).
Run 1: sham, L_pop, B_pop, W_pop (x3 doses) toward the OPPOSITE label = 10 conditions; run 2: sham, G_pop, L_item, B_item
(x3, opposite) + random at 12.5 (seed 123, sign toward opposite) = 11 conditions; run 3 (revision per Astra): sham, L_pop,
B_pop, W_pop (x3 doses) toward the item's OWN label (names suffixed '+own') = 10 conditions. The intervention sign uses the
ground-truth label (directions are label-free). Eval items = results/dstar_eval_items.json (order asserted). The manifest
results/tlever_pushes.json records hashes, names and the source runs.
"""
import hashlib
import json
import sys

import numpy as np

DOSES = [1.5625, 3.125, 12.5]


def nz(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main(rd, rt):
    zd, zt = np.load(f"{rd}/dstar.npz"), np.load(f"{rt}/truthcov.npz")
    items = json.load(open(f"{rd}/items.json")); items_t = json.load(open(f"{rt}/items.json"))
    assert [it["uid"] for it in items] == [it["uid"] for it in items_t] == [int(u) for u in zd["uid"]] == [int(u) for u in zt["uid"]]
    split = np.array([it["split"] for it in items]); tr = split == "train"
    ev_items = json.load(open("results/dstar_eval_items.json"))
    ev_idx = np.where(split == "eval")[0]
    assert [items[i]["uid"] for i in ev_idx] == [it["uid"] for it in ev_items]
    X = dict(L=zd["A"].astype(np.float64), B=zt["B"].astype(np.float64), W=zt["W"].astype(np.float64), G=zt["G"].astype(np.float64))
    assert all(np.isfinite(v).all() for v in X.values())
    pop = {f: nz(nz(v[tr]).mean(0)) for f, v in X.items()}
    y = np.array([int(it["truth"]) for it in ev_items]); sgn = np.where(y == 1, -1.0, 1.0)
    rng = np.random.default_rng(123)
    info = dict(cos_pop={f"{a}_{b}": float(pop[a] @ pop[b]) for a in "LBWG" for b in "LBWG" if a < b},
                item_vs_pop_median={f: float(np.median(nz(X[f][ev_idx]) @ pop[f])) for f in "LB"})
    info["sources"] = dict(dstar_vjp=rd, truthcov=rt)
    for run, fams, own in ((1, [("L_pop", None), ("B_pop", None), ("W_pop", None)], False), (2, [("G_pop", None), ("L_item", "L"), ("B_item", "B")], False),
                           (3, [("L_pop", None), ("B_pop", None), ("W_pop", None)], True)):
        names = ["sham"]; vecs = []
        sg = -sgn if own else sgn
        for nm, item in fams:
            for d in DOSES:
                names.append(f"{nm}@{d}" + ("+own" if own else ""))
                vecs.append(np.stack([sg[j] * d * (nz(X[item][i]) if item else pop[nm[0]]) for j, i in enumerate(ev_idx)]))
        if run == 2:
            r = nz(rng.normal(size=(len(ev_idx), X["L"].shape[1]))); names.append("rand@12.5")
            vecs.append(sgn[:, None] * 12.5 * r)
        P = np.concatenate([np.zeros((len(ev_idx), 1, X["L"].shape[1]))] + [v[:, None] for v in vecs], 1).astype(np.float32)
        fn = f"results/tlever_pushes_{run}.npz"; np.savez_compressed(fn, P=P, names=np.array(names))
        info[fn] = dict(sha256=hashlib.sha256(open(fn, "rb").read()).hexdigest(), names=names)
    json.dump(info, open("results/tlever_pushes.json", "w"), indent=1)
    print(json.dumps(info["cos_pop"]), info["item_vs_pop_median"])


if __name__ == "__main__":
    main(*[a.rstrip("/") for a in sys.argv[1:3]])
