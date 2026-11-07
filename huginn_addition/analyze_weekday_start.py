"""EXPLORATORY (not prespecified; added after the weekday pilot showed Huginn does not compute multi-day offsets):
is the START-DAY representation circular? Same machinery as analyze_weekday.py geometry, with the target factor = start
day and nuisance = template + offset + answer effects (training-fold OLS, transversal folds), top-20 PCs of the
training residuals (start effects retained), held-out R^2 of circle codes under all 360 cyclic orders, categorical,
line (all 5040), 200 random rank-2 codes. States: q (answer-prediction position) at loops 8, 32, 96, and the query's
start-day token position at loop 96.
Usage: python -I huginn_addition/analyze_weekday_start.py results/<wk_run>
"""
import itertools, json, sys
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from analyze_weekday import BLK, code_from_order, cyclic_orders, design, fit, fold_of, partial, sse_of, R  # noqa: E402


def geometry(P, X, target, folds):
    lab = np.array([p[target] if target != "offset" else p["offset"] - 1 for p in P])
    orders = cyclic_orders(); canon = orders.index(tuple(range(7)))
    lin = list(itertools.permutations(range(7))); canon_lin = lin.index(tuple(range(7)))
    scale = np.linalg.norm(code_from_order(tuple(range(7))))
    rnd = []
    for i in range(200):
        c = np.random.default_rng(100 + i).normal(size=(7, 2)); c -= c.mean(0); q_, _ = np.linalg.qr(c); rnd.append(q_ * scale / np.linalg.norm(q_))
    sse = dict(c=np.zeros(len(orders)), cat=0.0, l=np.zeros(len(lin)), r=np.zeros(200)); sst = 0.0
    nuis = [k for k in ("intercept", "template", "start", "offset", "answer") if k != target]
    for f in range(7):
        tr, te = folds != f, folds == f
        Ptr = [P[i] for i in np.where(tr)[0]]; Pte = [P[i] for i in np.where(te)[0]]
        Dtr, B = fit(Ptr, X[tr])
        rtr = X[tr] - partial(Dtr, B, nuis); rte = X[te] - partial(design(Pte), B, nuis)
        mu = rtr.mean(0); V = np.linalg.svd(rtr - mu, full_matrices=False)[2][:R].T
        Ytr, Yte = (rtr - mu) @ V, (rte - mu) @ V
        a, b = lab[tr], lab[te]
        for j, o in enumerate(orders):
            c = code_from_order(o); sse["c"][j] += sse_of(c[a], Ytr, c[b], Yte)
        oh = np.eye(7)[:, :6]; sse["cat"] += sse_of(oh[a], Ytr, oh[b], Yte)
        for j, o in enumerate(lin):
            c = np.zeros((7, 1)); c[list(o), 0] = np.arange(7); sse["l"][j] += sse_of(c[a], Ytr, c[b], Yte)
        for j, c in enumerate(rnd):
            sse["r"][j] += sse_of(c[a], Ytr, c[b], Yte)
        sst += float(((Yte - Ytr.mean(0)) ** 2).sum())
    r2 = lambda e: 1 - np.asarray(e) / sst
    circ, rr = r2(sse["c"]), r2(sse["r"])
    return dict(R2_canonical_circle=round(float(circ[canon]), 4), canonical_rank_of_360=int((circ > circ[canon]).sum() + 1),
                top3=[(list(orders[j]), round(float(circ[j]), 4)) for j in np.argsort(-circ)[:3]],
                R2_categorical=round(float(r2(sse["cat"])), 4), R2_line_canonical=round(float(r2(sse["l"][canon_lin])), 4),
                R2_line_best=round(float(r2(sse["l"]).max()), 4),
                R2_random_rank2=dict(median=round(float(np.median(rr)), 4), q95=round(float(np.quantile(rr, .95)), 4), max=round(float(rr.max()), 4)))


def main(run):
    z = np.load(f"{run}/wk.npz", allow_pickle=True); P = json.load(open(f"{run}/problems.json"))
    folds = np.array([fold_of(p) for p in P])
    tr = z["traj"].astype(np.float64)
    out = {}
    for name, X in (("q_loop8", tr[:, 8]), ("q_loop32", tr[:, 32]), ("q_loop96", tr[:, 96]), ("daytoken_loop96", z["fin"][:, 1].astype(np.float64))):
        out[name] = {"start": geometry(P, X, "start", folds)}
        print(name, json.dumps(out[name]))
    json.dump(out, open(f"{run}/analysis_start_exploratory.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
