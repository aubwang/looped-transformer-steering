"""Is there information in the intra-loop trajectory beyond the endpoint? (offline, existing runs; prespecified before
looking at any output of this script)
Usage: python -I huginn_addition/analyze_trajectory_info.py results/trajinfo

Classifier everywhere: logistic regression (L2, C = 1.0 fixed, no tuning), standardised features (scaler and any PCA fit on
the training fold only). Cross-validation grouped by unit (10 folds; truth: 5 folds with analyze_truth.folds_of units);
out-of-fold probabilities -> AUROC. Paired unit-bootstrap (2000 resamples of units) 95% CIs of AUROC differences.

T1 TRUTH STATES (results/truth-20261106-035631; target = statement truth; per dataset cities / compare; sites A_dot and
   B_ans; unit = city / number pair). Matched capacity:
   E40     = top-40 training-fold PCs of the loop-64 state;
   E20+S20 = top-20 PCs of the loop-64 state + top-20 PCs of the loop-t state, t in {4, 8, 16} (primary t = 8).
   INCREMENTAL if AUROC(E20+S20(8)) - AUROC(E40) has CI > 0. (Ceiling effects reported.)
T2 CORRECTNESS FROM PATHS (target = final answer correct). Feature blocks over a fixed loop grid L:
   carry_chain (results/carry_chain-20261106-024037; unit = prompt; correct = digit argmax at 96 == gold):
     E = final top-1 prob, top-1 minus top-2 prob, entropy (digit-renormalised, loop 96);
     T = log prob of the FINAL argmax digit at L = [2,4,6,8,12,16,24,32,48,64] + log drift at the answer position at L.
   census (4 shards results/census-2026110516*; unit = problem pid (both demo prompts together); correct = all answer
     positions' digit argmax at the last loop == truth):
     E = min over answer positions of the final top-1 minus top-2 prob, mean final top-1 prob, mean entropy;
     T = log drift at the last answer-prediction position and log max-over-positions drift at
         L = [4,8,16,32,64,128,256,511].
   settle (results/settle-20261106-...; unit = prompt; per answer position; correct = final argmax == truth token):
     E = top-1 log prob at loop 128;
     T = top-1 log prob, log drift, log KL at L = [2,4,6,8,12,16,24,32,48,64].
   Models: E; E+T; E+T_shuf (each item's loop order permuted, the same permutation across that item's blocks: same values
   and dimensionality, temporal order destroyed); E+T_unord (per block mean, sd, min, max: order-free summaries).
   VERDICT per dataset (revised after Astra's review, before the first run): TEMPORAL-SPECIFIC if AUROC(E+T) - AUROC(E)
   CI > 0 AND AUROC(E+T) exceeds the AUROC of all 10 time-shuffle seeds AND AUROC(E+T) - AUROC(E+T_shuf seed 0) CI > 0 AND
   AUROC(E+T) > AUROC(E+T_unord); INCREMENTAL (temporal contribution unresolved) if only the first holds; NO INCREMENT
   DETECTED otherwise. Bootstrap CIs are conditional descriptive intervals (fixed out-of-fold predictions; fitting/split
   uncertainty not included); fold class counts and a second split seed are reported (split sensitivity).
   census correctness = DIGIT-RESTRICTED argmax at every answer position == the true digit (as in analyze_census).
   T1 is interpreted as improved linear decoding over this endpoint baseline at equal feature count, not as information
   absent from the endpoint. CV folds: sklearn StratifiedGroupKFold (groups = units) for T2.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_truth import auroc, folds_of  # noqa: E402


def sgk_folds(y, units, k=10, seed=0):
    from sklearn.model_selection import StratifiedGroupKFold
    f = np.zeros(len(y), int)
    for j, (_, te) in enumerate(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, units)):
        f[te] = j
    return f


def oof_probs(X_fn, y, units, k=10, seed=0, folds=None):
    """X_fn(train_idx, test_idx) -> (Xtr, Xte). Returns out-of-fold P(y=1)."""
    f = folds if folds is not None else sgk_folds(y, units, k, seed)
    p = np.zeros(len(y))
    for j in np.unique(f):
        tr, te = np.where(f != j)[0], np.where(f == j)[0]
        Xtr, Xte = X_fn(tr, te)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Xtr), y[tr])
        p[te] = clf.predict_proba(sc.transform(Xte))[:, 1]
    return p


def boot_diff(pa, pb, y, units, B=2000, seed=1):
    rng = np.random.default_rng(seed); uu = np.unique(units); ib = {u: np.where(units == u)[0] for u in uu}
    vals = []
    for _ in range(B):
        ix = np.concatenate([ib[u] for u in rng.choice(uu, len(uu))])
        if len(np.unique(y[ix])) < 2:
            continue
        vals.append(auroc(pa[ix], y[ix]) - auroc(pb[ix], y[ix]))
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def t2(name, E, Tblocks, y, units):
    E = np.asarray(E, float); Tblocks = [np.asarray(b, float) for b in Tblocks]
    assert np.isfinite(E).all() and all(np.isfinite(b).all() for b in Tblocks), name
    n, nl = len(y), Tblocks[0].shape[1]
    T = np.concatenate(Tblocks, 1)
    Tu = np.concatenate([np.stack([b.mean(1), b.std(1), b.min(1), b.max(1)], 1) for b in Tblocks], 1)
    mk = lambda X: (lambda tr, te: (X[tr], X[te]))
    P = {k: oof_probs(mk(X), y, units) for k, X in (("E", E), ("E+T", np.c_[E, T]), ("E+T_unord", np.c_[E, Tu]))}
    shuf_auc = []
    for sd in range(10):
        rng = np.random.default_rng(100 + sd)
        perm = np.array([rng.permutation(nl) for _ in range(n)])
        Ts = np.concatenate([b[np.arange(n)[:, None], perm] for b in Tblocks], 1)
        pS = oof_probs(mk(np.c_[E, Ts]), y, units)
        if sd == 0:
            P["E+T_shuf"] = pS
        shuf_auc.append(auroc(pS, y))
    A = {k: auroc(v, y) for k, v in P.items()}
    d1, d2 = boot_diff(P["E+T"], P["E"], y, units), boot_diff(P["E+T"], P["E+T_shuf"], y, units)
    d3 = boot_diff(P["E+T_unord"], P["E"], y, units)
    temporal = d1[0] > 0 and A["E+T"] > max(shuf_auc) and d2[0] > 0 and A["E+T"] > A["E+T_unord"]
    v = "TEMPORAL-SPECIFIC" if temporal else ("INCREMENTAL (temporal contribution unresolved)" if d1[0] > 0 else "NO INCREMENT DETECTED")
    f = sgk_folds(y, units)
    split2 = {k: auroc(oof_probs(mk(X), y, units, seed=1), y) for k, X in (("E", E), ("E+T", np.c_[E, T]))}
    return dict(n=int(n), n_units=int(len(np.unique(units))), n_positive=int(y.sum()), base_rate=float(y.mean()), auroc=A,
                shuffle_auroc_10_seeds=dict(min=float(min(shuf_auc)), median=float(np.median(shuf_auc)), max=float(max(shuf_auc))),
                ET_minus_E_ci=d1, ET_minus_shuf_ci=d2, unord_minus_E_ci=d3, verdict=v,
                fold_positive_counts=[int(y[f == j].sum()) for j in range(f.max() + 1)], split_seed1_auroc=split2,
                note="CIs are conditional descriptive intervals (fixed out-of-fold predictions)")


def main(out):
    os.makedirs(out, exist_ok=True)
    res = {}
    # ---------------- T1 truth states
    R = "results/truth-20261106-035631"
    z = np.load(f"{R}/truth.npz"); P = json.load(open(f"{R}/problems.json"))
    CK = list(z["checkpoints"]); y = np.array([p["truth"] for p in P]); ds = np.array([p["dataset"] for p in P]); un = np.array([p["unit"] for p in P])
    t1 = {}
    for d in ("cities", "compare"):
        m = ds == d
        for site in ("stA", "stBa"):
            S = z[site][m].astype(np.float64); yy = y[m]; uu = un[m]; f = folds_of(uu, 5, 0)
            def pcs(X, tr, te, k):
                p = PCA(n_components=k, random_state=0).fit(X[tr]); return p.transform(X[tr]), p.transform(X[te])
            def e40(tr, te):
                return pcs(S[:, CK.index(64)], tr, te, 40)
            def es(t):
                def fn(tr, te):
                    a, b = pcs(S[:, CK.index(64)], tr, te, 20); c, d_ = pcs(S[:, CK.index(t)], tr, te, 20)
                    return np.c_[a, c], np.c_[b, d_]
                return fn
            pE = oof_probs(e40, yy, uu, folds=f)
            r = dict(E40=auroc(pE, yy))
            for t in (4, 8, 16):
                pS = oof_probs(es(t), yy, uu, folds=f)
                r[f"E20+S20(loop {t})"] = auroc(pS, yy)
                r[f"diff_ci_loop{t}"] = boot_diff(pS, pE, yy, uu)
            r["INCREMENTAL_primary_loop8"] = bool(r["diff_ci_loop8"][0] > 0)
            t1[f"{d}/{site}"] = r
    res["T1_truth_states"] = t1
    # ---------------- T2 carry_chain
    Rc = "results/carry_chain-20261106-024037"
    z = np.load(f"{Rc}/mcq.npz"); Pc = {p["uid"]: p for p in json.load(open(f"{Rc}/problems.json"))}
    assert z["done"].all()
    lp = z["ro_lp"].astype(np.float64); p_ = np.exp(lp); p_ = p_ / p_.sum(-1, keepdims=True); sched = list(z["sched"])
    gold = z["gold"]; y = (p_[:, -1].argmax(-1) == gold).astype(int)
    srt = np.sort(p_[:, -1], -1)
    E = np.c_[srt[:, -1], srt[:, -1] - srt[:, -2], -(p_[:, -1] * np.log(p_[:, -1] + 1e-12)).sum(-1)]
    fa = p_[:, -1].argmax(-1); L = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    li = [sched.index(t) for t in L]
    Tlp = np.log(p_[np.arange(len(y))[:, None], li, fa[:, None]] + 1e-12)
    q = (z["lens"] - 1).astype(int)
    Tdr = np.log(np.stack([z["drift"][i, [t - 1 for t in L], q[i]] for i in range(len(y))]) + 1e-12)
    res["T2_carry_chain"] = t2("carry_chain", E, [Tlp, Tdr], y, np.array([str(u) for u in z["uid"]]))
    # ---------------- T2 census
    Es, Ts1, Ts2, ys, us = [], [], [], [], []
    Lc = [4, 8, 16, 32, 64, 128, 256, 511]
    shards = sorted(glob.glob("results/census-20261105-16*"))
    assert len(shards) == 4, shards
    for run in shards:
        zc = np.load(f"{run}/census.npz", allow_pickle=True)
        mc = json.load(open(f"{run}/meta.json")); spd, dgt = mc["spd_tokens"], mc["dig_tokens"]
        assert (zc["steps"] == 512).all()
        for i in range(len(zc["uid"])):
            ids, pp, lpc = zc["ids"][i], zc["pred_pos"][i], zc["ro_lp"][i][-1].astype(np.float64)
            pc = np.exp(lpc); pc = pc / pc.sum(-1, keepdims=True)
            truth_tok = [ids[k + 1] for k in pp]
            truth_d = np.array([(spd if j == 0 else dgt).index(int(tk)) for j, tk in enumerate(truth_tok)])
            ys.append(int((pc.argmax(-1) == truth_d).all()))
            s_ = np.sort(pc, -1)
            Es.append([float((s_[:, -1] - s_[:, -2]).min()), float(s_[:, -1].mean()), float(-(pc * np.log(pc + 1e-12)).sum(-1).mean())])
            dr = zc["drift"][i]
            Ts1.append(np.log(dr[[t - 1 for t in Lc], pp[-1]] + 1e-12)); Ts2.append(np.log(dr[[t - 1 for t in Lc]].max(-1) + 1e-12))
            us.append(str(zc["pid"][i]))
    res["T2_census"] = t2("census", np.array(Es), [np.array(Ts1), np.array(Ts2)], np.array(ys), np.array(us))
    # ---------------- T2 settle
    Rs = "results/settle-20261105-204943" if os.path.exists("results/settle-20261105-204943") else sorted(glob.glob("results/settle-2026*"))[-1]
    assert int(np.load(f"{Rs}/settle.npz")["k_done"]) == 128
    zs = np.load(f"{Rs}/settle.npz")
    rows = zs["arn_rows"]; Ls = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    E, B1, B2, B3, yy, uu = [], [], [], [], [], []
    for (i, j, qq) in rows:
        E.append([zs["top1_lp"][i, -1, j]])
        B1.append(zs["top1_lp"][i, [t - 1 for t in Ls], j]); B2.append(np.log(zs["drift_pos"][i, [t - 1 for t in Ls], j] + 1e-12))
        B3.append(np.log(np.maximum(zs["kl"][i, [t - 1 for t in Ls], j], 1e-12)))  # loop grid starts at 2 (KL defined); clamp tiny negatives
        yy.append(int(zs["argmax"][i, -1, j] == zs["truth"][i, j])); uu.append(str(i))
    res["T2_settle"] = t2("settle", np.array(E, float), [np.array(B1, float), np.array(B2, float), np.array(B3, float)], np.array(yy), np.array(uu))
    json.dump(res, open(f"{out}/summary.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
