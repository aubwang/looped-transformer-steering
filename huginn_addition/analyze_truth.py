"""Truth experiment stage 1 (intervene_app.py `truth`). Prespecified before the run.
Usage: python -I huginn_addition/analyze_truth.py results/<truth_run>

Data: 240 statements = CITIES (60 cities x true/false country) + COMPARE (60 two-digit pairs x both orders). Held-out unit:
city / number pair (both statements of a unit are always in the same fold). States at checkpoint loops 1, 2, 4, 8, 16,
32, 64: A_dot (bare format, statement-final '.'), B_dot (judgment format, statement-final '.'), B_ans (judgment format,
answer position). Behaviour: judgment = sign of log p(' True') - log p(' False') at B_ans.
PROBE = mass-mean direction (difference of training class means), score = projection, threshold = midpoint of the two
training class-mean projections (Marks & Tegmark style). 5-fold CV over units within each dataset (seeded); AUROC and
accuracy; separation d' = (mean_true - mean_false) / pooled sd of held-out scores.
Q0 BEHAVIOUR: judgment accuracy at loop 64 per dataset (chance 50%); judgment logit difference by loop. Behaviour-linked
   analyses (Q4) are interpreted only for datasets with accuracy >= 70%.
Q1 TRUTH IS LINEARLY READABLE: held-out AUROC at A_dot, loop 64, per dataset; claim if a unit-preserving label
   permutation test (labels swapped within units, full CV refit, 200 permutations) gives p < 0.01. The unit bootstrap CI
   of held-out scores is reported as a conditional descriptive interval.
(Revised after Astra's review, before data: fold-internal fitting for every transfer/trajectory analysis; average-rank
AUROC; fold-wise score normalisation; bootstrap CIs are conditional descriptive intervals, not probe-uncertainty CIs;
Q4 measures probe/behaviour disagreement, not 'knows but says otherwise'.)
Q2 GENERALISATION: direction fit on all of one dataset, tested on the other (AUROC; threshold-free); both directions;
   and the cosine between the two datasets' mass-mean directions. Also A_dot -> B_dot transfer within dataset.
Q3 SHARPENING WITH LOOPS: held-out d' at A_dot at each checkpoint (direction fit at that loop); SHARPENS if d'(64) -
   d'(4) has a unit-bootstrap 95% CI above 0 (per dataset; the fold assignment is fixed and the bootstrap resamples the
   held-out units' scores). Also the loop-64 direction applied to earlier loops (fixed-direction trajectory).
Q4 INTERNAL VS SAID: Spearman between the held-out B_dot probe score and the loop-64 judgment logit difference; among
   items the model judges WRONG at loop 64, the fraction whose held-out B_dot probe classifies correctly ('knows but
   says otherwise'), vs the probe's accuracy on items judged right; the same at B_ans.
Late-8 drift at the three positions reported (stationarity); nothing gated on it.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.stats import spearmanr


def auroc(s, y):
    from scipy.stats import rankdata
    s, y = np.asarray(s), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))  # average ranks (ties get half credit)
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def dprime(s, y):
    a, b = s[y == 1], s[y == 0]
    return float((a.mean() - b.mean()) / np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))))


def folds_of(units, k=5, seed=0):
    uu = sorted(set(units)); rng = np.random.default_rng(seed); perm = rng.permutation(len(uu))
    fold = {uu[p]: i % k for i, p in enumerate(perm)}
    return np.array([fold[u] for u in units])


def cv_scores(X, y, units, k=5, seed=0, Xfit=None, Xapply=None):
    """Held-out mass-mean scores and predictions; folds over units. The direction and threshold are fit on training units
    of Xfit (default X) and applied to held-out units of Xapply (default X); scores are normalised by the training pooled
    sd of the training projections (fold-wise scale)."""
    Xfit = X if Xfit is None else Xfit; Xapply = X if Xapply is None else Xapply
    f = folds_of(units, k, seed)
    sc = np.zeros(len(y)); pred = np.zeros(len(y), int)
    for j in range(k):
        tr, te = f != j, f == j
        mt, mf = Xfit[tr & (y == 1)].mean(0), Xfit[tr & (y == 0)].mean(0)
        d = mt - mf
        thr = 0.5 * (mt @ d + mf @ d)
        ptr = Xfit[tr] @ d - thr
        sd = np.sqrt(0.5 * (ptr[y[tr] == 1].var(ddof=1) + ptr[y[tr] == 0].var(ddof=1)))
        sc[te] = (Xapply[te] @ d - thr) / sd
        pred[te] = (sc[te] > 0).astype(int)
    return sc, pred


def perm_test_auroc(X, y, units, n_perm=200, seed=2):
    """Unit-preserving label permutation: within each unit, flip the two labels with probability 1/2; refit the full CV."""
    obs = auroc(cv_scores(X, y, units)[0], y)
    rng = np.random.default_rng(seed); uu = sorted(set(units))
    cnt = 0
    for _ in range(n_perm):
        flip = {u: rng.random() < 0.5 for u in uu}
        yp = np.array([1 - t if flip[u] else t for t, u in zip(y, units)])
        if auroc(cv_scores(X, yp, units)[0], yp) >= obs:
            cnt += 1
    return float((cnt + 1) / (n_perm + 1))


def boot_units(fn, units, B=2000, seed=1):
    rng = np.random.default_rng(seed); uu = np.array(sorted(set(units))); idx_by = {u: np.where(units == u)[0] for u in uu}
    vals = []
    for _ in range(B):
        samp = np.concatenate([idx_by[u] for u in rng.choice(uu, len(uu))])
        v = fn(samp)
        if np.isfinite(v):
            vals.append(v)
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def main(run):
    z = np.load(f"{run}/truth.npz", allow_pickle=True)
    R = json.load(open(f"{run}/problems.json"))
    assert z["done"].all(), "incomplete run"
    CK = list(z["checkpoints"]); i64, i4 = CK.index(64), CK.index(4)
    y = np.array([r["truth"] for r in R]); ds = np.array([r["dataset"] for r in R]); units = np.array([r["unit"] for r in R])
    ld = z["lpB"][:, :, 0] - z["lpB"][:, :, 1]  # (n, K) judgment logit difference
    out = {"late_drift_median": dict(A_dot=float(np.nanmedian(z["late"][:, 0])), B_dot=float(np.nanmedian(z["late"][:, 1])),
                                     B_ans=float(np.nanmedian(z["late"][:, 2])))}
    beh = {}
    for d in ("cities", "compare"):
        m = ds == d
        beh[d] = dict(acc64=float(((ld[m, -1] > 0) == y[m]).mean()),
                      acc_by_loop={int(k): float(((ld[m, k - 1] > 0) == y[m]).mean()) for k in (1, 2, 4, 8, 16, 32, 64)},
                      tf_mass64=float(np.median(np.exp(z["lpB"][m, -1]).sum(-1))))
    out["Q0_behaviour"] = beh
    res = {}
    for d in ("cities", "compare"):
        m = ds == d; Xs = {k: z[k][m].astype(np.float64) for k in ("stA", "stBd", "stBa")}; yy = y[m]; uu = units[m]
        sc, pred = cv_scores(Xs["stA"][:, i64], yy, uu)
        auc_ci = boot_units(lambda ix: auroc(sc[ix], yy[ix]), uu)
        p_perm = perm_test_auroc(Xs["stA"][:, i64], yy, uu)
        q3 = {}
        scs = {}
        for ci, k in enumerate(CK):
            s_k, _ = cv_scores(Xs["stA"][:, ci], yy, uu); scs[k] = s_k
            q3[int(k)] = dict(dprime=dprime(s_k, yy), auroc=auroc(s_k, yy))
        dd = boot_units(lambda ix: dprime(scs[64][ix], yy[ix]) - dprime(scs[4][ix], yy[ix]), uu)
        fixed = {int(k): dprime(cv_scores(None, yy, uu, Xfit=Xs["stA"][:, i64], Xapply=Xs["stA"][:, ci])[0], yy) for ci, k in enumerate(CK)}
        a2b = auroc(cv_scores(None, yy, uu, Xfit=Xs["stA"][:, i64], Xapply=Xs["stBd"][:, i64])[0], yy)
        sBd, pBd = cv_scores(Xs["stBd"][:, i64], yy, uu); sBa, pBa = cv_scores(Xs["stBa"][:, i64], yy, uu)
        judged_right = (ld[m, -1] > 0) == yy
        q4 = dict(spearman_Bdot_vs_judgment=float(spearmanr(sBd, ld[m, -1])[0]), spearman_Bans_vs_judgment=float(spearmanr(sBa, ld[m, -1])[0]),
                  n_judged_wrong=int((~judged_right).sum()),
                  probe_Bdot_correct_when_judged_wrong=float((pBd[~judged_right] == yy[~judged_right]).mean()) if (~judged_right).any() else None,
                  probe_Bdot_correct_when_judged_right=float((pBd[judged_right] == yy[judged_right]).mean()),
                  probe_Bans_correct_when_judged_wrong=float((pBa[~judged_right] == yy[~judged_right]).mean()) if (~judged_right).any() else None,
                  interpretable=bool(beh[d]["acc64"] >= 0.7))
        jw = ~judged_right
        q4["errors_by_class"] = dict(true_judged_false=int((jw & (yy == 1)).sum()), false_judged_true=int((jw & (yy == 0)).sum()))
        res[d] = dict(Q1=dict(auroc=auroc(sc, yy), auroc_descriptive_ci95=auc_ci, perm_p=p_perm, accuracy=float((pred == yy).mean()),
                              holds=bool(p_perm < 0.01)),
                      Q3=dict(by_loop=q3, dprime_64_minus_4_descriptive_ci95=dd, sharpens_descriptive=bool(dd[0] > 0),
                              fixed_loop64_direction_dprime_heldout=fixed),
                      A_dot_to_B_dot_heldout_auroc=a2b,
                      Q4=q4, B_dot_auroc=auroc(sBd, yy), B_ans_auroc=auroc(sBa, yy))
    out["by_dataset"] = res
    # Q2 generalisation
    gen = {}
    dirs = {}
    for d in ("cities", "compare"):
        m = ds == d; X = z["stA"][m, i64].astype(np.float64)
        dirs[d] = X[y[m] == 1].mean(0) - X[y[m] == 0].mean(0)
    for a, b in (("cities", "compare"), ("compare", "cities")):
        mb = ds == b
        gen[f"{a}->{b}"] = auroc(z["stA"][mb, i64].astype(np.float64) @ dirs[a], y[mb])
    gen["cosine_cities_compare"] = float(dirs["cities"] @ dirs["compare"] / np.linalg.norm(dirs["cities"]) / np.linalg.norm(dirs["compare"]))
    out["Q2_generalisation"] = gen
    json.dump(out, open(f"{run}/analysis_summary.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
