"""Label-free sign selection for cheap truth levers: does steering add accuracy beyond the selector? (OFFLINE, exact; no new
model runs.) Prespecified before computing anything.
Usage: python -I huginn_addition/analyze_selector.py results/<truth_stage1_run> results/<tlever_run1> results/<tlever_run3> results/<dseval_run>

WHY OFFLINE IS EXACT: the tlever runs pushed every held-out statement with the population L, B and W levers at doses
1.5625 / 3.125 / 12.5 in BOTH directions (run 1 toward the opposite label, run 3 toward the item's own label). A sign
policy chooses, per item, '+' (toward True) or '-' (toward False); the steered outcome is exactly the recorded R of the
matching condition (own if the policy's sign agrees with the item's label, else opposite). The stage-1 truth run's B-format
answer-position state at loop 64 is the same x* (stage-1 R64 equals the dseval sham R at offset 0 exactly).
DATA: eval = the 60 held-out statements (30 units) of the tlever runs; training pool = the stage-1 statements of all OTHER
units (180 statements, 90 units; cities + compare). Labels from items (int).
SELECTOR (primary, frozen on the training pool): pooled mass-mean truth probe on the answer-position state at loop 64
(stBa, checkpoint 64): direction = mean(true) - mean(false), threshold = midpoint of the class-mean projections.
POLICIES evaluated on the 60 eval statements (judgment = sign of R at offset 24 unless stated):
  MODEL           sign(R_sham)
  CALIBRATED      sign(R_sham - tau), tau = midpoint of the training pool's class-mean R64 (readout-bias calibration)
  SELECTOR        the probe's prediction reported directly
  COMBINED        logistic regression on [R64, probe score] fit on the training pool (offline combination; the best
                  linear use of the same two signals)
  STEER_F@d       push lever F in {L, B, W} (population) at dose d with sign = the selector's prediction; judgment =
                  sign(steered R)
PRIMARY (H1): does ANY steering policy (9 = 3 levers x 3 doses) have higher accuracy than SELECTOR? Paired accuracy
   difference per policy, unit-cluster bootstrap within dataset (5000; as analyze_dstar.py); one-sided bootstrap
   p = (1 + #{diff* <= 0}) / (1 + B); Holm across the 9. Label 'STEERING ADDS ACCURACY BEYOND THE SELECTOR' if any
   Holm-adjusted p < 0.05, else 'NO ADDED ACCURACY DETECTED' (not evidence of equivalence).
SECONDARY (H2): the same comparison against COMBINED (Holm across 9).
DESCRIPTIVE: accuracy of every policy (with bootstrap CIs) overall and by dataset; agreement of each steered judgment with
the selector's prediction (transmission) by dose; corrections/corruptions relative to MODEL; on-prompt collateral (off-TF
KL at 24) of the chosen conditions; the selector's training and eval accuracy/AUROC.
REVISIONS (Astra pre-analysis review; nothing computed yet): (a) this is a RETROSPECTIVE DIAGNOSTIC on statements whose
intervention results were already inspected; (b) inference: per policy a paired UNIT-LEVEL SIGN-FLIP randomisation test
(unit difference d_u = sum over the unit's statements of correct_policy - correct_reference; H0: d_u symmetric about 0;
one-sided p = (1 + #{flipped sum >= observed}) / (1 + 20000)), Holm across the 9 steering policies; bootstrap CIs are
descriptive only; H2 and H3 are exploratory; (c) COMBINED is CROSS-FITTED: training probe scores for the logistic fit are
out-of-fold (5 folds over training units, seeded), standardised with out-of-fold statistics; it is one regularised linear
combination, not an upper bound; logistic fit guarded (finite, converged, nonzero scales); (d) new baseline
SELECTOR_BIAS = sign(R_sham - tau + a (2 sel - 1)), a chosen on the training pool (out-of-fold selector predictions) from
the grid {0, 0.5, 1, 2, 4, 8, 16} by training accuracy (ties -> smallest a); H3 (exploratory): steering vs SELECTOR_BIAS,
Holm across 9; (e) gates: NPZ uid order, sham equality across runs 1/3 and with the dseval run, items equal to the dseval
run, finite arrays; state identity is supported by identical construction (seed-0 init uid + 500000, K = 64, same model
revision) and exact readout equality, not by a direct state comparison; (f) sensitivity: number of MODEL, CALIBRATED,
SELECTOR_BIAS and COMBINED predictions that change if R64 is used instead of the sham R at 24; (g) per-dataset accuracies
with bootstrap CIs.
INTERPRETATION LIMITS: steered R is approximately the sham R plus a near-constant, selector-signed shift, so steering is
expected to behave like an offline selector-signed bias; the analysis tests answer accuracy only. Effects of steering on
downstream generation (where reading out the probe cannot substitute) are NOT tested. Same eval statements and single
prompt format as the tlever runs (fresh units/formats only warranted if H1 is positive).
"""
import json
import sys

import numpy as np
from scipy.stats import rankdata

DOSES = ["1.5625", "3.125", "12.5"]


def auroc(s, y):
    r = rankdata(s); pos = y == 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def logreg(X, y, iters=100, l2=1e-4):
    X1 = np.c_[np.ones(len(X)), X]; w = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ w)); g = X1.T @ (p - y) + l2 * w
        Hm = X1.T @ (X1 * (p * (1 - p))[:, None]) + l2 * np.eye(len(w))
        w -= np.linalg.solve(Hm, g)
    return w


def main(rs, r1, r3, rd):
    z = np.load(f"{rs}/truth.npz"); P = json.load(open(f"{rs}/problems.json"))
    ev = json.load(open(f"{r1}/items.json")); assert open(f"{r3}/items.json").read() == open(f"{r1}/items.json").read()
    ck = list(z["checkpoints"]).index(64)
    X = z["stBa"][:, ck].astype(np.float64); R64 = (z["lpB"][:, 63, 0] - z["lpB"][:, 63, 1]).astype(np.float64)
    ys = np.array([int(p["truth"]) for p in P]); us = np.array([p["unit"] for p in P])
    ev_idx = np.array([int(it["stage1_index"]) for it in ev]); eu = {it["unit"] for it in ev}
    tr = np.array([u not in eu for u in us]); assert tr.sum() == 180 and not set(us[tr]) & eu
    y = np.array([int(it["truth"]) for it in ev]); assert (ys[ev_idx] == y).all()
    units = np.array([it["unit"] for it in ev]); dsn = np.array([it["dataset"] for it in ev])
    # selector
    mt, mf = X[tr & (ys == 1)].mean(0), X[tr & (ys == 0)].mean(0); dvec = mt - mf; thr = 0.5 * (mt @ dvec + mf @ dvec)
    score = X @ dvec - thr
    sel = (score[ev_idx] > 0).astype(int)
    # runs
    Z1, Z3 = np.load(f"{r1}/dstar_eval.npz"), np.load(f"{r3}/dstar_eval.npz")
    n1, n3 = [str(x) for x in Z1["names"]], [str(x) for x in Z3["names"]]
    t24 = list(Z1["ro"]).index(24); assert list(Z1["ro"]) == list(Z3["ro"])
    sham = Z1["R"][:, n1.index("sham"), t24].astype(np.float64)
    zd = np.load(f"{rd}/dstar_eval.npz"); nd = [str(x) for x in zd["names"]]
    gates = dict(uids=bool([int(u) for u in Z1["uid"]] == [int(u) for u in Z3["uid"]] == [int(it["uid"]) for it in ev]),
                 items_match_dseval=[it["uid"] for it in ev] == [it["uid"] for it in json.load(open(f"{rd}/items.json"))],
                 sham_runs=float(np.max(np.abs(Z3["R"][:, n3.index("sham"), t24] - sham))),
                 sham_dseval=float(np.max(np.abs(zd["R"][:, nd.index("sham"), list(zd["ro"]).index(24)] - sham))),
                 stage1_R64_vs_sham0=float(np.max(np.abs(R64[ev_idx] - Z1["R"][:, n1.index("sham"), 0]))),
                 finite=bool(np.isfinite(Z1["R"]).all() and np.isfinite(Z3["R"]).all() and np.isfinite(X).all()))
    assert gates["uids"] and gates["items_match_dseval"] and gates["sham_runs"] < 1e-4 and gates["sham_dseval"] < 1e-3 \
        and gates["stage1_R64_vs_sham0"] < 1e-4 and gates["finite"], gates
    tau = 0.5 * (R64[tr & (ys == 1)].mean() + R64[tr & (ys == 0)].mean())
    # out-of-fold probe scores on the training pool (5 folds over training units)
    tri = np.where(tr)[0]; tu = sorted(set(us[tr])); perm = np.random.default_rng(5).permutation(len(tu)); fold = {tu[p_]: k % 5 for k, p_ in enumerate(perm)}
    oof = np.zeros(len(tri))
    for k in range(5):
        te_ = np.array([fold[us[i]] == k for i in tri]); fi = tri[~te_]
        a_, b_ = X[fi][ys[fi] == 1].mean(0), X[fi][ys[fi] == 0].mean(0); dv = a_ - b_
        oof[te_] = X[tri[te_]] @ dv - 0.5 * (a_ @ dv + b_ @ dv)
    mu, sd = oof.mean(), oof.std(); Rm, Rs = R64[tr].mean(), R64[tr].std()
    assert sd > 0 and Rs > 0
    w = logreg(np.c_[(R64[tr] - Rm) / Rs, (oof - mu) / sd], ys[tr].astype(float))
    assert np.isfinite(w).all()
    X1 = np.c_[np.ones(len(tri)), (R64[tr] - Rm) / Rs, (oof - mu) / sd]; pp = 1 / (1 + np.exp(-X1 @ w))
    grad_norm = float(np.linalg.norm(X1.T @ (pp - ys[tr]) + 1e-4 * w)); assert grad_norm < 1e-6, grad_norm
    comb_f = lambda Rv: ((np.c_[np.ones(60), (Rv - Rm) / Rs, (score[ev_idx] - mu) / sd] @ w) > 0).astype(int)
    sel_oof = (oof > 0).astype(int)
    grid = [0, 0.5, 1, 2, 4, 8, 16]
    accs = [float(((R64[tr] - tau + a * (2 * sel_oof - 1) > 0) == ys[tr]).mean()) for a in grid]
    a_star = grid[int(np.argmax(accs))]
    sb_f = lambda Rv: (Rv - tau + a_star * (2 * sel - 1) > 0).astype(int)
    pred = dict(MODEL=(sham > 0).astype(int), CALIBRATED=(sham - tau > 0).astype(int), SELECTOR=sel, SELECTOR_BIAS=sb_f(sham), COMBINED=comb_f(sham))
    R64e = R64[ev_idx]
    sens = dict(MODEL=int(((R64e > 0).astype(int) != pred["MODEL"]).sum()), CALIBRATED=int(((R64e - tau > 0).astype(int) != pred["CALIBRATED"]).sum()),
                SELECTOR_BIAS=int((sb_f(R64e) != pred["SELECTOR_BIAS"]).sum()), COMBINED=int((comb_f(R64e) != pred["COMBINED"]).sum()))
    kl = {}
    agree = sel == y  # the selector's sign equals the item's own label -> use the '+own' condition
    for F in ("L", "B", "W"):
        for d in DOSES:
            own = Z3["R"][:, n3.index(f"{F}_pop@{d}+own"), t24]; opp = Z1["R"][:, n1.index(f"{F}_pop@{d}"), t24]
            Rst = np.where(agree, own, opp).astype(np.float64)
            pred[f"STEER_{F}@{d}"] = (Rst > 0).astype(int)
            kown = Z3["offkl"][:, n3.index(f"{F}_pop@{d}+own"), t24]; kopp = Z1["offkl"][:, n1.index(f"{F}_pop@{d}"), t24]
            kl[f"STEER_{F}@{d}"] = float(np.median(np.where(agree, kown, kopp)))
    correct = {k: (v == y).astype(float) for k, v in pred.items()}
    rng = np.random.default_rng(0); DR = []
    for _ in range(5000):
        idx = []
        for d_ in ("cities", "compare"):
            uu = np.unique(units[dsn == d_])
            for u in rng.choice(uu, len(uu)):
                idx += list(np.where(units == u)[0])
        DR.append(np.array(idx))
    ci = lambda v: [float(np.quantile([v[ix].mean() for ix in DR], q)) for q in (.025, .975)]
    out = dict(gates=gates, sensitivity_R64_vs_R24_changed_predictions=sens, selector_bias_a=a_star, selector_bias_train_acc_by_a=dict(zip(map(str, grid), accs)),
               combined_grad_norm=grad_norm, selector=dict(train_acc=float(((score[tr] > 0) == ys[tr]).mean()), train_auroc=auroc(score[tr], ys[tr]),
                             eval_acc=float((sel == y).mean()), eval_auroc=auroc(score[ev_idx], y)),
               calibration_tau=float(tau), combined_weights=w.tolist())
    def ci_ds(v, ds_):
        m = dsn == ds_
        return [float(np.quantile([v[ix][m[ix]].mean() for ix in DR], q)) for q in (.025, .975)]
    out["accuracy"] = {k: dict(acc=float(v.mean()), ci=ci(v), cities=float(v[dsn == "cities"].mean()), cities_ci=ci_ds(v, "cities"),
                               compare=float(v[dsn == "compare"].mean()), compare_ci=ci_ds(v, "compare")) for k, v in correct.items()}
    steer = [k for k in pred if k.startswith("STEER")]

    uu_all = sorted(set(units)); rs = np.random.default_rng(11); FL = rs.choice([-1.0, 1.0], size=(20000, len(uu_all)))

    def holm(ref):
        ps = {}
        for k in steer:
            dlt = correct[k] - correct[ref]
            bs = np.array([dlt[ix].mean() for ix in DR])
            du = np.array([dlt[units == u].sum() for u in uu_all]); obs = du.sum()
            p_ = float((1 + ((FL @ du) >= obs).sum()) / (1 + len(FL)))
            ps[k] = dict(diff=float(dlt.mean()), ci_descriptive=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                         n_units_nonzero=int((du != 0).sum()), p_one_sided=p_)
        order = sorted(steer, key=lambda k: ps[k]["p_one_sided"]); m = len(order); run = 0.0
        for i, k in enumerate(order):
            run = max(run, min(1.0, (m - i) * ps[k]["p_one_sided"])); ps[k]["p_holm"] = run
        return ps
    h1 = holm("SELECTOR"); h2 = holm("COMBINED"); h3 = holm("SELECTOR_BIAS")
    out["H1_vs_SELECTOR"] = dict(tests=h1, label=("STEERING ADDS ACCURACY BEYOND THE SELECTOR" if any(v["p_holm"] < 0.05 for v in h1.values())
                                                 else "NO ADDED ACCURACY DETECTED"))
    out["H2_exploratory_vs_COMBINED"] = dict(tests=h2, label=("STEERING ADDS ACCURACY BEYOND THE OFFLINE COMBINATION" if any(v["p_holm"] < 0.05 for v in h2.values())
                                                 else "NO ADDED ACCURACY DETECTED"))
    out["H3_exploratory_vs_SELECTOR_BIAS"] = dict(tests=h3, label=("STEERING ADDS ACCURACY BEYOND THE SELECTOR-SIGNED BIAS" if any(v["p_holm"] < 0.05 for v in h3.values())
                                                 else "NO ADDED ACCURACY DETECTED"))
    out["transmission_agreement_with_selector"] = {k: float((pred[k] == sel).mean()) for k in steer}
    out["corrections_corruptions_vs_MODEL"] = {k: dict(corrections=int(((pred["MODEL"] != y) & (pred[k] == y)).sum()),
                                                      corruptions=int(((pred["MODEL"] == y) & (pred[k] != y)).sum())) for k in pred if k != "MODEL"}
    out["collateral_offtf_kl_median"] = kl
    json.dump(out, open(f"{r1}/selector_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*[a.rstrip("/") for a in sys.argv[1:5]])
