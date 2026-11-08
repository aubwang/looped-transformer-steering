"""Checks for docs/freeloop.typ (free loops: what is generic in the steering geometry).

Part A  Theorem 4 (free chain) and Corollary 4, numerically, for J = A O with O Haar-orthogonal: errors shrink with d.
Part B  Theorem 5 (self-gain gamma I + beta g g^T + A O): exact Sherman-Morrison reduction at finite d, and the limits.
Part C  Huginn tests on existing per-item arrays (no new runs). The arrays live on the Modal volume `huginn-results`:
          dstar_vjp-20261106-061708/dstar.npz, truthcov-20261106-234613/truthcov.npz,
          dglever_train-20261106-080920/dglever.npz, adaptcov-20261106-224906/adaptcov.npz
        Fetch them (one file per call; see memory note on rate limits) into $FREELOOP_DATA (default /tmp/hdata) with
          modal volume get huginn-results <run>/<file> $FREELOOP_DATA/<run>_<file>
        The Huginn numbers of Section 4 are asserted here to their displayed rounding.
Part D  The random-probe runs freeprobe-20261107-130520 (truth) and freeprobe_digits-20261107-133750 (digits): loop
        signatures, the pre-registered free-model verdicts, the readout-suppression replication, the standardized
        excess (Proposition R), the Theorem 2 partial sums, and tail diagnostics (Sections 1, 5, 6). Arrays are read from
        results/<run>/freeprobe.npz (git-ignored; fetch from the Modal volume, one file per call). analyze_freeprobe.py
        is run in a scratch directory (it writes its summary there), and its output must equal the stored
        results/<run>/analysis_summary.json; nothing under results/ is written.
Part E  Theorems 1-3 and Proposition R (random directions, spectral cancellation, Schur bound, readout excess) on
        explicit matrices and measures, including the counterexamples quoted in Section 2.
Part F  The synthetic calibration of Section 5, read from results/synth_calibration/calibration.json (written by
        huginn_addition/calibrate_freeprobe.py, CPU only, about 15 minutes; not rerun here).
Numbers in the note that are not asserted: the 2026-11-07 run costs (GPU-minutes), and quotations from analytic.pdf.
Usage: python -I docs/freeloop_check.py        (exit code 1 on any failure; Parts C-D are skipped, loudly, without data)
"""
import contextlib
import io
import json
import os
import sys
import tempfile

import numpy as np
from scipy.stats import spearmanr

FAIL = []


def ok(cond, msg):
    if not cond:
        FAIL.append(msg)


def haar(d, rng):
    Z = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(Z)
    return Q * np.sign(np.diag(R))


def chain(J, gh, T):
    W, w = [], gh.copy()
    for _ in range(T):
        w = J.T @ w
        W.append(w)
    return np.array(W)


def lever_stats(W, gh):
    """Finite-T statistics with g = g-hat (|g| = 1); definitions of freeloop.typ Section 1."""
    a = W.sum(0); w1 = W[0]
    al = gh @ w1; sg = np.sqrt(max(w1 @ w1 - al ** 2, 0.0))
    na = np.linalg.norm(a); x = gh @ a
    sig = 1 / (1 + x); psi = x * sig
    E = sig * (w1 @ a - al * x)
    return dict(alpha=al, sg=sg, Gam=na, c0=x / na, c1=w1 @ a / na, kappa=w1 @ a / (np.linalg.norm(w1) * na),
                sigma=sig, psi=psi, E=E, C2=psi - al - E)


# ---------------------------------------------------------------- Part A
def part_a():
    rng = np.random.default_rng(0)
    T = 8
    errs = {}
    for d in (500, 1000, 2000, 4000):
        e_gram, e_g, e_multi, e_cor = [], [], [], []
        for rep in range(3):
            U = haar(d, rng); V = haar(d, rng)
            sv = 0.95 * rng.beta(.6, 1.4, d) / 0.3        # ||A|| may exceed 1: Theorem 4 is finite-T
            sv = sv / np.sqrt((sv ** 2).mean()) * 0.7     # s = 0.7
            A = (U * sv) @ V.T
            s = np.sqrt((sv ** 2).mean())
            # readouts aligned with A's top left singular directions (readout-specific s_g != s)
            G = np.stack([U[:, 0] + U[:, 1], U[:, 2] - .5 * U[:, 0], rng.standard_normal(d) / np.sqrt(d)], 1)
            G /= np.linalg.norm(G, axis=0)
            O = haar(d, rng); J = A @ O
            Ws = [chain(J, G[:, j], T) for j in range(G.shape[1])]
            Bm = (A.T @ G).T @ (A.T @ G)                  # B_jk = <A^T g_j, A^T g_k>
            for j, W in enumerate(Ws):
                sg2 = Bm[j, j]
                pred = np.diag([sg2 * s ** (2 * t) for t in range(T)])
                e_gram.append(np.abs(W @ W.T - pred).max() / sg2)
                e_g.append(np.abs(W @ G[:, j]).max() / np.sqrt(sg2))
                st = lever_stats(W, G[:, j]); S = sum(s ** (2 * t) for t in range(T))
                cp = dict(c0=0, alpha=0, Gam=np.sqrt(sg2 * S), c1=np.sqrt(sg2 / S), kappa=1 / np.sqrt(S), E=sg2, C2=-sg2,
                          psi=0, sigma=1)
                e_cor.append(max(abs(st[k] - v) / max(1, abs(v)) for k, v in cp.items()))
            for j in range(3):
                for k in range(j + 1, 3):
                    for t in range(T):
                        pr = Bm[j, k] * s ** (2 * t)
                        e_multi.append(abs(Ws[j][t] @ Ws[k][t] - pr) / np.sqrt(Bm[j, j] * Bm[k, k]))
                    e_multi.append(np.abs(Ws[j] @ G[:, k]).max() / np.sqrt(Bm[j, j]))
                    aj, ak = Ws[j].sum(0), Ws[k].sum(0)
                    e_multi.append(abs(aj @ ak / np.linalg.norm(aj) / np.linalg.norm(ak) - Bm[j, k] / np.sqrt(Bm[j, j] * Bm[k, k])))
        errs[d] = [np.median(e) for e in (e_gram, e_g, e_multi, e_cor)]
        print(f"A d={d}: median errors gram {errs[d][0]:.4f}  <g,w_t> {errs[d][1]:.4f}  multi {errs[d][2]:.4f}  corollary {errs[d][3]:.4f}")
    for i, nm in enumerate(("gram", "g-orth", "multi", "corollary")):
        # O(d^-1/2): err*sqrt(d) must not grow (non-convergence would multiply it by sqrt(8) = 2.8 from 500 to 4000)
        ok(errs[4000][i] < errs[500][i] and errs[4000][i] * np.sqrt(4000) < 2 * errs[500][i] * np.sqrt(500),
           f"A {nm} error not O(d^-1/2)")
        ok(errs[4000][i] < 0.06, f"A {nm} error at d=4000 too large: {errs[4000][i]}")
    # spectral radius ~ s (single ring), and the T = infinity formulas with ||A|| < 1
    d = 2000
    U = haar(d, rng); V = haar(d, rng)
    sv = rng.uniform(.2, .95, d); A = (U * sv) @ V.T; s = np.sqrt((sv ** 2).mean())
    J = A @ haar(d, rng)
    rho = np.abs(np.linalg.eigvals(J)).max()
    print(f"A spectral radius {rho:.3f} vs s {s:.3f}")
    ok(abs(rho - s) < 0.05 and (f"{rho:.3f}", f"{s:.3f}") == ("0.621", "0.614"), f"A rho ~ s {rho} {s}")
    gh = U[:, 0]; sg = sv[0]
    a = np.linalg.solve(np.eye(d) - J.T, J.T @ gh)
    Gp = sg / np.sqrt(1 - s ** 2)
    ok(abs(np.linalg.norm(a) - Gp) / Gp < 0.05 and abs(gh @ a) / np.linalg.norm(a) < 0.05
       and abs((J.T @ gh) @ a / (np.linalg.norm(J.T @ gh) * np.linalg.norm(a)) - np.sqrt(1 - s ** 2)) < 0.03,
       "A infinite-T corollary")


# ---------------------------------------------------------------- Part B
def thm2_pred(gam, beta, s, sg):
    lam = gam + beta
    F2 = sg ** 2 / ((1 - gam) ** 2 - s ** 2)
    D = np.sqrt(lam ** 2 + F2); sgn = np.sign(1 - lam)
    return dict(alpha=lam, sigma=1 - lam, psi=lam, f=np.sqrt(F2), E=sg ** 2 / (1 - gam), EC=0.0, c0=sgn * lam / D,
                Gam=D / abs(1 - lam), c1=sgn * (lam ** 2 + sg ** 2 / (1 - gam)) / D,
                kappa=sgn * (lam ** 2 + sg ** 2 / (1 - gam)) / (np.sqrt(lam ** 2 + sg ** 2) * D))


def exact_stats(J, gh):
    d = len(gh)
    a = np.linalg.solve(np.eye(d) - J.T, J.T @ gh)
    w1 = J.T @ gh; al = gh @ w1; na = np.linalg.norm(a); x = gh @ a
    sig = 1 / (1 + x); psi = x * sig
    P = np.eye(d) - np.outer(gh, gh)
    E = sig * (w1 @ P @ a)
    return dict(alpha=al, sigma=sig, psi=psi, f=abs(sig) * np.linalg.norm(P @ a), E=E, EC=psi - al, c0=x / na, Gam=na,
                c1=w1 @ a / na, kappa=w1 @ a / (np.linalg.norm(w1) * na))


def part_b():
    rng = np.random.default_rng(1)
    # exact finite-d Sherman-Morrison reduction, for arbitrary (not random) matrices
    for trial in range(200):
        d = rng.integers(3, 9)
        A = rng.standard_normal((d, d)) * .25; O = haar(d, rng)
        gh = rng.standard_normal(d); gh /= np.linalg.norm(gh)
        gam, beta = rng.uniform(-.3, .3), rng.uniform(-.3, .3)
        J = gam * np.eye(d) + beta * np.outer(gh, gh) + A @ O
        if np.max(np.abs(np.linalg.eigvals(A @ O / (1 - gam)))) >= 1:
            continue
        Rp = np.linalg.inv(np.eye(d) - A @ O / (1 - gam))
        bp = beta / (1 - gam); r = gh @ Rp @ gh
        lhs = np.linalg.solve(np.eye(d) - J.T, gh)
        rhs = Rp.T @ gh / ((1 - gam) * (1 - bp * r))
        ok(np.allclose(lhs, rhs, rtol=1e-9, atol=1e-11), f"B sherman-morrison {trial}")
        ok(abs(gh @ np.linalg.solve(np.eye(d) - J, gh) - r / ((1 - gam) * (1 - bp * r))) < 1e-9 * max(1, abs(r)),
           f"B green {trial}")
    # limits of Theorem 5
    errs = {}
    for d in (500, 2000):
        e = []
        for gam, beta in [(0, 0), (0, .15), (.1, 0), (.1, -.2), (-.15, .25)]:
            U = haar(d, rng); V = haar(d, rng)
            sv = rng.uniform(.1, .9, d) * (1 - abs(gam)) * .95; A = (U * sv) @ V.T   # ||A|| < 1 - |gamma|
            s = np.sqrt((sv ** 2).mean())
            gh = (U[:, 0] + U[:, 3]) / np.sqrt(2)
            sg = np.linalg.norm(A.T @ gh)
            J = gam * np.eye(d) + beta * np.outer(gh, gh) + A @ haar(d, rng)
            X, P = exact_stats(J, gh), thm2_pred(gam, beta, s, sg)
            e.append(max(abs(X[k] - P[k]) / max(1, abs(P[k])) for k in P))
        errs[d] = np.median(e)
        print(f"B d={d}: Theorem 5 median max-relative error {errs[d]:.4f}")
    ok(errs[2000] < errs[500] and errs[2000] < 0.03, "B Theorem 5 limits")
    # boundary branches at d = 2000: lambda > 1 (sigma < 0), s_g = 0 (readout in the kernel of A^T), A = 0 (s = s_g = 0)
    d = 2000
    for name, gam, beta in [("lambda>1", .1, 1.2), ("s_g=0", .1, .2), ("A=0", .1, .2), ("lambda<0", .05, -.3)]:
        U = haar(d, rng); V = haar(d, rng)
        sv = rng.uniform(.1, .9, d) * .85; gh = U[:, 0]
        if name == "s_g=0":
            sv[0] = 0.0
        if name == "A=0":
            sv[:] = 0.0
        A = (U * sv) @ V.T; s = np.sqrt((sv ** 2).mean()); sg = np.linalg.norm(A.T @ gh)
        J = gam * np.eye(d) + beta * np.outer(gh, gh) + A @ haar(d, rng)
        X, P = exact_stats(J, gh), thm2_pred(gam, beta, s, sg)
        err = max(abs(X[k] - P[k]) / max(1, abs(P[k])) for k in P)
        print(f"B boundary {name}: max-relative error {err:.4f} (sigma {X['sigma']:.3f})")
        ok(err < 0.05, f"B boundary {name} {err}")
        if name == "lambda>1":
            ok(X["sigma"] < 0 and X["f"] > 0, "B lambda>1 has sigma < 0 and |f| > 0")


# ---------------------------------------------------------------- Part C
def load(data):
    f = lambda n: os.path.join(data, n)
    D = np.load(f("dstar_vjp-20261106-061708_dstar.npz")); C = np.load(f("truthcov-20261106-234613_truthcov.npz"))
    ok((D["uid"] == C["uid"]).all() and np.array_equal(D["G"], C["G"]), "C truth runs align")
    truth = [x.astype(np.float64) for x in (D["G"], D["A"], C["B"])]
    t = np.load(f("dglever_train-20261106-080920_dglever.npz")); z = np.load(f("adaptcov-20261106-224906_adaptcov.npz"))
    idx = {u: i for i, u in enumerate(z["uid"])}
    ok(all(u in idx for u in t["uid"]), "C digit items found")
    ai = [idx[u] for u in t["uid"]]
    gdiff = np.linalg.norm(t["G"] - z["G"][ai], axis=2) / np.linalg.norm(t["G"], axis=2)
    ok(np.median(gdiff) < 1e-5 and f"{np.median(gdiff):.0e}" == "6e-07", f"C digit readout gradients agree {np.median(gdiff)}")
    digits = [x.astype(np.float64) for x in (t["G"], t["AtG"], z["B"][ai])]   # (20, 10, d)
    return truth, digits


def stats(G, A, B):
    ng = np.linalg.norm(G, axis=-1); na = np.linalg.norm(A, axis=-1)
    gh = G / ng[..., None]; bh = B / ng[..., None]
    nb = np.linalg.norm(bh, axis=-1)
    al = (gh * bh).sum(-1); x = (gh * A).sum(-1) / ng
    Gam = na / ng; c0 = x / Gam; c1 = (bh * A).sum(-1) / na; kappa = c1 / nb
    sig = 1 / (1 + x); psi = x * sig
    E = sig * ((bh * A).sum(-1) / ng - al * x); C2 = psi - al - E
    return dict(nb=nb, alpha=al, sg=np.sqrt(nb ** 2 - al ** 2), c0=c0, Gam=Gam, c1=c1, kappa=kappa, psi=psi, E=E, C2=C2,
                rel=(E + C2) / E)


def q(v):
    return np.median(v), np.quantile(v, .1), np.quantile(v, .9)


def r(v, n):
    return f"{v:.{n}f}".replace("-0.", "−0.").replace("-", "−")


def part_c(data):
    truth, digits = load(data)
    out = {}
    for name, arrs in (("truth", truth), ("digits", digits)):
        S = stats(*arrs)
        flat = {k: v.ravel() for k, v in S.items()}
        out[name] = (S, flat)
        print(f"== {name}: n = {flat['alpha'].size}")
        for k in ("nb", "alpha", "c0", "Gam", "c1", "kappa", "E", "C2", "rel"):
            m, lo, hi = q(flat[k]); print(f"   {k:6s} {m: .3f} [{lo: .3f}, {hi: .3f}]")
        cp = np.corrcoef(flat["alpha"], flat["psi"])[0, 1]; sl = np.polyfit(flat["alpha"], flat["psi"], 1)[0]
        sp = spearmanr(flat["c0"], flat["alpha"] / flat["Gam"])[0]
        print(f"   psi~alpha corr {cp:.3f} slope {sl:.3f};  spearman(c0, alpha/Gamma) {sp:.3f}")
        out[name] += (cp, sl, sp)
    # displayed numbers (freeloop.typ, Section 5 table and text)
    T, Tf, tcp, tsl, tsp = out["truth"]; Dg, Df, dcp, dsl, dsp = out["digits"]
    want = {
        ("truth", "nb"): "0.71", ("truth", "alpha"): "0.12", ("truth", "c0"): "0.09", ("truth", "Gam"): "1.24",
        ("truth", "c1"): "0.48", ("truth", "kappa"): "0.68", ("truth", "E"): "0.53", ("truth", "C2"): "−0.55",
        ("digits", "nb"): "0.82", ("digits", "alpha"): "0.04", ("digits", "c0"): "0.03", ("digits", "Gam"): "1.45",
        ("digits", "c1"): "0.48", ("digits", "kappa"): "0.59", ("digits", "E"): "0.68", ("digits", "C2"): "−0.66",
    }
    for (nm, k), v in want.items():
        F = out[nm][1]
        n = 2 if k != "Gam" else 2
        ok(r(np.median(F[k]), n) == v, f"C {nm} {k}: {r(np.median(F[k]), n)} != {v}")
    for nm, F, iv in (("truth", Tf, ("−0.038", "−0.044", "−0.027")), ("digits", Df, ("0.009", "−0.055", "0.068"))):
        got = tuple(r(x, 3) for x in q(F["rel"]))
        ok(got == iv, f"C {nm} (E+2C)/E {got} != {iv}")
    ok((r(tcp, 2), r(tsl, 2), r(tsp, 2)) == ("0.90", "0.91", "0.93"), f"C truth psi/alpha {tcp, tsl, tsp}")
    ok((r(dcp, 2), r(dsl, 2), r(dsp, 2)) == ("0.87", "0.95", "0.90"), f"C digits psi/alpha {dcp, dsl, dsp}")
    # s implied by kappa = sqrt(1 - s^2)
    for nm, F, v in (("truth", Tf, "0.73"), ("digits", Df, "0.81")):
        ok(r(np.sqrt(1 - np.median(F["kappa"]) ** 2), 2) == v, f"C {nm} implied s")
    # within-item variance shares (digits: 10 readouts share one J)
    def share(v):
        w = v.var(axis=1).mean(); b = v.mean(axis=1).var(); return w / (w + b)
    sh = {k: r(share(Dg[k]), 2) for k in ("sg", "alpha", "kappa")}
    print("   digits within-item shares", sh, " within sd alpha", r(np.sqrt(Dg['alpha'].var(axis=1).mean()), 3))
    ok(sh == {"sg": "0.80", "alpha": "0.81", "kappa": "0.46"}, f"C within shares {sh}")
    ok(r(np.sqrt(Dg['alpha'].var(axis=1).mean()), 3) == "0.073", "C digits alpha within sd")
    # three-parameter leave-one-out predictions (digits); s from the other nine readouts' kappa
    P = {k: np.zeros_like(Dg["Gam"]) for k in ("Gam", "c0", "c1")}
    for i in range(Dg["Gam"].shape[0]):
        for k in range(10):
            s2 = 1 - np.median(np.delete(Dg["kappa"][i], k)) ** 2
            lam, sg = Dg["alpha"][i, k], Dg["sg"][i, k]
            p = thm2_pred(0.0, lam, np.sqrt(s2), sg)
            P["Gam"][i, k], P["c0"][i, k], P["c1"][i, k] = p["Gam"], p["c0"], p["c1"]
    loo = {k: (r(spearmanr(P[k].ravel(), Dg[k].ravel())[0], 2), r(np.median(np.abs(np.log(Dg[k] / P[k]))) if k == "Gam"
               else np.median(np.abs(Dg[k] - P[k])), 3)) for k in P}
    print("   digits LOO (spearman, median error)", loo)
    ok(loo == {"Gam": ("0.69", "0.114"), "c0": ("0.91", "0.015"), "c1": ("0.73", "0.029")}, f"C LOO {loo}")
    # several readouts: lever cosines vs one-step pullback cosines
    G, A, B = digits
    nrm = lambda X: X / np.linalg.norm(X, axis=2)[..., None]
    off = ~np.eye(10, dtype=bool)
    CA = np.einsum("ijd,ikd->ijk", nrm(A), nrm(A))[:, off].ravel(); CB = np.einsum("ijd,ikd->ijk", nrm(B), nrm(B))[:, off].ravel()
    mr = (r(np.corrcoef(CB, CA)[0, 1], 2), r(np.polyfit(CB, CA, 1)[0], 2))
    print("   digits cos(a_j,a_k) ~ cos(b_j,b_k): corr, slope", mr)
    ok(mr == ("0.83", "0.76"), f"C multi-readout {mr}")
    # one-VJP prediction of the steering advantage 1/c0 for truth, with s = rho = 0.77 (ownspec median)
    lam, sg = Tf["alpha"], Tf["sg"]
    adv = np.sqrt(lam ** 2 + sg ** 2 / (1 - .77 ** 2)) / np.abs(lam)
    av = tuple(r(x, 1) for x in q(adv)); obs = r(np.median(1 / Tf["c0"]), 1)
    print("   truth one-VJP advantage (rho=0.77)", av, " observed 1/c0", obs, " behavioural 8.1/0.93 =", r(8.1 / .93, 1))
    ok(av == ("9.1", "8.8", "9.3") and obs == "11.1" and r(8.1 / .93, 1) == "8.7", f"C advantage {av} {obs}")
    ok(np.all(Tf["c0"] > 0) and np.all(Tf["alpha"] > 0) and r(np.mean(Df["c0"] < 0), 2) == "0.29", "C c0 signs")
    agree = np.mean(np.sign(Df["c0"]) == np.sign(Df["alpha"]))
    print(f"   digits sign(c0) == sign(alpha): {agree:.3f}")
    ok(r(agree, 3) == "0.895", f"C digits sign agreement {agree}")
    # analytic.pdf's proposed regime on negative-start digit readouts (finite-T reconstruction; C via the identity)
    neg = Df["c0"] < 0
    xx = Df["psi"] / (1 - Df["psi"]); sg_ = 1 / (1 + xx)
    Dd = np.sqrt(Df["psi"] ** 2 + (Df["Gam"] * np.sqrt(1 - Df["c0"] ** 2) * np.abs(sg_)) ** 2)
    conds = dict(sigma=sg_ > 0, E=Df["E"] > 0, alphaE=Df["alpha"] + Df["E"] > 0, c1=Df["c1"] > 0,
                 cor3a=np.abs(Df["alpha"] * Df["c0"]) < Df["E"] / Dd)
    allc = np.all([v[neg] for v in conds.values()], 0)
    print(f"   digits negative starts {neg.sum()}: all regime conditions {allc.sum()}; alpha<0 {np.sum(Df['alpha'][neg] < 0)}"
          f"; alpha<0 among positive starts {np.sum(Df['alpha'][~neg] < 0)}/{np.sum(~neg)}")
    ok((neg.sum(), allc.sum(), np.sum(Df["alpha"][neg] < 0), np.sum(Df["alpha"][~neg] < 0)) == (59, 59, 53, 15),
       "C digits regime counts")
    # spectral radius estimates quoted from run ownspec (addition prompts, 40 positions, leading Ritz modulus)
    import json
    here = os.path.dirname(os.path.abspath(__file__))
    S = json.load(open(os.path.join(here, "..", "results", "ownspec-20261105-182425", "analysis_summary.json")))
    m = np.array([np.median([abs(complex(x)) for x in row["lead64"]]) for row in S["rows"]])   # median over 3 starts
    ok((len(m), r(np.median(m), 2), r(m.min(), 2), r(m.max(), 2)) == (40, "0.77", "0.66", "0.87"), f"C ownspec {np.median(m)}")


# ---------------------------------------------------------------- Part D
RUNS = dict(truth="freeprobe-20261107-130520", digits="freeprobe_digits-20261107-133750")


def part_d(data):
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "huginn_addition"))
    import analyze_freeprobe as af
    truth, digits = load(data)
    res = {}
    for name, run, NR, ref in (("truth", RUNS["truth"], 1, "dstar_vjp-20261106-061708_dstar.npz"),
                               ("digits", RUNS["digits"], 2, "dglever_train-20261106-080920_dglever.npz")):
        rd = os.path.join(here, "..", "results", run)
        tmp = tempfile.mkdtemp(prefix="freeloop_check_")
        for f in ("freeprobe.npz", "meta.json", "items.json"):
            os.symlink(os.path.abspath(os.path.join(rd, f)), os.path.join(tmp, f))
        with contextlib.redirect_stdout(io.StringIO()):
            out = af.main(tmp, os.path.join(data, ref))
        fresh = json.load(open(os.path.join(tmp, "analysis_summary.json"))); fresh.pop("run")
        stored = json.load(open(os.path.join(rd, "analysis_summary.json"))); stored.pop("run")
        ok(fresh == stored, f"D {name}: stored analysis_summary.json differs from a fresh analysis")
        ok(out["gates"]["PASS"], f"D {name} gates")
        z = np.load(os.path.join(rd, "freeprobe.npz")); GR = z["gram"]; T, K = 48, 4
        gam = np.array([[GR[i, m * (T + 1), m * (T + 1) + 1] for m in range(NR, NR + K)] for i in range(len(GR))]).mean(1)
        c0p = np.array([[af.start_stats(GR[i], m, T)["c0"] for m in range(NR, NR + K)] for i in range(len(GR))])
        Ep = np.array([[af.start_stats(GR[i], m, T)["E"] + af.start_stats(GR[i], m, T)["C2"] for m in range(NR, NR + K)]
                       for i in range(len(GR))]).mean(1)
        advr = 1 / np.abs(c0p).mean(1)
        sig = out["SIGNATURE"]; P1 = out["P1"]; P2 = out["P2"]
        k2 = np.array([np.mean([GR[i, m * (T + 1), m * (T + 1) + 2] for m in range(NR, NR + K)]) for i in range(len(GR))]) - gam ** 2
        phi2_share = np.median((k2 / (1 - gam)) / Ep)
        # readout suppression with ALL readouts of the existing arrays (truth: 40 of the 120 items; digits: 10 readouts)
        if name == "truth":
            D = np.load(os.path.join(data, ref)); idx = {u: i for i, u in enumerate(D["uid"])}; kk = [idx[u] for u in z["uid"]]
            St = stats(*truth); al = St["alpha"][kk][:, None]; c0g = St["c0"][kk][:, None]
        else:
            t = np.load(os.path.join(data, ref)); ok((t["uid"] == z["uid"]).all(), "D digits uid alignment")
            St = stats(*digits); al = St["alpha"]; c0g = St["c0"]
        exc = af.boot_median(al - gam[:, None]); frac = np.mean(al < gam[:, None])
        ratio = af.boot_median((1 / np.abs(c0g)) / advr[:, None])
        res[name] = dict(
            gamma=r(sig["trace_moments_m1_to_m8"][0][0], 3), m2=r(sig["trace_moments_m1_to_m8"][1][0], 3),
            m3=r(sig["trace_moments_m1_to_m8"][2][0], 3), m4=r(sig["trace_moments_m1_to_m8"][3][0], 3),
            kappa2=r(sig["kappa2"][0], 3), ratio_lo=r(min(sig["norm_ratio_t2_to_t10"]), 2), ratio_hi=r(max(sig["norm_ratio_t2_to_t10"]), 2),
            lag1_t1=r(sig["lag1_cos_t_1_2_3_5_10_20"][0], 2), lag1_t20=r(sig["lag1_cos_t_1_2_3_5_10_20"][5], 2),
            adv_rand=r(sig["random_adv"][0], 2), Gam_rand=r(sig["random_Gamma"][0], 2), c0_rand=r(sig["random_c0"][0], 2),
            cancel=r(P1["descr"]["probe_cancel_signed"][0], 3), P1b=r(P1["P1b_logGamma"], 2), P1c=r(P1["P1c_profile"], 2),
            P1d=r(P1["P1d_sd_kappa"], 3), P1e=r(P1["P1e_cos"], 3), verdict=P1["verdict"], P2_Gam=r(P2["Gamma"]["med_err"], 2),
            exc=tuple(r(v, 3) for v in exc), frac=r(frac, 2), n_read=int(al.size), adv_ratio=tuple(r(v, 2) for v in ratio),
            phi2_share=r(phi2_share, 2), s_hat=r(out["Q3"]["s_hat"][0], 2))
        print(f"== D {name}:", res[name])
    want = dict(
        truth=dict(gamma="0.242", m2="0.037", m3="−0.005", m4="−0.008", kappa2="−0.021", ratio_lo="0.71", ratio_hi="0.73",
                   lag1_t1="0.30", lag1_t20="0.54", adv_rand="4.06", Gam_rand="1.07", c0_rand="0.25", cancel="−0.098",
                   P1b="0.46", P1c="0.45", P1d="0.019", P1e="0.037", verdict="NOT SUPPORTED", P2_Gam="0.31",
                   exc=("−0.121", "−0.123", "−0.117"), frac="1.00", n_read=40, adv_ratio=("2.75", "2.71", "2.82"),
                   phi2_share="0.81", s_hat="0.59"),
        digits=dict(gamma="0.256", m2="0.047", m3="−0.003", m4="−0.008", kappa2="−0.019", ratio_lo="0.72", ratio_hi="0.75",
                    lag1_t1="0.35", lag1_t20="0.77", adv_rand="4.33", Gam_rand="1.23", c0_rand="0.23", cancel="−0.088",
                    P1b="0.51", P1c="0.74", P1d="0.016", P1e="0.044", verdict="NOT SUPPORTED", P2_Gam="0.37",
                    exc=("−0.213", "−0.238", "−0.198"), frac="1.00", n_read=200, adv_ratio=("4.35", "3.71", "5.90"),
                    phi2_share="0.74", s_hat="0.62"))
    for name in want:
        for k, v in want[name].items():
            ok(res[name][k] == v, f"D {name} {k}: {res[name][k]} != {v}")
    # probe medians for the readout-versus-random table (over all probes of all items)
    tab = dict(truth=("0.64", "0.24", "0.25", "1.07", "0.47", "0.73", "0.34", "−0.38"),
               digits=("0.67", "0.26", "0.23", "1.23", "0.46", "0.70", "0.39", "−0.43"))
    for name, run, NR in (("truth", RUNS["truth"], 1), ("digits", RUNS["digits"], 2)):
        GR = np.load(os.path.join(here, "..", "results", run, "freeprobe.npz"))["gram"]
        st = [af.start_stats(GR[i], m, 48) for i in range(len(GR)) for m in range(NR, NR + 4)]
        med = lambda f: np.median([f(x) for x in st])
        got = (r(med(lambda x: np.sqrt(x["n1"])), 2), r(med(lambda x: x["alpha"]), 2), r(med(lambda x: x["c0"]), 2),
               r(med(lambda x: x["Gamma"]), 2), r(med(lambda x: x["c1"]), 2), r(med(lambda x: x["kappa"]), 2),
               r(med(lambda x: x["E"]), 2), r(med(lambda x: x["C2"]), 2))
        print(f"   D {name} probe medians |J^T z|, alpha, c0, Gamma, c1, kappa, E, 2C: {got}")
        ok(got == tab[name], f"D {name} probe table {got}")
    # where the shifted free model fails: chain correlation and decay, observed vs predicted from (gamma, s) at step 1
    from math import comb
    want_chain = dict(truth=("0.30", "0.63", "0.36", "0.94", "0.457", "0.456", "0.07", "0.11"),
                      digits=("0.35", "0.63", "0.69", "0.94", "0.504", "0.489", "0.08", "0.16"))
    for name, run, NR in (("truth", RUNS["truth"], 1), ("digits", RUNS["digits"], 2)):
        GR = np.load(os.path.join(here, "..", "results", run, "freeprobe.npz"))["gram"]; T = 48
        oc1, pc1, oc10, pc10, on2, pn2, on8, pn8 = ([] for _ in range(8))
        for i in range(len(GR)):
            al = np.array([GR[i, m * (T + 1), m * (T + 1) + 1] for m in range(NR, NR + 4)])
            n1 = np.array([GR[i, m * (T + 1) + 1, m * (T + 1) + 1] for m in range(NR, NR + 4)])
            g_, s2_ = al.mean(), n1.mean() - al.mean() ** 2
            gm = lambda t, u: sum(comb(t, k) * comb(u, k) * g_ ** (t - k) * g_ ** (u - k) * s2_ ** k for k in range(min(t, u) + 1))
            for m in range(NR, NR + 4):
                o = m * (T + 1); B = GR[i, o:o + T + 1, o:o + T + 1]
                oc1.append(B[1, 2] / np.sqrt(B[1, 1] * B[2, 2])); pc1.append(gm(1, 2) / np.sqrt(gm(1, 1) * gm(2, 2)))
                oc10.append(B[10, 11] / np.sqrt(B[10, 10] * B[11, 11])); pc10.append(gm(10, 11) / np.sqrt(gm(10, 10) * gm(11, 11)))
                on2.append(np.sqrt(B[2, 2])); pn2.append(np.sqrt(gm(2, 2))); on8.append(np.sqrt(B[8, 8])); pn8.append(np.sqrt(gm(8, 8)))
        got = (r(np.median(oc1), 2), r(np.median(pc1), 2), r(np.median(oc10), 2), r(np.median(pc10), 2),
               r(np.median(pn2), 3), r(np.median(on2), 3), r(np.median(on8), 2), r(np.median(pn8), 2))
        print(f"   D {name} chain: cos(w1,w2) obs/pred, cos(w10,w11) obs/pred, |w2| pred/obs, |w8| obs/pred: {got}")
        ok(got == want_chain[name], f"D {name} chain {got}")
    # truth decomposition of the advantage ratio: (x_rand / x_g) (Gamma_g / Gamma_rand), per item (all truth c0 > 0)
    GR = np.load(os.path.join(here, "..", "results", RUNS["truth"], "freeprobe.npz"))["gram"]
    xr = np.array([np.mean([af.start_stats(GR[i], m, 48)["c0"] * af.start_stats(GR[i], m, 48)["Gamma"] for m in range(1, 5)]) for i in range(len(GR))])
    Gr = np.array([np.mean([af.start_stats(GR[i], m, 48)["Gamma"] for m in range(1, 5)]) for i in range(len(GR))])
    sg0 = [af.start_stats(GR[i], 0, 48) for i in range(len(GR))]
    xg = np.array([x["c0"] * x["Gamma"] for x in sg0]); Gg = np.array([x["Gamma"] for x in sg0])
    dec = (r(np.median(xr / xg), 2), r(np.median(Gg / Gr), 2))
    print(f"   D truth advantage decomposition: x_rand/x_g {dec[0]}, Gamma_g/Gamma_rand {dec[1]}")
    ok(dec == ("2.32", "1.16"), f"D truth decomposition {dec}")


# ---------------------------------------------------------------- Part E
def part_e():
    rng = np.random.default_rng(5)
    # Theorem 1: probe statistics -> normalized traces, for a non-normal, non-free J; error shrinks with d
    errs = {}
    for d in (400, 1600):
        e = []
        for rep_ in range(3):
            J = 0.2 * np.eye(d) + rng.standard_normal((d, d)) * 0.5 / np.sqrt(d) + np.triu(rng.standard_normal((d, d)), 1) * 0.3 / np.sqrt(d)
            M = np.linalg.solve(np.eye(d) - J, J)  # J (I - J)^{-1} (they commute)
            lam = np.linalg.eigvals(J)
            tauJ, tauM, tauMM, tauJM = np.trace(J) / d, np.trace(M) / d, (M ** 2).sum() / d, np.trace(J @ M.T) / d
            ok(abs(tauM - np.mean(lam / (1 - lam)).real) < 1e-8 and abs(tauJ - lam.mean().real) < 1e-10, "E trace = eigenvalue average")
            for _ in range(4):
                zz = rng.standard_normal(d); zz /= np.linalg.norm(zz)
                w1 = J.T @ zz; a = M.T @ zz
                e.append(max(abs(zz @ w1 - tauJ) / max(1, abs(tauJ)), abs(zz @ a - tauM) / max(1, abs(tauM)),
                             abs(a @ a - tauMM) / max(1, abs(tauMM)), abs(w1 @ a - tauJM) / max(1, abs(tauJM))))
        errs[d] = np.median(e)
    print(f"E Theorem 1 median relative errors: d=400 {errs[400]:.4f}, d=1600 {errs[1600]:.4f}")
    ok(errs[1600] < errs[400] / 1.5 and errs[1600] < 0.08, "E Theorem 1 concentration (rate at least d^-1/2-like)")
    # Theorem 2: exact formula, rotation invariance, Jensen on the real line, sign of kappa_2
    def phi(lam, w):
        g = np.sum(w * lam); return 1 - 1 / np.sum(w / (1 - lam)) - g, g
    for trial in range(300):
        m = rng.integers(2, 30); w = rng.dirichlet(np.ones(m)); g0 = rng.uniform(-.5, .6)
        z_ = rng.standard_normal(m) + 1j * rng.standard_normal(m); z_ -= np.sum(w * z_)
        z_ *= rng.uniform(.05, .9) * (1 - g0) / np.abs(z_).max()
        lam = g0 + z_
        ph, g = phi(lam, w)
        Sser = sum(np.sum(w * (lam - g) ** k) / (1 - g) ** k for k in range(2, 400))
        ok(abs(ph - ((1 - g) * Sser / (1 + Sser))) < 1e-9 * max(1, abs(ph)), f"E Theorem 2 exact formula {trial}")
        lr = g0 + rng.uniform(-.9, .9, m) * (1 - g0) * .9
        ok(phi(lr.astype(complex), w)[0].real >= -1e-12, f"E Theorem 2b Jensen {trial}")
    ring = 0.3 + 0.5 * np.exp(2j * np.pi * np.arange(64) / 64); ok(abs(phi(ring, np.ones(64) / 64)[0]) < 1e-8, "E Theorem 2a (64-fold symmetric: kappa_k = 0 for k < 64)")
    ok(abs(phi(np.array([0.3 + 0j]), np.ones(1))[0]) < 1e-15, "E point mass")
    ph2, g2 = phi(np.array([1.9, 2.1]), np.array([.5, .5]))  # Theorem 2(3) remark: for gamma > 1 the sign flips
    ok(abs(g2 - 2) < 1e-12 and abs(ph2 + 0.01) < 1e-12, f"E Theorem 2(3) gamma > 1 example Phi {ph2}")
    ok(phi(np.array([0.3 + .5j, 0.3 - .5j]), np.ones(2) / 2)[0].real < 0, "E vertical pair negative")
    ok(phi(np.array([-0.2 + 0j, 0.8 + 0j]), np.ones(2) / 2)[0].real > 0, "E horizontal pair positive")
    # Theorem 3: Schur bound, equality for normal J
    d = 300
    for kind in ("normal", "nonnormal"):
        Qm = haar(d, rng)
        if kind == "normal":
            Tm = np.zeros((d, d))
            for k in range(0, d, 2):
                a_, b_ = rng.uniform(-.4, .4), rng.uniform(-.4, .4); Tm[k:k + 2, k:k + 2] = [[.2 + a_, b_], [-b_, .2 + a_]]
        else:
            Tm = np.triu(rng.standard_normal((d, d)) * .4 / np.sqrt(d), 1) + np.diag(rng.uniform(-.3, .6, d))
        J = Qm @ Tm @ Qm.T
        M = np.linalg.solve(np.eye(d) - J, J); nu = np.linalg.eigvals(M)
        lhs = (M ** 2).sum() / d; rhs = np.mean(np.abs(nu) ** 2)
        c0lim = (np.trace(M) / d) / np.sqrt(lhs); c0nb = np.mean(nu).real / np.sqrt(rhs)
        print(f"E Theorem 3 {kind}: tau(MM^T) {lhs:.4f} >= mean|nu|^2 {rhs:.4f}; |c0| {abs(c0lim):.4f} <= {abs(c0nb):.4f}")
        ok(lhs >= rhs - 1e-10 and abs(c0lim) <= abs(c0nb) + 1e-10, f"E Theorem 3 {kind}")
        if kind == "normal":
            ok(abs(lhs - rhs) < 1e-8, "E Theorem 3 equality for normal J")
        else:
            ok(lhs > 1.02 * rhs, "E Theorem 3 strict for this non-normal J")


def part_d2(data):
    """Proposition R on Huginn, Theorem 2 partial sums, tails, Q3 and P2 sample sizes (Sections 1, 5, 6)."""
    from math import comb
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "huginn_addition"))
    import analyze_freeprobe as af
    truth, digits = load(data)
    T, K, d = 48, 4, 5280
    want = dict(
        truth=dict(Z=("−15.3", "−15.4", "−14.8"), Zmax="−12.4", sd_pred="0.0079", sd_obs="0.0083", sd_ratio="1.04",
                   below_all="1.00", min_def="0.095", se="0.004", k3="−0.003", ps_dev_ok=True, last_ok=True,
                   geo="9.8e-06", Q3="mixed", gam_ci=("0.240", "0.243"), n_P2=40, phi_tr="−0.034"),
        digits=dict(Z=("−25.6", "−28.5", "−24.0"), Zmax="−7.9", sd_pred="0.0083", sd_obs="0.0082", sd_ratio="0.99",
                    below_all="1.00", min_def="0.066", se="0.004", k3="−0.005", ps_dev_ok=True, last_ok=True,
                    geo="1.6e-04", Q3="mixed", gam_ci=("0.254", "0.260"), n_P2=40, phi_tr="−0.034"))
    for name, NR, ref in (("truth", 1, "dstar_vjp-20261106-061708_dstar.npz"), ("digits", 2, None)):
        rd = os.path.join(here, "..", "results", RUNS[name])
        z = np.load(os.path.join(rd, "freeprobe.npz")); GR = z["gram"]; n = len(GR); o = lambda m: m * (T + 1)
        al = np.array([[GR[i, o(m), o(m) + 1] for m in range(NR, NR + K)] for i in range(n)])
        n1 = np.array([[GR[i, o(m) + 1, o(m) + 1] for m in range(NR, NR + K)] for i in range(n)])
        mk = np.array([[[GR[i, o(m), o(m) + k] for k in range(1, 9)] for m in range(NR, NR + K)] for i in range(n)]).mean(1)
        gam = al.mean(1)
        tB2 = (mk[:, 1] + n1.mean(1)) / 2 - gam ** 2            # Proposition R: tau(B^2) from the probes
        sd = np.sqrt(2 * tB2 / (d + 2))
        if name == "truth":
            D = np.load(os.path.join(data, ref)); idx = {u: i for i, u in enumerate(D["uid"])}
            ar = stats(*truth)["alpha"][[idx[u] for u in z["uid"]]][:, None]
        else:
            ar = stats(*digits)["alpha"]
        exc = ar - gam[:, None]; Z = exc / sd[:, None]
        st = [[af.start_stats(GR[i], m, T) for m in range(NR, NR + K)] for i in range(n)]
        E2C = np.array([[x["E"] + x["C2"] for x in row] for row in st]).mean(1)
        xs = np.array([[x["c0"] * x["Gamma"] for x in row] for row in st]).mean(1)
        m_all = np.concatenate([np.ones((n, 1)), mk], 1)
        kap = np.array([[sum(comb(k, j) * m_all[i, j] * (-gam[i]) ** (k - j) for j in range(k + 1)) for k in range(9)]
                        for i in range(n)])
        share = {}
        for Kmax in range(2, 9):
            S = sum(kap[:, k] / (1 - gam) ** k for k in range(2, Kmax + 1))
            share[Kmax] = np.median((1 - gam) * S / (1 + S) / E2C)
        last, geo = [], []
        for i in range(n):
            for m in range(NR + K):
                x = af.start_stats(GR[i], m, T); nr = x["norms"]; rr = np.mean(nr[T - 8:T] / nr[T - 9:T - 1])
                last.append(nr[-1] / x["Gamma"]); geo.append(nr[-1] * rr / (1 - rr) / x["Gamma"])
        summ = json.load(open(os.path.join(rd, "analysis_summary.json")))
        bz = af.boot_median(Z)
        got = dict(Z=tuple(r(v, 1) for v in bz), Zmax=r(Z.max(), 1), sd_pred=r(np.sqrt((sd ** 2).mean()), 4),
                   sd_obs=r(np.sqrt(al.var(1, ddof=1).mean()), 4),
                   sd_ratio=r(np.sqrt(al.var(1, ddof=1).mean() / (sd ** 2).mean()), 2),
                   below_all=r(np.mean(ar < al.min(1)[:, None]), 2), min_def=r((-exc).min(), 3),
                   se=r(np.median(al.std(1, ddof=1) / np.sqrt(K)), 3), k3=r(np.median(kap[:, 3]), 3),
                   ps_dev_ok=bool(round(min(share[k] for k in range(3, 9)), 2) >= 0.92 and round(max(share[k] for k in range(3, 9)), 2) <= 1.08), last_ok=bool(max(last) < 3e-5),
                   geo=f"{max(geo):.1e}", Q3=summ["Q3"]["reading"],
                   gam_ci=(r(summ["Q3"]["gamma_hat"][1], 3), r(summ["Q3"]["gamma_hat"][2], 3)), n_P2=n * NR,
                   phi_tr=r(np.median(1 - 1 / (1 + xs) - gam), 3))
        print(f"== D2 {name}:", got)
        print(f"   D2 {name} partial-sum shares K=2..8:", {k: round(float(v), 3) for k, v in share.items()},
              f"max last/lever {max(last):.2e}")
        ok(r(share[2], 2) == ("0.84" if name == "truth" else "0.77"), f"D2 {name} K=2 series share {share[2]}")
        for k, v in want[name].items():
            ok(got[k] == v, f"D2 {name} {k}: {got[k]} != {v}")


def part_e2():
    """Theorem 2(3) explicit bound, the Section 2 counterexamples, Proposition R."""
    rng = np.random.default_rng(11)

    def phi(lam, w):
        g = np.sum(w * lam); return 1 - 1 / np.sum(w / (1 - lam)) - g, g
    worst = 0.0
    for trial in range(3000):
        m = rng.integers(2, 40); w = rng.dirichlet(np.ones(m))
        g0 = rng.uniform(-.8, .8) + (1j * rng.uniform(-.3, .3) if trial % 2 else 0)
        z_ = rng.standard_normal(m) + 1j * rng.standard_normal(m); z_ -= np.sum(w * z_)
        z_ *= rng.uniform(.01, .5) * abs(1 - g0) / np.abs(z_).max()
        lam = g0 + z_; ph, g = phi(lam, w); rr = np.abs(lam - g).max()
        worst = max(worst, abs(ph - np.sum(w * (lam - g) ** 2) / (1 - g)) / (6 * rr ** 3 / abs(1 - g) ** 2))
    print(f"E Theorem 2(3): worst |Phi - kappa_2/(1-gamma)| / (6 r^3/|1-gamma|^2) = {worst:.3f}")
    ok(worst <= 1, "E Theorem 2(3) explicit remainder bound")
    lam = np.array([0.9, -0.9, 0.5j, -0.5j]); w = np.array([.1, .1, .4, .4])
    ph, g = phi(lam, w); k2 = np.sum(w * (lam - g) ** 2)
    ok(abs(g) < 1e-15 and r(k2.real, 3) == "−0.038" and r(ph.real, 2) == "0.41", f"E counterexample {k2} {ph}")
    # Theorem 3 needs mean|nu|^2 bounded below: M = diag(1, d^-1/2, ...) gives a non-degenerate c0 law
    sds = {}
    for d in (1000, 100000):
        mvec = np.full(d, d ** -0.5); mvec[0] = 1.0; c0 = []
        for _ in range(300):
            zz = rng.standard_normal(d); zz /= np.linalg.norm(zz)
            c0.append(np.sum(mvec * zz ** 2) / np.sqrt(np.sum(mvec ** 2 * zz ** 2)))
        sds[d] = np.std(c0)
    print(f"E Theorem 3 degenerate example: sd c0 at d=1e3 {sds[1000]:.3f}, d=1e5 {sds[100000]:.3f}")
    ok(sds[100000] > 0.5 * sds[1000] and sds[100000] > 0.05, "E Theorem 3 counterexample does not concentrate")
    # Proposition R: identity, variance over the sphere, subspace mean
    d = 40; J = 0.3 * np.eye(d) + rng.standard_normal((d, d)) / np.sqrt(d) + np.triu(rng.standard_normal((d, d)), 1) * .2
    tau = lambda X: np.trace(X) / d
    H = (J + J.T) / 2; B = H - tau(J) * np.eye(d)
    ok(abs(tau(B @ B) - ((tau(J @ J) + tau(J @ J.T)) / 2 - tau(J) ** 2)) < 1e-12, "E Proposition R tau(B^2) identity")
    Zs = rng.standard_normal((400000, d)); Zs /= np.linalg.norm(Zs, axis=1, keepdims=True)
    alz = np.einsum("ni,ij,nj->n", Zs, J, Zs)
    var_pred = 2 * tau(B @ B) / (d + 2)
    print(f"E Proposition R: mean {alz.mean():.5f} vs tau(J) {tau(J):.5f}; var {alz.var():.6f} vs {var_pred:.6f}")
    ok(abs(alz.mean() - tau(J)) < 4 * np.sqrt(var_pred / len(alz)) and abs(alz.var() / var_pred - 1) < 0.02,
       "E Proposition R mean and variance")
    Ub = np.linalg.qr(rng.standard_normal((d, 10)))[0]
    Xi = rng.standard_normal((200000, 10)); Xi /= np.linalg.norm(Xi, axis=1, keepdims=True); Zs = Xi @ Ub.T
    alz = np.einsum("ni,ij,nj->n", Zs, J, Zs); mp = np.trace(Ub.T @ J @ Ub) / 10
    ok(abs(alz.mean() - mp) < 4 * alz.std() / np.sqrt(len(alz)), f"E Proposition R subspace mean {alz.mean()} {mp}")


def part_f():
    """Synthetic calibration at d = 5280 (Section 5)."""
    here = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(here, "..", "results", "synth_calibration", "calibration.json")
    if not os.path.exists(f):
        FAIL.append("F calibration.json missing (run huginn_addition/calibrate_freeprobe.py)"); return
    c = json.load(open(f))
    ok(set(c) == {"free", "symmetric", "cyclic", "slowmode", "block"}, f"F configs {set(c)}")
    for k, v in c.items():
        ok(v["d"] == 5280 and v["gates"] and v["K"] == 4, f"F {k} d/gates")
    P = lambda k: c[k]["P1_pass"]
    fr = c["free"]
    ok(fr["n"] == 40 and fr["T"] == 48 and all(P("free").values()) and fr["P2_pass"], "F free loop passes P1 and P2")
    mar = sorted(fr["P1_margin"].values()); p2m = (0.10 / fr["P2_logGamma_err"], 0.05 / fr["P2_kappa_err"])
    fl = lambda x: f"{np.floor(10 * x) / 10:.1f}"  # "at least" in the text: round down
    got = (fl(mar[0]), r(mar[-1], 0), fl(min(p2m)))
    print(f"F free margins: P1 min {got[0]} max {got[1]}, P2 min {got[2]}")
    ok(got == FREE_MARGINS, f"F free margins {got}")
    ok(not P("symmetric")["P1a"] and not P("symmetric")["P1b"] and not P("symmetric")["P1c"], "F symmetric fails P1a-c")
    ok(P("cyclic")["P1a"] and not P("cyclic")["P1b"] and not P("cyclic")["P1c"], "F cyclic passes P1a, fails P1b-c")
    ok(all(P("slowmode").values()) and c["slowmode"]["P2_pass"] and c["slowmode"]["T"] == 250,
       "F slow mode invisible (passes P1 and P2)")
    ok(all(P("block").values()) and not c["block"]["P2_pass"], "F block: P1 passes, P2 fails")


FREE_MARGINS = ("2.9", "13", "7.6")  # min P1 margin, max P1 margin, min P2 margin (calibration run 2026-11-07)



CTRL = dict(truth=("ctrlprobe-20261107-144949", "freeprobe-20261107-130520"),
            digits=("ctrlprobe_digits-20261107-144951", "freeprobe_digits-20261107-133750"))
OURO_RUN = "ouro-20261107-150835"
WEIGHTS_RUN = "weights-20261107-145443"
G_WANT = {'truth': {'n': 40, 'tau': ('0.241', '0.240', '0.242'), 'D_task': ('−0.119', '−0.121', '−0.116'), 'Z_task': '−15.1', 'D_C1': '−0.236', 'D_C2': '−0.181', 'D_C3': ('−0.224', '−0.231', '−0.216'), 'rho_C3': ('1.88', '1.80', '1.95'), 'Dpair_C3': ('0.102', '0.098', '0.115'), 'Z_C3': '−28.0', 'readings': ('SHARED', 'SHARED', 'SHARED'), 'below_task': '40/40', 'below_ctrl': '960/960', 'sd': ('0.0079', '0.0081'), 'sens': ('SHARED', 'SHARED', 'SHARED'), 'norm1': ('0.64', '0.62'), 'cos1': ('0.38', '0.03'), 'alpha_C3': '0.018', 'a2': ('0.037', '−0.005'), 'ret': ('0.038', '0.0047'), 'comp': ('0.061', '0.241'), 'perp': '0.002', 'rank_R': (24, 25)},
          'digits': {'n': 20, 'tau': ('0.258', '0.256', '0.259'), 'D_task': ('−0.213', '−0.236', '−0.198'), 'Z_task': '−25.3', 'D_C1': '−0.204', 'D_C2': '−0.234', 'D_C3': ('−0.200', '−0.210', '−0.192'), 'rho_C3': ('0.94', '0.84', '1.03'), 'Dpair_C3': ('−0.013', '−0.030', '−0.005'), 'Z_C3': '−24.3', 'readings': ('SHARED', 'SHARED', 'SHARED'), 'below_task': '200/200', 'below_ctrl': '480/480', 'sd': ('0.0083', '0.0083'), 'sens': None, 'norm1': ('0.67', '0.64'), 'cos1': ('0.38', '0.09'), 'alpha_C3': '0.059', 'a2': ('0.046', '0.002'), 'ret': ('0.042', '0.0063'), 'comp': ('0.067', '0.257'), 'perp': '0.042', 'rank_R': (32, 33)},
          'ouro': {'n': 40, 'tau': ('0.060', '0.057', '0.061'), 'D_task': ('0.097', '0.086', '0.104'), 'Z_task': '5.1', 'D_C1': '0.015', 'D_C2': '0.039', 'D_C3': ('0.022', '0.016', '0.025'), 'D_flip': '0.020', 'coord': ('0.92', 'COORDINATE-WISE'), 'Z_C3': '1.1', 'below_C3': '0.04', 'primary': ('NOT DETECTED', 'trace'), 'primary_k2': 'NOT DETECTED', 'tau_k2': '0.049', 'D_C3_k2': '0.024', 'norm1': ('0.82', '0.99'), 'cos1': ('0.07', '0.08'), 'alpha_C3': '0.081', 'acc': '1.00', 'drift34': '0.17', 'val': 0.0, 'logit': '3e-05', 'flip_diff': ('−0.0004', '−0.006', '0.008')},
          'weights': {'tau': '0.110', 'unembed': '0.101', 'combo': '0.102', 'gWg': ('0.063', '0.049')}}  # values displayed in Sections 6-8 (runs of 2026-11-07)


def part_g(data):
    """Read-space control (ctrlprobe runs) and the second model (Ouro-1.4B) (Sections 6-7)."""
    here = os.path.dirname(os.path.abspath(__file__)); res = os.path.join(here, "..", "results")
    sys.path.insert(0, os.path.join(here, "..", "huginn_addition"))
    import analyze_ctrlprobe as ac
    import analyze_ouro as ao
    import analyze_readspace as ar
    got = {}
    for name, (run, ref) in CTRL.items():
        with contextlib.redirect_stdout(io.StringIO()):
            s = ac.gates_and_eval(os.path.join(res, run), os.path.join(res, ref))
        m = s["main"]; ok(s["gates"]["PASS"], f"G {name} gates")
        rs = ar.run(os.path.join(res, run))
        ok(s["gates"]["probe_alpha_vs_freeprobe"] < 5e-8, f"G {name} old probes reproduce alpha to 5e-8")
        z = np.load(os.path.join(res, run, "ctrlprobe.npz")); fam = z["fam"]
        g3 = z["gram3"][z["done"]]; al = g3[:, :, 0, 1]; P = (fam == 4) | (fam == 5)
        below = al < al[:, P].min(1)[:, None]
        got[name] = dict(
            n=m["n"], tau=tuple(r(v, 3) for v in m["tau"]),
            D_task=tuple(r(v, 3) for v in m["task"]["Delta"]), Z_task=r(m["task"]["Z"][0], 1),
            D_C1=r(m["C1"]["Delta"][0], 3), D_C2=r(m["C2"]["Delta"][0], 3), D_C3=tuple(r(v, 3) for v in m["C3"]["Delta"]),
            rho_C3=tuple(r(v, 2) for v in m["C3"]["rho"]), Dpair_C3=tuple(r(v, 3) for v in m["C3"]["D_pair"]),
            Z_C3=r(m["C3"]["Z"][0], 1), readings=tuple(m[f]["reading"] for f in ("C1", "C2", "C3")),
            below_task=f"{int(below[:, fam == 0].sum())}/{int((fam == 0).sum() * len(al))}",
            below_ctrl=f"{int(below[:, (fam >= 1) & (fam <= 3)].sum())}/{int(((fam >= 1) & (fam <= 3)).sum() * len(al))}",
            sd=(r(m["probe_sd"]["pred_rms"], 4), r(m["probe_sd"]["obs_rms"], 4)),
            sens=tuple(s["sensitivity_drift_le_1e-3"][f]["reading"] for f in ("C1", "C2", "C3")) if "sensitivity_drift_le_1e-3" in s else None,
            norm1=(r(rs["probes"]["norm1"], 2), r(rs["C3"]["norm1"], 2)), cos1=(r(rs["probes"]["cos1"], 2), r(rs["C3"]["cos1"], 2)),
            alpha_C3=r(rs["C3"]["alpha"], 3), a2=(r(rs["probes"]["alpha2"], 3), r(rs["C3"]["alpha2"], 3)),
            ret=(r(rs["retention"]["read_in_R"], 3), r(rs["retention"]["baseline_R"], 4)),
            comp=(r(rs["compression"]["trace_over_dim"]["read"], 3), r(rs["compression"]["trace_over_dim"]["probes"], 3)),
            perp=r(abs(m["C3"]["perp"]["rho"][0] - m["C3"]["rho"][0]), 3), rank_R=tuple(rs["retention"]["rank_R"]))
    with contextlib.redirect_stdout(io.StringIO()):
        o = ao.main(os.path.join(res, OURO_RUN))
    ok(o["gates"]["PASS"], "G Ouro gates")
    e = o["loop_k3"]; e2 = o["loop_k2"]; rot = e["rotation"]
    got["ouro"] = dict(
        n=e["n"], tau=tuple(r(v, 3) for v in e["tau"]), D_task=tuple(r(v, 3) for v in e["task"]["Delta"]), Z_task=r(e["task"]["Z"][0], 1),
        D_C1=r(e["C1"]["Delta"][0], 3), D_C2=r(e["C2"]["Delta"][0], 3), D_C3=tuple(r(v, 3) for v in e["C3"]["Delta"]),
        D_flip=r(e["C3flip"]["Delta"][0], 3), coord=(r(e["coordinate_control"]["ratio"], 2), e["coordinate_control"]["reading"]),
        Z_C3=r(e["C3"]["Z"][0], 1), below_C3=r(e["C3"]["below_all"], 2), primary=(o["PRIMARY"], e["branch"]),
        primary_k2=e2["primary"], tau_k2=r(e2["tau"][0], 3), D_C3_k2=r(e2["C3"]["Delta"][0], 3),
        norm1=(r(rot["probes"]["norm1"], 2), r(rot["C3"]["norm1"], 2)), cos1=(r(rot["probes"]["cos1"], 2), r(rot["C3"]["cos1"], 2)),
        alpha_C3=r(rot["C3"]["alpha"], 3), acc=r(o["behaviour"]["acc"], 2), drift34=r(o["state_drift"]["k3->4"]["rel"], 2),
        val=o["gates"]["val_state_rel_max"], logit=f"{o['gates']['logit_err_max']:.0e}",
        flip_diff=(r(e["C3_minus_C3flip"][0], 4), r(e["C3_minus_C3flip"][1], 3), r(e["C3_minus_C3flip"][2], 3)))
    mins = sum(json.load(open(os.path.join(res, rr, "meta.json")))["runtime_s"] for rr in
               (RUNS["truth"], RUNS["digits"], CTRL["truth"][0], CTRL["digits"][0], OURO_RUN)) / 60
    print(f"   production GPU minutes {mins:.1f}"); ok(round(mins) == 21, f"G GPU minutes {mins}")
    W = json.load(open(os.path.join(res, WEIGHTS_RUN, "summary.json")))
    got["weights"] = dict(tau=r(W["tau_Ws"], 3), unembed=r(W["med"]["unembed"], 3), combo=r(W["med"]["combo"], 3))
    A = np.load(os.path.join(data, "adaptcov-20261106-224906_adaptcov.npz")); Tc = np.load(os.path.join(data, "truthcov-20261106-234613_truthcov.npz"))
    Gd, Wd = A["G"].astype(np.float64), A["W"].astype(np.float64); Gt, Wt = Tc["G"].astype(np.float64), Tc["W"].astype(np.float64)
    got["weights"]["gWg"] = (r(np.median((Gt * Wt).sum(-1) / (Gt ** 2).sum(-1)), 3), r(np.median((Gd * Wd).sum(-1) / (Gd ** 2).sum(-1)), 3))
    for k, v in got.items():
        print(f"== G {k}:", v)
    if G_WANT is None:
        FAIL.append("G_WANT not set"); return
    for k, v in G_WANT.items():
        for kk, vv in v.items():
            ok(got[k][kk] == vv, f"G {k}.{kk}: got {got[k][kk]} want {vv}")


def part_h():
    """Exploratory chain test of the start-invariance mechanism (Section 6, 'A candidate mechanism: an exploratory chain test')."""
    here = os.path.dirname(os.path.abspath(__file__)); res = os.path.join(here, "..", "results")
    sys.path.insert(0, os.path.join(here, "..", "huginn_addition"))
    import analyze_chains as ach
    want = dict(truth=dict(rng=("1.11", "1.33"), below="2.8", nb=(9, 320), p32="5e-05", cos=("0.015", "0.30")),
                digits=dict(rng=("1.22", "1.64"), below="0.9", nb=(3, 320), p32="1.4e-04", cos=("0.095", "0.35")))
    for name in ("truth", "digits"):
        c = ach.chains(os.path.join(res, RUNS[name])); t2 = ach.twostep(os.path.join(res, CTRL[name][0]))
        ok(all(v > 1 for v in c["ratio"][:32]), f"H {name}: readout/probe chain ratio > 1 for all t <= 32")
        got = dict(rng=(r(c["ratio_min_1_32"], 2), r(c["ratio_max_1_32"], 2)), below=f"{100 * c['frac_readout_below_probe_median_1_8']:.1f}", nb=(c["n_below_1_8"], c["n_obs_1_8"]),
                   p32=f"{c['probe_norm_t32']:.0e}" if name == "truth" else f"{c['probe_norm_t32']:.1e}",
                   cos=(r(t2["C3"]["cos12"], 3), r(t2["probes"]["cos12"], 2)))
        print(f"== H {name}:", got)
        for k, v in want[name].items():
            ok(got[k] == v, f"H {name}.{k}: got {got[k]} want {v}")


if __name__ == "__main__":
    part_a()
    part_b()
    part_e()
    part_e2()
    part_f()
    data = os.environ.get("FREELOOP_DATA", "/tmp/hdata")
    need = ["dstar_vjp-20261106-061708_dstar.npz", "truthcov-20261106-234613_truthcov.npz",
            "dglever_train-20261106-080920_dglever.npz", "adaptcov-20261106-224906_adaptcov.npz"]
    ran_c = all(os.path.exists(os.path.join(data, n)) for n in need)
    if ran_c:
        part_c(data)
    else:
        print(f"PART C SKIPPED: Huginn arrays not found in {data} (see module docstring)")
    here = os.path.dirname(os.path.abspath(__file__))
    ran_d = ran_c and all(os.path.exists(os.path.join(here, "..", "results", v, "freeprobe.npz")) for v in RUNS.values())
    if ran_d:
        part_d(data)
        part_d2(data)
        part_g(data)
        part_h()
    else:
        print("PART D SKIPPED: freeprobe arrays not found under results/ (see module docstring)")
    if FAIL:
        print(f"FAILED ({len(FAIL)}):", FAIL[:20]); sys.exit(1)
    print("ALL CHECKS PASSED" + ("" if ran_d else " (Parts C-D incomplete)"))
